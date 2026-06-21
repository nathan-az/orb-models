"""Does a torch orb-v3 training step fit a *full bucket-packed batch*?

The FFD packer (`orb_models.common.dataset`) chose fixed-shape budgets for JAX
finetuning: `edges=44000, nodes=700, graphs=190`. JAX needs
those caps because XLA compiles on shape; torch does NOT pad, but a packed bucket
still hands torch a single disjoint `AtomGraphs` batch whose real content can reach
those caps. Before committing to a torch-vs-JAX throughput comparison we need to
know the obvious failure mode first: **does that bucket OOM the 10 GB card under the
conservative double-backward (create_graph=True), and if not, how fast is it?**

Same model setup as the inference bench (`benchmarks/inference/bench.py`):
real orb-v3-conservative-inf-omat in train mode, confidence head dropped for E/F/S
parity, optional per-stack gradient checkpointing). The difference: instead of ONE
fixed system it packs many rattled Cu fcc supercells into a single bucket that fills
the chosen budgets, then times the full step (loss -> double backward -> optim).

Why dummy Cu rather than the real DB: this question is about *shape* (node/edge/graph
counts), not labels -- targets are random, same as the benchmark. Cu fcc has degree
~78 vs the MPtrj mean ~62 the budget was sized for, so `--max-neighbors` caps the
per-node degree to land the bucket on the real (nodes~700, edges~44000) corner;
`--reps` sets per-system size (reps=2 -> 32 atoms -> ~20 graphs/bucket like the
edge-bound average; reps=1 -> 4 atoms -> the tiny-graph corner that needs graphs=190).

Reported per config: achieved bucket (nodes/edges/graphs), whether it ran or OOM'd,
step time, and throughput as **examples/s (= graphs packed / step)** and atoms/s.

Examples:
    # edge+node-bound worst case (degree capped to the MPtrj ~62 regime):
    python benchmarks/training/check_torch_bucket.py --max-neighbors 62
    # real graph construction (degree ~78 -> edges bind at ~560 nodes):
    python benchmarks/training/check_torch_bucket.py --max-neighbors 120
    # tiny-graph corner that exercises the graphs=190 cap:
    python benchmarks/training/check_torch_bucket.py --reps 1 --max-neighbors 62
    # with per-stack gradient checkpointing:
    python benchmarks/training/check_torch_bucket.py --checkpoint
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the per-stack checkpoint wrapper from the torch trainer + shared bench
# constants so this stays a thin shape-feasibility probe.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as _common  # noqa: E402
from mlflow_util import MlflowLogger, add_mlflow_args  # noqa: E402
from train_torch import _CkptStack  # noqa: E402

from ase.build import bulk  # noqa: E402
from orb_models.common.atoms.batch.graph_batch import AtomGraphs  # noqa: E402
from orb_models.common.training.util import get_optim  # noqa: E402
from orb_models.forcefield import pretrained  # noqa: E402
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Bucket budgets (defaults = the chosen FFD packing budgets).
    p.add_argument("--target-edges", type=int, default=44000, help="Bucket edge cap.")
    p.add_argument("--target-nodes", type=int, default=700, help="Bucket node cap.")
    p.add_argument("--target-graphs", type=int, default=190, help="Bucket graph cap.")
    # Per-system size + degree control (to land on the real dataset's corner).
    p.add_argument("--reps", type=int, default=2,
                   help="fcc Cu supercell reps per system (reps=2 -> 32 atoms).")
    p.add_argument("--max-neighbors", type=int, default=62,
                   help="Per-node neighbour cap (62 ~= MPtrj mean degree; 120 = orb default).")
    # Step config.
    p.add_argument("--checkpoint", action="store_true", help="Per-gnn-stack grad checkpointing.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    add_mlflow_args(p)
    return p.parse_args()


def build_bucket(args, device: torch.device) -> AtomGraphs:
    """First-fit-pack rattled Cu systems into ONE bucket up to the three budgets.

    Greedily add identically-sized rattled supercells (distinct seeds -> distinct
    edge counts, like real packing) until the next system would breach the node,
    edge or graph cap. Returns the disjoint-batched `AtomGraphs`.
    """
    adapter = ForcefieldAtomsAdapter(radius=_common.RADIUS, max_num_neighbors=args.max_neighbors)
    per_graphs: list[AtomGraphs] = []
    n_nodes = n_edges = 0
    seed = 0
    while True:
        atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat(args.reps)
        atoms.rattle(0.1, seed=seed)
        seed += 1
        g = adapter.from_ase_atoms(
            atoms, device=device, max_num_neighbors=args.max_neighbors,
        )
        gn, ge = int(g.n_node.sum()), int(g.n_edge.sum())
        # Stop before exceeding any cap (need >=1 system).
        if per_graphs and (
            n_nodes + gn > args.target_nodes
            or n_edges + ge > args.target_edges
            or len(per_graphs) + 1 > args.target_graphs
        ):
            break
        per_graphs.append(g)
        n_nodes += gn
        n_edges += ge

    batch = per_graphs[0] if len(per_graphs) == 1 else AtomGraphs.batch(per_graphs)
    # Attach shared random targets (values irrelevant -- this is a shape/memory probe).
    n_atoms = int(batch.n_node.sum())
    n_graphs = len(per_graphs)
    raw = _common.make_targets(n_atoms, n_graphs)
    batch.node_targets["forces"] = torch.from_numpy(raw["forces"]).to(device=device)
    batch.system_targets["energy"] = torch.from_numpy(raw["energy"]).to(device=device)
    batch.system_targets["stress"] = torch.from_numpy(raw["stress"]).to(device=device)
    return batch


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    print("Loading orb-v3-conservative-inf-omat (train mode)...")
    model, _ = pretrained.orb_v3_conservative_inf_omat(
        device=device, precision="float32-high", compile=False, train=True
    )
    if "confidence" in model.heads:
        del model.heads["confidence"]
        model.extra_properties = [p for p in model.extra_properties if p != "confidence"]
    if args.checkpoint:
        stacks = model.model.gnn_stacks
        model.model.gnn_stacks = torch.nn.ModuleList([_CkptStack(s) for s in stacks])
    model.train()

    print(f"Packing bucket up to edges={args.target_edges} nodes={args.target_nodes} "
          f"graphs={args.target_graphs} (reps={args.reps}, max_neighbors={args.max_neighbors})...")
    batch = build_bucket(args, device)
    n_atoms, n_edges = int(batch.n_node.sum()), int(batch.n_edge.sum())
    n_graphs = int(batch.n_node.numel())
    print(f"Bucket built: {n_graphs} graphs, {n_atoms} atoms, {n_edges} edges "
          f"(degree {n_edges / n_atoms:.1f}) on {device}")

    # MLflow: log the full setup. torch doesn't pad/chunk/checkpoint-remat the way
    # JAX does, but we log those keys as "n/a" so runs line up across frameworks.
    logger = MlflowLogger(args, run_name="torch_bucket", params={
        "method": "torch",
        "model": "orb_v3_conservative_inf_omat",
        "target_edges": args.target_edges,
        "target_nodes": args.target_nodes,
        "target_graphs": args.target_graphs,
        "reps": args.reps,
        "max_neighbors": args.max_neighbors,
        "checkpoint": args.checkpoint,
        "ckpt_mode": "stack" if args.checkpoint else "n/a",
        "chunk": "n/a",
        "chunk_encoder": "n/a",
        "padded": False,
        "warmup": args.warmup,
        "steps": args.steps,
        "device": args.device,
        "radius": _common.RADIUS,
        "lr": _common.LR,
        "total_steps": _common.TOTAL_STEPS,
        "precision": _common.PRECISION,
        "loss_weights": _common.LOSS_WEIGHTS,
        "bucket_graphs": n_graphs,
        "bucket_atoms": n_atoms,
        "bucket_edges": n_edges,
    })

    optimizer, scheduler = get_optim(_common.LR, _common.TOTAL_STEPS, model)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        out = model.loss(batch)
        out.loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return float(out.loss.detach())

    try:
        print(f"Warm-up ({args.warmup} steps)...")
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()

        print(f"Timing ({args.steps} steps)...")
        times: list[float] = []
        peaks: list[int] = []
        for _ in range(args.steps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            peaks.append(torch.cuda.max_memory_allocated())
    except torch.cuda.OutOfMemoryError as e:
        print("\n=== RESULT: OOM ===")
        print(f"  bucket        : {n_graphs} graphs, {n_atoms} atoms, {n_edges} edges")
        print(f"  checkpoint    : {args.checkpoint}")
        print(f"  error         : {str(e).splitlines()[0]}")
        print(f"  peak before   : {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB "
              f"of {torch.cuda.get_device_properties(device).total_memory / 2**20:,.0f} MiB")
        logger.set_tags({"result": "OOM", "error": str(e).splitlines()[0]})
        logger.log_metrics({"peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20})
        logger.finish(status="FAILED")
        sys.exit(1)

    t = np.asarray(times) * 1e3
    median_s = float(np.median(times))
    for i, (ms, pk) in enumerate(zip(t, peaks)):
        logger.log_metrics({"step_ms": float(ms), "step_peak_mib": pk / 2**20}, step=i)
    logger.set_tags({"result": "OK"})
    logger.log_metrics({
        "step_ms_median": float(np.median(t)),
        "step_ms_p10": float(np.percentile(t, 10)),
        "step_ms_p90": float(np.percentile(t, 90)),
        "examples_per_s": n_graphs / median_s,
        "atoms_per_s": n_atoms / median_s,
        "peak_mem_mib": max(peaks) / 2**20,
    })
    logger.finish()
    print("\n=== RESULT: OK (fits) ===")
    print(f"  bucket        : {n_graphs} graphs, {n_atoms} atoms, {n_edges} edges "
          f"(degree {n_edges / n_atoms:.1f})")
    print(f"  checkpoint    : {args.checkpoint}")
    print(f"  step time     : median {np.median(t):.1f} ms "
          f"(p10 {np.percentile(t, 10):.1f} / p90 {np.percentile(t, 90):.1f})")
    print(f"  throughput    : {n_graphs / median_s:,.1f} examples/s  "
          f"({n_atoms / median_s:,.0f} atoms/s)")
    print(f"  peak memory   : {max(peaks) / 2**20:,.1f} MiB of "
          f"{torch.cuda.get_device_properties(device).total_memory / 2**20:,.0f} MiB")


if __name__ == "__main__":
    main()
