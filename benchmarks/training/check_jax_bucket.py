"""JAX counterpart of `check_torch_bucket.py`: does a JAX orb-v3 train step fit a
full bucket-padded batch, for both grad paths (jvp + reverse)?

Same probe, same bucket parameters as the torch check -- but where torch runs the
packed batch unpadded, JAX must pad it to the FIXED budget shape so XLA compiles the
step once (`to_padded_numpy`; see train_jax). So the compiled shape here is exactly
the chosen FFD bucket `(nodes=700, edges=44000, graphs=190 -> g_pad=191)` regardless
of how much real content we packed -- that padded shape is what determines memory and
step time. Throughput is still reported on the REAL packed graphs (examples/s =
graphs-packed / step), the same denominator as torch, so the two are comparable.

Uses: `build_orb_v3_conservative_jax` (random
weights -- this is shape/throughput, not parity), `convert_to_chunked` (the edge-axis
chunking WIDTH lever + activation-checkpointing DEPTH lever), and the real
`make_train_step` with `grad_fn` = jvp (forward-over-reverse) or reverse
(reverse-over-reverse). The bucket is packed identically to the torch probe (rattled
Cu fcc, `--max-neighbors 62` to hit the MPtrj ~62 degree corner) then padded.

Examples:
    # jvp, no memory levers, at the full bucket:
    python benchmarks/training/check_jax_bucket.py --method jax_jvp
    # reverse path:
    python benchmarks/training/check_jax_bucket.py --method jax_reverse
    # jvp with edge chunking (the lever that pushes max edges):
    python benchmarks/training/check_jax_bucket.py --method jax_jvp --chunk 4096 --chunk-encoder
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Must precede `import jax`: don't let XLA pre-grab the GPU so peak_bytes_in_use is real.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as _common  # noqa: E402
from mlflow_util import MlflowLogger, add_mlflow_args  # noqa: E402

import jax  # noqa: E402
from ase.build import bulk  # noqa: E402
from orb_models.common.atoms.batch.graph_batch import AtomGraphs  # noqa: E402
from orb_models.common.atoms.jax import graph_batch as jgb  # noqa: E402
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter  # noqa: E402
from orb_models.forcefield.models.jax.optimisation import (  # noqa: E402
    CHECKPOINTERS,
    convert_to_chunked,
)
from orb_models.forcefield.models.jax.conservative_regressor import (  # noqa: E402
    compute_grads_jvp,
    compute_grads_reverse,
)
from orb_models.forcefield.models.jax.port_weights import (  # noqa: E402
    build_orb_v3_conservative_jax,
)
from orb_models.forcefield.models.jax.train import (  # noqa: E402
    init_opt_state,
    make_optimizer,
    make_train_step,
)

_GRAD_FNS = {"jax_reverse": compute_grads_reverse, "jax_jvp": compute_grads_jvp}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method", default="jax_jvp", choices=sorted(_GRAD_FNS),
                   help="Gradient path: jax_jvp (forward-over-reverse) or jax_reverse.")
    # Bucket budgets (defaults = the chosen FFD packing budgets) -- the PADDED shape.
    p.add_argument("--target-edges", type=int, default=44000, help="Bucket edge cap (e_pad).")
    p.add_argument("--target-nodes", type=int, default=700, help="Bucket node cap (n_pad).")
    p.add_argument("--target-graphs", type=int, default=190,
                   help="Bucket graph cap; g_pad = this + 1 absorbing padding graph.")
    # Per-system size + degree control (must match the torch probe to compare).
    p.add_argument("--reps", type=int, default=2,
                   help="fcc Cu supercell reps per system (reps=2 -> 32 atoms).")
    p.add_argument("--max-neighbors", type=int, default=62,
                   help="Per-node neighbour cap (62 ~= MPtrj mean degree; 120 = orb default).")
    # Memory levers (forwarded to convert_to_chunked).
    p.add_argument("--chunk", type=int, default=0, help="Edge-axis tile width (0 disables).")
    p.add_argument("--chunk-encoder", action="store_true", help="Also tile the encoder edge_fn.")
    p.add_argument("--checkpoint", action="store_true", help="Activation rematerialisation.")
    p.add_argument("--ckpt-mode", default="stack", help="Checkpoint policy (see CHECKPOINTERS).")
    # Step config.
    p.add_argument("--warmup", type=int, default=3, help="Warm-up steps (incl. one-time compile).")
    p.add_argument("--steps", type=int, default=10)
    add_mlflow_args(p)
    return p.parse_args()


def build_packed_bucket(args):
    """Pack rattled Cu systems on CPU up to the budgets, then snapshot to JAX.

    Identical packing policy to `check_torch_bucket.build_bucket` so the REAL content
    matches; returns (torch_graph, n_real_graphs, n_atoms, n_edges). The caller pads
    it to the fixed bucket via `to_padded_numpy`.
    """
    adapter = ForcefieldAtomsAdapter(radius=_common.RADIUS, max_num_neighbors=args.max_neighbors)
    per_graphs: list[AtomGraphs] = []
    n_nodes = n_edges = 0
    seed = 0
    while True:
        atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat(args.reps)
        atoms.rattle(0.1, seed=seed)
        seed += 1
        g = adapter.from_ase_atoms(atoms, device="cpu", max_num_neighbors=args.max_neighbors)
        gn, ge = int(g.n_node.sum()), int(g.n_edge.sum())
        if per_graphs and (
            n_nodes + gn > args.target_nodes
            or n_edges + ge > args.target_edges
            or len(per_graphs) + 1 > args.target_graphs
        ):
            break
        per_graphs.append(g)
        n_nodes += gn
        n_edges += ge

    torch_graph = per_graphs[0] if len(per_graphs) == 1 else AtomGraphs.batch(per_graphs)
    n_atoms = int(torch_graph.n_node.sum())
    n_real_edges = int(torch_graph.n_edge.sum())
    return torch_graph, len(per_graphs), n_atoms, n_real_edges


def main() -> None:
    args = parse_args()
    if args.checkpoint and args.ckpt_mode not in CHECKPOINTERS:
        raise SystemExit(f"--ckpt-mode must be one of {sorted(CHECKPOINTERS)}.")
    device = jax.devices()[0]

    print(f"Packing bucket up to edges={args.target_edges} nodes={args.target_nodes} "
          f"graphs={args.target_graphs} (reps={args.reps}, max_neighbors={args.max_neighbors})...")
    torch_graph, n_graphs, n_atoms, n_real_edges = build_packed_bucket(args)

    # Attach synthetic targets to the torch batch, then convert + pad to the FIXED
    # budget shape (what XLA compiles on) in one numpy pass. g_pad = graphs + 1.
    g_pad = args.target_graphs + 1
    raw = _common.make_targets(n_atoms, n_graphs)
    torch_graph.system_targets["energy"] = torch.tensor(raw["energy"])
    torch_graph.node_targets["forces"] = torch.tensor(raw["forces"])
    torch_graph.system_targets["stress"] = torch.tensor(raw["stress"])
    graph_np, targets_np = jgb.to_padded_numpy(
        torch_graph, args.target_nodes, args.target_edges, g_pad, has_stress=True
    )
    graph = jax.device_put(graph_np)
    targets = jax.device_put(targets_np)
    print(f"Bucket built: {n_graphs} real graphs, {n_atoms} atoms, {n_real_edges} edges "
          f"(degree {n_real_edges / n_atoms:.1f})  ->  padded to "
          f"{args.target_nodes} nodes / {args.target_edges} edges / {g_pad} graphs on {device}")

    # MLflow: log the full JAX setup -- including the chunk/checkpoint levers and the
    # padded compile shape that don't apply to torch (logged as their own keys so the
    # two frameworks' runs stay comparable in the same experiment).
    logger = MlflowLogger(args, run_name=args.method, params={
        "method": args.method,
        "model": "orb_v3_conservative",
        "target_edges": args.target_edges,
        "target_nodes": args.target_nodes,
        "target_graphs": args.target_graphs,
        "reps": args.reps,
        "max_neighbors": args.max_neighbors,
        "chunk": args.chunk,
        "chunk_encoder": args.chunk_encoder,
        "checkpoint": args.checkpoint,
        "ckpt_mode": args.ckpt_mode if args.checkpoint else "n/a",
        "padded": True,
        "pad_nodes": args.target_nodes,
        "pad_edges": args.target_edges,
        "pad_graphs": g_pad,
        "warmup": args.warmup,
        "steps": args.steps,
        "device": str(device),
        "radius": _common.RADIUS,
        "lr": _common.LR,
        "total_steps": _common.TOTAL_STEPS,
        "loss_weights": _common.LOSS_WEIGHTS,
        "bucket_graphs": n_graphs,
        "bucket_atoms": n_atoms,
        "bucket_edges": n_real_edges,
    })

    model = build_orb_v3_conservative_jax(key=jax.random.PRNGKey(0))
    model = convert_to_chunked(
        model, chunk=args.chunk, chunk_encoder=args.chunk_encoder,
        checkpoint=args.checkpoint, ckpt_mode=args.ckpt_mode,
    )
    optimizer = make_optimizer(lr=_common.LR, total_steps=_common.TOTAL_STEPS)
    opt_state = init_opt_state(model, optimizer)
    step = make_train_step(optimizer, _common.LOSS_WEIGHTS, grad_fn=_GRAD_FNS[args.method])

    lever = f"chunk={args.chunk}{'+enc' if args.chunk_encoder else ''}" \
            f"{' ckpt-' + args.ckpt_mode if args.checkpoint else ''}"
    try:
        print(f"Warm-up ({args.warmup} steps, incl. one-time jit compile)... [{args.method} {lever}]")
        t0 = time.perf_counter()
        for _ in range(args.warmup):
            model, opt_state, breakdown = step(model, opt_state, graph, targets)
        jax.block_until_ready((model, opt_state))
        warmup_s = time.perf_counter() - t0

        print(f"Timing ({args.steps} steps)...")
        times: list[float] = []
        for _ in range(args.steps):
            s = time.perf_counter()
            model, opt_state, breakdown = step(model, opt_state, graph, targets)
            jax.block_until_ready((model, opt_state))
            times.append(time.perf_counter() - s)
    except Exception as e:  # XLA raises a generic RuntimeError (ResourceExhausted) on OOM
        msg = str(e).splitlines()[0]
        oom = "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()
        print(f"\n=== RESULT: {'OOM' if oom else 'ERROR'} ===")
        print(f"  method        : {args.method}  [{lever}]")
        print(f"  padded shape  : {args.target_nodes} nodes / {args.target_edges} edges / {g_pad} graphs")
        print(f"  error         : {msg}")
        logger.set_tags({"result": "OOM" if oom else "ERROR", "error": msg})
        logger.finish(status="FAILED")
        sys.exit(1)

    t = np.asarray(times) * 1e3
    median_s = float(np.median(times))
    peak_mem = int(device.memory_stats()["peak_bytes_in_use"])
    total_mem = int(device.memory_stats().get("bytes_limit", 0))
    for i, ms in enumerate(t):
        logger.log_metrics({"step_ms": float(ms)}, step=i)
    logger.set_tags({"result": "OK"})
    logger.log_metrics({
        "step_ms_median": float(np.median(t)),
        "step_ms_p10": float(np.percentile(t, 10)),
        "step_ms_p90": float(np.percentile(t, 90)),
        "examples_per_s": n_graphs / median_s,
        "atoms_per_s": n_atoms / median_s,
        "peak_mem_mib": peak_mem / 2**20,
        "warmup_total_s": warmup_s,
    })
    logger.finish()
    print("\n=== RESULT: OK (fits) ===")
    print(f"  method        : {args.method}  [{lever}]")
    print(f"  real bucket   : {n_graphs} graphs, {n_atoms} atoms, {n_real_edges} edges")
    print(f"  padded shape  : {args.target_nodes} nodes / {args.target_edges} edges / {g_pad} graphs")
    print(f"  warm-up       : {warmup_s:.1f} s total ({warmup_s / max(args.warmup, 1):.1f} s/step incl. compile)")
    print(f"  step time     : median {np.median(t):.1f} ms "
          f"(p10 {np.percentile(t, 10):.1f} / p90 {np.percentile(t, 90):.1f})")
    print(f"  throughput    : {n_graphs / median_s:,.1f} examples/s  "
          f"({n_atoms / median_s:,.0f} atoms/s on real content)")
    print(f"  peak memory   : {peak_mem / 2**20:,.1f} MiB"
          + (f" of {total_mem / 2**20:,.0f} MiB" if total_mem else ""))


if __name__ == "__main__":
    main()
