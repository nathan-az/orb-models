"""Finetune the torch orb-v3 conservative forcefield -- the baseline for the
torch-vs-JAX comparison (`train_jax.py` is the experimental arm).

Same bucketed data path as the JAX trainer (FFD-packed buckets, size-decorrelated;
see `orb_models.common.dataset.bucket_sampler`), but torch runs each bucket
*unpadded* -- there is no XLA shape constraint, so no padding. The bucket caps
double as a memory guard under the conservative double-backward: raise them on a
larger GPU, or use `grad_accum_steps` to emulate a larger effective batch from
several small buckets.

torch's force/stress come from autograd of the energy (one conservative
double-backward via `create_graph=True`, inside `model.loss`); there is a single
gradient path, so unlike JAX there is no `method` to swap -- this is the fixed
reference. Optional per-gnn-stack activation checkpointing is available (`--`-free
OmegaConf: `checkpoint=true`), though it barely helps the backward-over-backward.

Config is a frozen dataclass populated by OmegaConf -- override with `key=value`
and/or `config=path.yaml`:

    python benchmarks/training/train_torch.py \
        data_path=/path/to/ase_sqlite.db \
        edge_budget=16000 node_budget=260 graph_budget=8 degree_estimate=62 \
        max_steps=2000 lr=3e-4 grad_accum_steps=1
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from types import SimpleNamespace

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from orb_models.common.dataset import augmentations, property_definitions
from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.common.dataset.bucket_sampler import (
    BucketBatchSampler,
    BucketCollator,
    read_natoms,
)
from orb_models.common.dataset.loaders import worker_init_fn
from orb_models.common.training.util import get_optim, set_torch_precision
from orb_models.forcefield import pretrained

from mlflow_util import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MlflowLogger


@dataclass
class TorchTrainConfig:
    # --- data ---------------------------------------------------------------
    data_path: str = "???"  # required: path to an ASE SQLite dataset (set on the CLI)
    dataset_name: str = "mp-traj"
    num_workers: int = 4
    # --- model --------------------------------------------------------------
    base_model: str = "orb_v3_conservative_inf_omat"
    no_stress: bool = False
    # Precision lever for the precision-matched torch-vs-JAX comparison.
    # "float32-highest" = true fp32 matmuls (no TF32), the mode that aligns with
    # JAX matmul_precision=highest; "float32-high" = TF32 (~2x, matches JAX
    # matmul_precision=tensorfloat32). Threaded into `set_torch_precision` + the
    # pretrained loader so both default dtype and matmul precision follow it.
    precision: str = "float32-high"
    # --- bucket budgets + packing (defaults sized for a memory-constrained GPU) ----
    edge_budget: int = 16000  # hard cap AND FFD edge cap (estimate space)
    node_budget: int = 260
    graph_budget: int = 16
    degree_estimate: float = 62.0
    chunk_size: int = 1000
    # --- optimisation -------------------------------------------------------
    max_steps: int = 1000
    lr: float = 3e-4
    energy_loss_weight: float = 1.0
    forces_loss_weight: float = 10.0
    stress_loss_weight: float = 1.0
    grad_accum_steps: int = 1
    gradient_clip_val: float = 0.0  # 0 disables
    checkpoint: bool = False  # per-gnn-stack activation checkpointing
    # --- bookkeeping --------------------------------------------------------
    seed: int = 1234
    device: str = "cuda"
    log_every: int = 10
    warmup_steps: int = 5  # excluded from median step_time + peak-mem measurement
    out_dir: str = "ckpts"
    save_every_steps: int = 0
    config: str = ""
    # --- mlflow -------------------------------------------------------------
    enable_mlflow: bool = True
    mlflow_uri: str = DEFAULT_TRACKING_URI
    mlflow_experiment: str = DEFAULT_EXPERIMENT
    run_name: str | None = None


class _CkptStack(torch.nn.Module):
    """Wrap one gnn stack so its activations are rematerialised in the backward.

    `use_reentrant=False` supports the conservative double-backward (create_graph).
    """

    def __init__(self, stack: torch.nn.Module):
        super().__init__()
        self.stack = stack

    def forward(self, *args, **kwargs):
        return checkpoint(self.stack, *args, use_reentrant=False, **kwargs)


def load_config() -> TorchTrainConfig:
    base = OmegaConf.structured(TorchTrainConfig)
    cli = OmegaConf.from_cli()
    if cli.get("config"):
        base = OmegaConf.merge(base, OmegaConf.load(cli.config))
    cfg = OmegaConf.merge(base, cli)
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def build_loader(cfg: TorchTrainConfig, atoms_adapter, *, has_stress: bool):
    graph_targets = ["energy", "stress"] if has_stress else ["energy"]
    target_config = property_definitions.instantiate_property_config(
        {"graph": graph_targets, "node": ["forces"]}
    )
    dataset = AseSqliteDataset(
        cfg.dataset_name,
        cfg.data_path,
        atoms_adapter=atoms_adapter,
        target_config=target_config,
        augmentations=[augmentations.rotate_randomly],
    )
    natoms = read_natoms(cfg.data_path)
    print(f"Dataset: {len(dataset)} structures from {cfg.data_path}")

    budgets = {
        "edges": cfg.edge_budget,
        "nodes": cfg.node_budget,
        "graphs": cfg.graph_budget,
    }
    sampler = BucketBatchSampler(
        natoms,
        budgets,
        degree_estimate=cfg.degree_estimate,
        chunk_size=cfg.chunk_size,
        shuffle=True,
        seed=cfg.seed,
    )
    collator = BucketCollator(
        atoms_adapter.batch,
        n_max=cfg.node_budget,
        e_max=cfg.edge_budget,
        g_max=cfg.graph_budget,
    )
    loader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collator,
        batch_sampler=sampler,
        timeout=10 * 60 if cfg.num_workers > 0 else 0,
    )
    return loader, sampler, collator


def micro_batch_counts(micro: list) -> tuple[int, int, int]:
    """Sum real graphs, atoms, edges across micro-batches in one optimiser step."""
    graphs = atoms = edges = 0
    for batch in micro:
        graphs += int(batch.n_node.numel())
        atoms += int(batch.n_node.sum())
        edges += int(batch.n_edge.sum())
    return graphs, atoms, edges


def main() -> None:
    cfg = load_config()
    has_stress = not cfg.no_stress
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    set_torch_precision(cfg.precision)

    # 1. Pretrained model in train mode; drop the confidence head for E/F/S parity
    #    with the JAX port (which has no confidence head).
    print(f"Loading pretrained torch model: {cfg.base_model} (train mode)")
    model, atoms_adapter = getattr(pretrained, cfg.base_model)(
        device=device, precision=cfg.precision, compile=False, train=True
    )
    if "confidence" in model.heads:
        del model.heads["confidence"]
        model.extra_properties = [p for p in model.extra_properties if p != "confidence"]
    if cfg.checkpoint:
        stacks = model.model.gnn_stacks
        model.model.gnn_stacks = torch.nn.ModuleList([_CkptStack(s) for s in stacks])

    # Loss weights + stress toggle (mirror finetune.py).
    model.loss_weights.update(
        {
            "energy": cfg.energy_loss_weight,
            "forces": cfg.forces_loss_weight,
            "stress": cfg.stress_loss_weight,
        }
    )
    if has_stress:
        model.enable_stress()
    elif model.has_stress:
        model.disable_stress()
    model.to(device=device)
    model.train()

    # 2. Data.
    loader, sampler, collator = build_loader(cfg, atoms_adapter, has_stress=has_stress)

    # 3. Optimiser + schedule (matches JAX make_optimizer; sized from max_steps).
    optimizer, scheduler = get_optim(cfg.lr, cfg.max_steps, model)
    accum = max(1, cfg.grad_accum_steps)
    clip = cfg.gradient_clip_val or None

    os.makedirs(cfg.out_dir, exist_ok=True)
    print(
        f"Training baseline on {device}; bucket caps="
        f"({cfg.node_budget} nodes / {cfg.edge_budget} edges / {cfg.graph_budget} graphs) "
        f"accum={accum} checkpoint={cfg.checkpoint}"
    )

    mlflow = MlflowLogger(
        SimpleNamespace(
            enable_mlflow=cfg.enable_mlflow,
            run_name=cfg.run_name,
            mlflow_experiment=cfg.mlflow_experiment,
            mlflow_uri=cfg.mlflow_uri,
        ),
        run_name=cfg.run_name or "torch",
        params={
            "framework": "torch",
            "base_model": cfg.base_model,
            "precision": cfg.precision,
            "edge_budget": cfg.edge_budget,
            "node_budget": cfg.node_budget,
            "graph_budget": cfg.graph_budget,
            "degree_estimate": cfg.degree_estimate,
            "grad_accum_steps": accum,
            "gradient_clip_val": cfg.gradient_clip_val,
            "checkpoint": cfg.checkpoint,
            "lr": cfg.lr,
            "max_steps": cfg.max_steps,
            "energy_loss_weight": cfg.energy_loss_weight,
            "forces_loss_weight": cfg.forces_loss_weight,
            "stress_loss_weight": cfg.stress_loss_weight,
            "no_stress": cfg.no_stress,
            "seed": cfg.seed,
            "device": str(device),
        },
    )

    # 4. Train loop driven by max_steps; grad accumulation over `accum` buckets.
    n_steps = 0
    running = 0.0
    running_step_time = 0.0
    step_times: list[float] = []
    epoch = 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    micro: list = []
    accum_loss = 0.0
    step_t0: float | None = None
    while n_steps < cfg.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if batch is None:  # collator dropped the whole bucket
                continue
            if step_t0 is None:
                step_t0 = time.perf_counter()
            batch = batch.to(device)
            out = model.loss(batch)
            loss = out.loss
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            (loss / accum).backward()
            micro.append(batch)
            accum_loss += float(loss.detach())
            if len(micro) < accum:
                continue

            n_graphs, n_atoms, n_edges = micro_batch_counts(micro)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip if clip is not None else float("inf")
                )
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_loss = accum_loss / accum
            # optimizer.step() dispatches async; sync so step_time captures the
            # optimizer-update tail too (matches jax's block_until_ready).
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - step_t0
            step_t0 = None
            running += step_loss
            running_step_time += step_time
            step_times.append(step_time)
            n_steps += 1
            micro = []
            accum_loss = 0.0
            # Drop warmup (cudnn autotune / allocator growth) from the peak-mem
            # window so it reflects steady-state training.
            if n_steps == cfg.warmup_steps and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if n_steps % cfg.log_every == 0:
                print(
                    f"  step {n_steps}/{cfg.max_steps}  "
                    f"loss={step_loss:.4f} (avg {running / n_steps:.4f})  "
                    f"grad_norm={grad_norm:.4f}  "
                    f"step_time={step_time:.3f}s (avg {running_step_time / n_steps:.3f}s)  "
                    f"graphs={n_graphs} atoms={n_atoms} edges={n_edges}"
                )
            # no good reason not to log to tracking server every step
            mlflow.log_metrics(
                {
                    "loss": step_loss,
                    "grad_norm": grad_norm,
                    "step_time": step_time,
                    "graphs_in_step": n_graphs,
                    "atoms_in_step": n_atoms,
                    "edges_in_step": n_edges,
                },
                step=n_steps,
            )
            if cfg.save_every_steps and n_steps % cfg.save_every_steps == 0:
                path = os.path.join(cfg.out_dir, f"torch_finetune_step{n_steps}.ckpt")
                torch.save(model.state_dict(), path)
                print(f"  saved {path}")
            if n_steps >= cfg.max_steps:
                break
        epoch += 1

    dt = time.perf_counter() - t0
    # Median over post-warmup steps: robust to the slow first step(s).
    timed = step_times[cfg.warmup_steps :] or step_times
    median_step = statistics.median(timed) if timed else 0.0
    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else 0.0
    )
    mean_step_time = sum(timed) / len(timed)
    first_step_time = step_times[0]
    print(
        f"Done {n_steps} steps in {dt:.1f}s ({n_steps / dt:.2f} steps/s)  "
        f"mean_loss={running / max(n_steps, 1):.4f}  "
        f"median_step_time={median_step:.3f}s (over {len(timed)} steps)  "
        f"mean_step_time={mean_step_time:.3f}s  "
        f"first_step_time={first_step_time:.3f}s  "
        f"peak_mem={peak_mem_gb:.2f}GB  "
    )
    if collator.n_dropped_members:
        print(
            f"NOTE: collator trimmed {collator.n_dropped_members} member(s) across "
            f"{collator.n_trimmed_buckets} bucket(s) for exceeding the edge estimate."
        )
    final = os.path.join(cfg.out_dir, "torch_finetune_final.ckpt")
    torch.save(model.state_dict(), final)
    print(f"Saved {final}")
    mlflow.log_metrics(
        {
            "mean_loss": running / max(n_steps, 1),
            "first_step_time": first_step_time,
            "mean_step_time": mean_step_time,
            "median_step_time": median_step,
            "peak_mem_gb": peak_mem_gb,
        },
        step=n_steps,
    )
    mlflow.finish()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
