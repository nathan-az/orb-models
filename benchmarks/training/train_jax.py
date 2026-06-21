"""Finetune the JAX orb-v3 conservative forcefield on an ASE SQLite dataset.

This is the JAX arm of the torch-vs-JAX comparison (`train_torch.py` is the
baseline). It is the experimental surface: swap the gradient *method* (jvp /
reverse) and the memory *optimisations* (edge chunking / activation checkpointing)
from `...models.jax.optimisation`.

Pipeline per step (the bucketed data path lives in the library):

    BucketBatchSampler  -> shuffle index, chunk, FFD-pack to budgets, shuffle bins
      -> DataLoader + JaxBucketCollator(adapter.batch)  disjoint torch AtomGraphs ->
         graph_batch.to_padded_numpy   FIXED-bucket padded numpy arrays (in workers)
      -> jax.device_put                arrays + targets onto device (main thread)
      -> make_train_step(...) / make_accum_train_step

Why a fixed bucket: `eqx.filter_jit` keys on array shapes, so without padding to
one bucket XLA would recompile every step. The packer batches by *budget* (not a
fixed system count) so each bucket fills the bucket tightly; `to_padded_numpy` tops
it up and the loss masks the padding out.

Config is a frozen dataclass populated by OmegaConf -- override on the CLI with
`key=value` (dotlist) and/or `config=path.yaml`:

    python benchmarks/training/train_jax.py \
        data_path=/path/to/ase_sqlite.db \
        edge_budget=44000 node_budget=700 graph_budget=190 degree_estimate=62 \
        method=jax_jvp chunk=4096 chunk_encoder=true max_steps=2000 lr=3e-4
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from types import SimpleNamespace

# Must precede `import jax`: don't let XLA pre-grab 75% of the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from omegaconf import OmegaConf  # noqa: E402

import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from orb_models.common.atoms.jax import graph_batch as jgb  # noqa: E402
from orb_models.common.dataset import augmentations, property_definitions  # noqa: E402
from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset  # noqa: E402
from orb_models.common.dataset.bucket_sampler import (  # noqa: E402
    BucketBatchSampler,
    BucketCollator,
    read_natoms,
)
from orb_models.common.dataset.loaders import worker_init_fn  # noqa: E402
from orb_models.forcefield import pretrained  # noqa: E402
from orb_models.forcefield.models.jax.conservative_regressor import (  # noqa: E402
    compute_grads_jvp,
    compute_grads_reverse,
)
from orb_models.forcefield.models.jax.optimisation import (  # noqa: E402
    CHECKPOINTERS,
    convert_to_chunked,
)
from orb_models.forcefield.models.jax.port_weights import (  # noqa: E402
    load_orb_v3_conservative_into_jax,
)
from orb_models.forcefield.models.jax.train import (  # noqa: E402
    init_opt_state,
    make_accum_train_step,
    make_optimizer,
    make_train_step,
)
from mlflow_util import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MlflowLogger  # noqa: E402

_GRAD_FNS = {"jax_reverse": compute_grads_reverse, "jax_jvp": compute_grads_jvp}


@dataclass
class JaxTrainConfig:
    # --- data ---------------------------------------------------------------
    data_path: str = "???"  # required: path to an ASE SQLite dataset (set on the CLI)
    dataset_name: str = "mp-traj"
    num_workers: int = 4
    # --- model --------------------------------------------------------------
    base_model: str = "orb_v3_conservative_inf_omat"
    no_stress: bool = False  # train on energy+forces only
    # fp32 matmul mode, pinned so JAX doesn't silently run XLA's (TF32) default
    # while torch is on float32-highest. "highest" = true fp32 (matches torch
    # precision=float32-highest); "tensorfloat32"/"high" = TF32 (matches torch
    # precision=float32-high). Keep this aligned with the torch arm.
    matmul_precision: str = "highest"
    # --- bucket (the fixed compile shape) + packing -------------------------
    edge_budget: int = 44000  # e_pad; FFD edge cap (estimate space) AND hard cap
    node_budget: int = 700  # n_pad
    graph_budget: int = 190  # g_pad = graph_budget + 1
    degree_estimate: float = 62.0  # edges ~= n_atoms * this (MPtrj mean ~62)
    dataset_chunk_size: int = 1000  # streaming shuffle-buffer window for FFD
    # --- optimisation -------------------------------------------------------
    max_steps: int = 1000
    lr: float = 3e-4
    energy_loss_weight: float = 1.0
    forces_loss_weight: float = 10.0
    stress_loss_weight: float = 1.0
    method: str = "jax_jvp"  # jax_jvp (forward-over-reverse) | jax_reverse
    grad_accum_steps: int = 1  # buckets per optimiser update
    # --- memory levers (forwarded to convert_to_chunked) --------------------
    layer_chunk_size: int = 0  # edge-axis tile width (0 disables)
    chunk_encoder: bool = False
    checkpoint: bool = False
    ckpt_mode: str = "stack"
    # --- bookkeeping --------------------------------------------------------
    seed: int = 1234
    log_every: int = 10
    warmup_steps: int = 5  # excluded from median step_time
    out_dir: str = "ckpts"
    save_every_steps: int = 0  # 0 disables intermediate saves
    config: str = ""  # optional YAML to merge under the CLI overrides
    # --- mlflow -------------------------------------------------------------
    enable_mlflow: bool = True
    mlflow_uri: str = DEFAULT_TRACKING_URI
    mlflow_experiment: str = DEFAULT_EXPERIMENT
    run_name: str | None = None


def load_config() -> JaxTrainConfig:
    """Schema dataclass <- optional YAML <- CLI dotlist overrides."""
    base = OmegaConf.structured(JaxTrainConfig)
    cli = OmegaConf.from_cli()
    if cli.get("config"):
        base = OmegaConf.merge(base, OmegaConf.load(cli.config))
    cfg = OmegaConf.merge(base, cli)
    if cfg.method not in _GRAD_FNS:
        raise SystemExit(f"method must be one of {sorted(_GRAD_FNS)}.")
    if cfg.checkpoint and cfg.ckpt_mode not in CHECKPOINTERS:
        raise SystemExit(f"ckpt_mode must be one of {sorted(CHECKPOINTERS)}.")
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


class JaxBucketCollator:
    """`BucketCollator` (drop-overflow guard) + numpy bucket padding, in one
    `collate_fn` so both run in DataLoader workers. Emits the fixed-shape padded
    `(JaxAtomGraphs, targets)` as host numpy arrays; the train loop device_puts
    them. Replaces the old main-thread `to_jax`/`pad_to_bucket` (eager jnp, ~300ms
    /batch)."""

    def __init__(
        self, base: BucketCollator, *, n_pad, e_pad, g_pad, has_stress,
        reference_coefficients=None,
    ):
        self.base = base
        self.n_pad = n_pad
        self.e_pad = e_pad
        self.g_pad = g_pad
        self.has_stress = has_stress
        # fp64 (118,) reference-energy coefficients. When set (with a fp64 dataset),
        # `to_padded_numpy` precomputes the fp64 interaction target on the host.
        self.reference_coefficients = reference_coefficients

    def __call__(self, samples):
        torch_batch = self.base(samples)
        if torch_batch is None:  # whole bucket dropped as too big
            return None
        return jgb.to_padded_numpy(
            torch_batch, self.n_pad, self.e_pad, self.g_pad,
            has_stress=self.has_stress,
            reference_coefficients=self.reference_coefficients,
        )

    # surface the base collator's drop counters for the end-of-run report
    @property
    def n_dropped_members(self):
        return self.base.n_dropped_members

    @property
    def n_trimmed_buckets(self):
        return self.base.n_trimmed_buckets


def build_loader(
    cfg: JaxTrainConfig, atoms_adapter, *, has_stress: bool, reference_coefficients=None
):
    """ASE SQLite -> bucket-packed DataLoader (torch AtomGraphs on CPU) + sampler.

    Returns (loader, sampler, collator) so the caller can `sampler.set_epoch` and
    read the collator's drop counters.

    When `reference_coefficients` is given, the dataset is built in fp64 so the
    absolute energy label survives the torch graph-construction cast, and the
    collator precomputes the fp64 `raw - reference` interaction target on the host
    (the model still trains in fp32; only the small interaction reaches the device).
    """
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
        # fp64 keeps the energy label precise on the host for the reference
        # subtraction; harmless when no reference is supplied.
        dtype=torch.float64 if reference_coefficients is not None else None,
    )
    natoms = read_natoms(cfg.data_path)
    print(f"Dataset: {len(dataset)} structures from {cfg.data_path}")

    # FFD budgets in estimate space. Edge axis packs to the same cap the bucket
    # pads to; degree_estimate decides how close the estimate sits to reality.
    budgets = {
        "edges": cfg.edge_budget,
        "nodes": cfg.node_budget,
        "graphs": cfg.graph_budget,
    }
    sampler = BucketBatchSampler(
        natoms,
        budgets,
        degree_estimate=cfg.degree_estimate,
        chunk_size=cfg.dataset_chunk_size,
        shuffle=True,
        seed=cfg.seed,
    )
    collator = JaxBucketCollator(
        BucketCollator(
            atoms_adapter.batch,
            n_max=cfg.node_budget,
            e_max=cfg.edge_budget,
            g_max=cfg.graph_budget,
        ),
        n_pad=cfg.node_budget,
        e_pad=cfg.edge_budget,
        g_pad=cfg.graph_budget + 1,
        has_stress=has_stress,
        reference_coefficients=reference_coefficients,
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


def prepare_batch(batch):
    """Worker-padded numpy `(graph, targets)` -> same pytrees on device.

    All conversion + bucket padding now happens in the DataLoader workers
    (`JaxBucketCollator` -> `to_padded_numpy`); the main thread only transfers the
    fixed-shape arrays to the device in one `device_put`.
    """
    graph_np, targets_np = batch
    return jax.device_put((graph_np, targets_np))


def real_batch_counts(graph) -> tuple[int, int, int]:
    """Real (non-padding) graphs, atoms, edges in one padded bucket."""
    mask = jgb.real_graph_mask(graph)
    n_graphs = int(jnp.sum(mask))
    n_atoms = int(jnp.sum(jnp.where(mask, graph.n_node, 0)))
    n_edges = int(jnp.sum(jnp.where(mask, graph.n_edge, 0)))
    return n_graphs, n_atoms, n_edges


def micro_batch_counts(micro: list) -> tuple[int, int, int]:
    """Sum real counts across micro-batches in one optimiser step."""
    graphs = atoms = edges = 0
    for graph, _ in micro:
        g, a, e = real_batch_counts(graph)
        graphs += g
        atoms += a
        edges += e
    return graphs, atoms, edges


def main() -> None:
    cfg = load_config()
    has_stress = not cfg.no_stress

    # Pin fp32 matmul mode so the JAX/torch comparison is precision-matched
    # (otherwise XLA picks its own default, often TF32, on Ampere).
    jax.config.update("jax_default_matmul_precision", cfg.matmul_precision)

    torch.manual_seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # 1. Pretrained torch model (weights + atoms adapter) -> fresh JAX regressor.
    print(f"Loading pretrained torch model: {cfg.base_model}")
    torch_model, atoms_adapter = getattr(pretrained, cfg.base_model)(
        device="cpu", precision="float32-high"
    )
    print("Porting weights into JAX...")
    model = load_orb_v3_conservative_into_jax(torch_model, key=key)
    model = convert_to_chunked(
        model,
        chunk=cfg.layer_chunk_size,
        chunk_encoder=cfg.chunk_encoder,
        checkpoint=cfg.checkpoint,
        ckpt_mode=cfg.ckpt_mode,
    )
    del torch_model

    # 2. Data.
    # fp64 reference coefficients (upcast from the model's fp32 buffer) let the
    # collator build the energy target via a fp64 host subtraction -- precise even
    # at OMol ~1e5 eV scale, where the in-graph fp32 `raw - reference` loses ~meV.
    reference_coefficients = np.asarray(
        model.energy_head.reference.coefficients, dtype=np.float64
    )
    loader, sampler, collator = build_loader(
        cfg, atoms_adapter, has_stress=has_stress,
        reference_coefficients=reference_coefficients,
    )

    # 3. Optimiser + jitted step. LR schedule sized from max_steps.
    loss_weights = {
        "energy": cfg.energy_loss_weight,
        "forces": cfg.forces_loss_weight,
        "stress": cfg.stress_loss_weight,
    }
    optimizer = make_optimizer(lr=cfg.lr, total_steps=cfg.max_steps)
    opt_state = init_opt_state(model, optimizer)
    grad_fn = _GRAD_FNS[cfg.method]
    accum = max(1, cfg.grad_accum_steps)
    if accum > 1:
        step = make_accum_train_step(
            optimizer, loss_weights, has_stress=has_stress, grad_fn=grad_fn
        )
    else:
        step = make_train_step(
            optimizer, loss_weights, has_stress=has_stress, grad_fn=grad_fn
        )

    os.makedirs(cfg.out_dir, exist_ok=True)
    lever = (
        f"chunk={cfg.layer_chunk_size}{'+enc' if cfg.chunk_encoder else ''}"
        f"{' ckpt-' + cfg.ckpt_mode if cfg.checkpoint else ''}"
    )
    print(
        f"Training: method={cfg.method} [{lever}] accum={accum} on {jax.devices()[0]}; "
        f"bucket=({cfg.node_budget} nodes / {cfg.edge_budget} edges / {cfg.graph_budget + 1} graphs)"
    )

    mlflow = MlflowLogger(
        SimpleNamespace(
            enable_mlflow=cfg.enable_mlflow,
            run_name=cfg.run_name,
            mlflow_experiment=cfg.mlflow_experiment,
            mlflow_uri=cfg.mlflow_uri,
        ),
        run_name=cfg.method,
        params={
            "method": cfg.method,
            "base_model": cfg.base_model,
            "matmul_precision": cfg.matmul_precision,
            "edge_budget": cfg.edge_budget,
            "node_budget": cfg.node_budget,
            "graph_budget": cfg.graph_budget,
            "degree_estimate": cfg.degree_estimate,
            "grad_accum_steps": accum,
            "chunk": cfg.layer_chunk_size,
            "chunk_encoder": cfg.chunk_encoder,
            "checkpoint": cfg.checkpoint,
            "ckpt_mode": cfg.ckpt_mode if cfg.checkpoint else "n/a",
            "lr": cfg.lr,
            "max_steps": cfg.max_steps,
            "energy_loss_weight": cfg.energy_loss_weight,
            "forces_loss_weight": cfg.forces_loss_weight,
            "stress_loss_weight": cfg.stress_loss_weight,
            "no_stress": cfg.no_stress,
            "seed": cfg.seed,
            "device": str(jax.devices()[0]),
        },
    )

    # 4. Train loop driven by max_steps. The first step pays the one-time compile.
    n_steps = 0
    running = 0.0
    running_step_time = 0.0
    step_times: list[float] = []
    epoch = 0
    t0 = time.perf_counter()
    step_t0: float | None = None
    while n_steps < cfg.max_steps:
        sampler.set_epoch(epoch)
        micro: list = []  # accumulated (graph, targets) for the next update
        for batch in loader:
            if batch is None:  # collator dropped the whole bucket
                continue
            if step_t0 is None:
                step_t0 = time.perf_counter()
            micro.append(prepare_batch(batch))
            if len(micro) < accum:
                continue
            n_graphs, n_atoms, n_edges = micro_batch_counts(micro)
            model, opt_state, metrics = (
                step(model, opt_state, micro)
                if accum > 1
                else step(model, opt_state, *micro[0])
            )
            # step()/device_put dispatch async; block on GPU completion so
            # step_time reflects real compute (matches torch's in-window sync).
            jax.block_until_ready((model, opt_state, metrics))
            step_time = time.perf_counter() - step_t0
            step_t0 = None
            micro = []
            loss = float(metrics["total"])
            running += loss
            running_step_time += step_time
            step_times.append(step_time)
            n_steps += 1
            if n_steps % cfg.log_every == 0:
                print(
                    f"  step {n_steps}/{cfg.max_steps}  "
                    f"loss={loss:.4f} (avg {running / n_steps:.4f})  "
                    f"grad_norm={float(metrics['grad_norm']):.4f}  "
                    f"step_time={step_time:.3f}s (avg {running_step_time / n_steps:.3f}s)  "
                    f"graphs={n_graphs} atoms={n_atoms} edges={n_edges}"
                )
            mlflow.log_metrics(
                {
                    "loss": loss,
                    "grad_norm": float(metrics["grad_norm"]),
                    "step_time": step_time,
                    "graphs_in_step": n_graphs,
                    "atoms_in_step": n_atoms,
                    "edges_in_step": n_edges,
                },
                step=n_steps,
            )
            if cfg.save_every_steps and n_steps % cfg.save_every_steps == 0:
                path = os.path.join(cfg.out_dir, f"jax_finetune_step{n_steps}.eqx")
                eqx.tree_serialise_leaves(path, model)
                print(f"  saved {path}")
            if n_steps >= cfg.max_steps:
                break
        epoch += 1

    dt = time.perf_counter() - t0
    # Median over post-warmup steps: robust to the one-time jit compile.
    timed = step_times[cfg.warmup_steps :] or step_times
    median_step = statistics.median(timed) if timed else 0.0
    # XLA exposes a process-wide peak; unlike torch it can't be reset, so this
    # includes the compilation peak (usually <= the steady-state training peak).
    try:
        peak_mem_gb = (
            jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9
        )
    except Exception:
        peak_mem_gb = 0.0
    mean_step_time = sum(timed) / len(timed)
    first_step_time = step_times[0]
    print(
        f"Done {n_steps} steps in {dt:.1f}s ({n_steps / dt:.2f} steps/s)  "
        f"median_step_time={median_step:.3f}s (over {len(timed)} steps)  "
        f"mean_step_time={mean_step_time:.3f}s  "
        f"first_step_time={first_step_time:.3f}s  "
        f"peak_mem={peak_mem_gb:.2f}GB (process-wide, incl. compile)  "
        f"mean_loss={running / max(n_steps, 1):.4f}"
    )
    if collator.n_dropped_members:
        print(
            f"NOTE: collator trimmed {collator.n_dropped_members} member(s) across "
            f"{collator.n_trimmed_buckets} bucket(s) for exceeding the edge estimate."
        )
    final = os.path.join(cfg.out_dir, "jax_finetune_final.eqx")
    eqx.tree_serialise_leaves(final, model)
    print(f"Saved {final}")
    mlflow.log_metrics(
        {
            "first_step_time": first_step_time,
            "mean_loss": running / max(n_steps, 1),
            "median_step_time": median_step,
            "mean_step_time": mean_step_time,
            "peak_mem_gb": peak_mem_gb,
        },
        step=n_steps,
    )
    mlflow.finish()


if __name__ == "__main__":
    import multiprocessing

    # Spawn (not fork) so DataLoader workers don't inherit a CUDA/JAX context.
    multiprocessing.set_start_method("spawn", force=True)
    main()
