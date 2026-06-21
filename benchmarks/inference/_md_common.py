"""Framework-agnostic core of the MD-style inference benchmark.

The torch and JAX entry points (`bench_md_torch.py`, `bench_md_jax.py`) share
everything here -- config, the velocity-Verlet driver, the size-sweep loop, and
optional MLflow logging -- and each supplies only a `Stepper`: an object that
knows how to build a graph from positions (`prep`) and evaluate forces (`force`)
in its framework. torch and JAX can't share a CUDA context in one process, so the
split keeps each framework's import at the top of its own script (no lazy imports).

Config follows the train scripts: a frozen dataclass schema, overridable by an
optional YAML and then by CLI dotlist args, e.g.
    python bench_md_jax.py n_side=10 steps=5 slack=0.35
    python bench_md_torch.py config=presets/sweep.yaml sweep=true
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from omegaconf import OmegaConf

from electrolyte import build_electrolyte

# Reuse the train scripts' MLflow wrapper (sibling benchmarks/training dir). Optional:
# a clean no-op if the dir/module/mlflow isn't importable, so the bench still runs.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
try:
    from mlflow_util import DEFAULT_TRACKING_URI, MlflowLogger  # type: ignore
except Exception:  # noqa: BLE001
    DEFAULT_TRACKING_URI = "http://localhost:5000"
    MlflowLogger = None  # type: ignore


@dataclass
class BenchMDConfig:
    # --- system -------------------------------------------------------------
    n_side: int = 4  # n_side**3 waters; 4->~192 atoms, 10->~2880
    n_side_max: int = 8  # upper bound when sweep=true
    sweep: bool = False  # grow n_side from n_side..n_side_max until OOM/fail
    ion_fraction: float = 0.06
    seed: int = 0
    # --- graph / model ------------------------------------------------------
    radius: float = 6.0
    max_neighbors: int = 20
    lr_wavelength: float = 1.0  # PME reciprocal cutoff knob
    stress: bool = False  # forces-only by default (NVE MD needs no stress)
    # fp32 matmul precision: "high" = TF32 (orb's loader default, see
    # pretrained.py set_torch_precision); "highest" = true fp32.
    precision: str = "high"
    # --- MD -----------------------------------------------------------------
    steps: int = 10  # timed MD steps (+1 warmup / compile)
    timestep_fs: float = 0.5
    # --- JAX capacity padding -----------------------------------------------
    slack: float = 0.35  # edge-capacity redundancy: pad edges to needed*(1+slack)
    # --- memory/speed levers ------------------------------------------------
    compile: bool = False  # (torch) torch.compile(mode="default", dynamic=True)
    # activation checkpointing (""=off). Vocabulary is framework-specific:
    #   torch: reentrant | non-reentrant  (checkpoint_sequential over the MLP layers)
    #   jax:   stack | full | encoder | stack_manual  (remat whole GNS stacks/encoder)
    checkpoint: str = ""
    # edge-axis chunking (JAX only; 0=off). Tiles each GNS stack over its edges so only
    # `chunk` edge-rows are live at once -- combinable with `checkpoint`.
    chunk: int = 0
    chunk_encoder: bool = False  # also chunk the encoder edge_fn (no effect if chunk==0)
    chunk_remat: bool = True  # recompute each tile in the backward pass instead of storing it
    # --- bookkeeping --------------------------------------------------------
    out: str = ""  # append JSON line(s) to this file
    config: str = ""  # optional YAML merged UNDER the CLI overrides
    # --- mlflow (off by default: a bench is often run without a server) ------
    enable_mlflow: bool = False
    mlflow_uri: str = DEFAULT_TRACKING_URI
    mlflow_experiment: str = "orb-md-bench"
    run_name: str | None = None


def load_config(schema: type = BenchMDConfig) -> BenchMDConfig:
    """Schema dataclass <- optional YAML <- CLI dotlist overrides."""
    base = OmegaConf.structured(schema)
    cli = OmegaConf.from_cli()
    if cli.get("config"):
        base = OmegaConf.merge(base, OmegaConf.load(cli.config))
    cfg = OmegaConf.merge(base, cli)
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def grow_capacity(n: int, slack: float) -> int:
    """Capacity = needed * (1 + slack), like a std::vector reserve. The slack soaks
    up edge-count fluctuation as atoms move so jit recompiles only on real growth."""
    return int(np.ceil(n * (1.0 + slack)))


def summarize(times_s: list[float]) -> dict:
    a = np.array(times_s) * 1e3  # ms
    return {
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "mean_ms": float(np.mean(a)),
    }


class Stepper(Protocol):
    """What each framework provides to the shared driver."""

    atoms: Any  # the ASE system this stepper was built for

    def reset_mem(self) -> None: ...
    def prep(self, positions: np.ndarray) -> Any: ...  # host NL build + H2D (timed)
    def force(self, state: Any) -> tuple[np.ndarray, float]: ...  # (forces[N,3], energy)
    def peak_mem_mib(self) -> float: ...
    def status_for(self, exc: Exception) -> str: ...  # classify a failure (e.g. "OOM")
    def extra(self) -> dict: ...  # framework-specific fields (e_cap, n_recompiles, ...)
    def cleanup(self) -> None: ...  # free device memory before the next sweep size


def run_trajectory(cfg: BenchMDConfig, stepper: Stepper) -> dict:
    """Velocity-Verlet (NVE) loop; host numpy integrator identical across frameworks.

    Per step times PREP (build graph from positions + H2D) vs FORCE (device eval).
    The force kernel runs forward+backward (forces are autograd grads), so peak
    memory is backward-dominated -- which is what eventually OOMs at large sizes.
    """
    atoms = stepper.atoms
    n_atoms = len(atoms)
    dt = cfg.timestep_fs * units.fs

    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(cfg.seed))
    pos = atoms.get_positions()
    mom = atoms.get_momenta()
    masses = atoms.get_masses()[:, None]

    prep_t, force_t, energies = [], [], []
    stepper.reset_mem()
    try:
        for s in range(cfg.steps + 1):  # +1 warmup (JAX compile)
            t0 = time.perf_counter()
            state = stepper.prep(pos)
            prep = time.perf_counter() - t0

            t0 = time.perf_counter()
            f, e = stepper.force(state)
            force = time.perf_counter() - t0

            f = f[:n_atoms]  # drop JAX padding-atom rows (no-op for torch)
            energies.append(e)
            # leapfrog half-kick (benchmark trajectory; not a conservation test)
            mom = mom + 0.5 * dt * f
            pos = pos + dt * mom / masses
            if s > 0:
                prep_t.append(prep)
                force_t.append(force)
    except Exception as exc:  # noqa: BLE001 -- record OOM/runtime failures, don't crash a sweep
        return {"n_atoms": n_atoms, "status": stepper.status_for(exc),
                "peak_mem_mib": stepper.peak_mem_mib(), **stepper.extra()}

    return {
        "n_atoms": n_atoms, "status": "ok", "peak_mem_mib": stepper.peak_mem_mib(),
        "energy_drift_ev": energies[-1] - energies[0],
        "prep": summarize(prep_t), "force": summarize(force_t), **stepper.extra(),
    }


def _flatten_metrics(result: dict) -> dict:
    m: dict[str, float] = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            m[k] = float(v)
        elif isinstance(v, dict):
            m.update({f"{k}_{kk}": float(vv) for kk, vv in v.items()
                      if isinstance(vv, (int, float))})
    return m


def run_sizes(cfg: BenchMDConfig, make_stepper: Callable[[BenchMDConfig, Any], Stepper]) -> None:
    """Drive one (or a sweep of) system size(s); print + optionally log/append each."""
    logger = None
    if cfg.enable_mlflow and MlflowLogger is not None:
        params = {k: ("" if v is None else v) for k, v in asdict(cfg).items()}
        logger = MlflowLogger(cfg, run_name=cfg.run_name or "bench_md", params=params)

    sizes = range(cfg.n_side, cfg.n_side_max + 1) if cfg.sweep else [cfg.n_side]
    try:
        for n_side in sizes:
            atoms = build_electrolyte(n_side, ion_fraction=cfg.ion_fraction, seed=cfg.seed)
            stepper = make_stepper(cfg, atoms)
            result = {"n_side": n_side, **run_trajectory(cfg, stepper)}
            print(json.dumps(result))
            if logger is not None:
                logger.log_metrics(_flatten_metrics(result), step=n_side)
            if cfg.out:
                with open(cfg.out, "a") as fh:
                    fh.write(json.dumps(result) + "\n")
            stop = cfg.sweep and result.get("status") != "ok"
            # Free this size's device buffers BEFORE building the next (larger) size,
            # so a sweep doesn't allocate the next bucket on top of this one's. Matters
            # for JAX: XLA caches executables + holds prior-capacity buffers
            # (PREALLOCATE=false), which otherwise false-OOMs a one-process sweep.
            stepper.cleanup()
            del stepper, atoms
            if stop:
                print(f"# stop sweep: n_side={n_side} -> {result.get('status')}")
                break
    finally:
        if logger is not None:
            logger.finish()
