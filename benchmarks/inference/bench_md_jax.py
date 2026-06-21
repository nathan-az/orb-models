"""JAX (jit) arm of the MD-style inference benchmark on aqueous NaCl(aq).

See `_md_common.py` for the driver/config. Run ONE framework per process. The atom
count is fixed during MD, so only the EDGE count (and PME pair list) fluctuate as
atoms move: we pad edges to a CAPACITY = needed*(1+slack) (a std::vector reserve),
so `eqx.filter_jit` recompiles only when an edge count exceeds capacity, then the
capacity grows. `n_recompiles` should be 1 across a whole trajectory.

NOTE on sweeps: between sizes we free the model + jit cache (`cleanup`) and, when
`sweep=true`, switch to the platform (cudaMalloc/Free) allocator so freed memory is
actually returned -- otherwise the default BFC allocator hoards prior-capacity
buffers and false-OOMs the next, bigger size. This makes a one-process sweep close
to truthful, but it can still report a ceiling ~1 size pessimistic; for the exact
single-size ceiling, run one size per process.

  python benchmarks/inference/bench_md_jax.py n_side=10 steps=5 slack=0.35
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _sweeping() -> bool:
    """Pre-jax-import peek at CLI (and optional YAML) for sweep=true. CLI wins."""
    cli_sweep, cfg_path = None, None
    for a in sys.argv[1:]:
        if a.startswith("sweep="):
            cli_sweep = a.split("=", 1)[1]
        elif a.startswith("config="):
            cfg_path = a.split("=", 1)[1]
    if cli_sweep is not None:
        return str(cli_sweep).lower() in ("true", "1", "yes")
    if cfg_path:
        try:
            from omegaconf import OmegaConf  # no jax dependency

            return bool(OmegaConf.load(cfg_path).get("sweep", False))
        except Exception:  # noqa: BLE001
            return False
    return False


# For a one-process SWEEP, use the platform (cudaMalloc/Free) allocator: it returns
# freed memory between sizes, so the next size doesn't fail on the previous size's
# pooled/fragmented buffers (the default BFC allocator hoards them -> false OOM).
# It's slower per alloc, so single-size timing runs keep the default BFC allocator.
if _sweeping():
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.jax.conservative_regressor import predict
from orb_models.forcefield.models.jax.optimisation import CHECKPOINTERS, convert_to_chunked
from orb_models.forcefield.models.jax.pme import build_pme_batch, build_pme_structure
from orb_models.forcefield.models.jax.port_weights import build_orbmol_v2_jax

from _md_common import BenchMDConfig, grow_capacity, load_config, run_sizes

_CKPT_OFF = frozenset({"", "none"})


class JaxStepper:
    def __init__(self, cfg: BenchMDConfig, atoms):
        self.cfg = cfg
        jax.config.update("jax_default_matmul_precision", cfg.precision)
        dtype = jnp.float32
        self.cast = lambda t: jax.tree_util.tree_map(  # noqa: E731
            lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, t
        )
        # Arrays land on the default (GPU) device; no device_put on the model (it has
        # non-array function leaves device_put rejects).
        self.model = self.cast(build_orbmol_v2_jax(key=jax.random.PRNGKey(0)))
        ckpt_on = cfg.checkpoint not in _CKPT_OFF
        if ckpt_on and cfg.checkpoint not in CHECKPOINTERS:
            raise SystemExit(
                f"jax ckpt_mode must be one of {sorted({*CHECKPOINTERS, *_CKPT_OFF})}; "
                f"got {cfg.checkpoint!r}."
            )
        if cfg.chunk or ckpt_on:
            # Memory levers, recomputed/streamed in the force backward to cut peak memory
            # (slower). `chunk` tiles each GNS stack over its edges (ChunkedStack handles
            # orbmol_v2's additive conditioning); `ckpt_mode` remats whole stacks/encoder
            # ('stack'/'encoder'/'full'). The two compose.
            self.model = convert_to_chunked(
                self.model,
                chunk=cfg.chunk,
                chunk_encoder=cfg.chunk_encoder,
                checkpoint=ckpt_on,
                ckpt_mode=cfg.checkpoint,
                chunk_remat=cfg.chunk_remat,
            )
        self.adapter = ForcefieldAtomsAdapter(radius=cfg.radius, max_num_neighbors=cfg.max_neighbors)
        atoms.info["charge"] = 0.0
        atoms.info["spin"] = 1.0
        self.atoms = atoms

        self.N = len(atoms)
        self.n_pad, self.g_pad = self.N + 8, 2  # atoms fixed in MD -> n_pad fixed
        self.cell = atoms.get_cell().array
        self.pbc = np.array([True, True, True])
        self.step = eqx.filter_jit(lambda g, m: predict(g, m, has_stress=cfg.stress))
        self.e_cap = 0
        self.sigs: set = set()

    def reset_mem(self):
        pass  # JAX peak_bytes_in_use is a process high-water mark; nothing to reset

    def prep(self, positions: np.ndarray):
        self.atoms.set_positions(positions)
        graph = self.adapter.from_ase_atoms(self.atoms, edge_method="knn_scipy", device="cpu")
        n_edge = int(graph.n_edge.sum())
        if n_edge + 8 > self.e_cap:  # grow capacity (one recompile)
            self.e_cap = grow_capacity(n_edge + 8, self.cfg.slack)
        gnp, _ = jgb.to_padded_numpy(graph, self.n_pad, self.e_cap, self.g_pad,
                                     has_stress=self.cfg.stress)
        jg = self.cast(gnp)
        struct = build_pme_structure(positions, self.cell, self.pbc, self.cfg.lr_wavelength)
        pme = self.cast(build_pme_batch([struct], num_atoms=self.n_pad, num_structures=self.g_pad))
        jg = eqx.tree_at(lambda g: g.pme_prep, jg, pme, is_leaf=lambda x: x is None)
        jg = jax.device_put(jg)
        jax.block_until_ready(jg)
        self.sigs.add(tuple(tuple(np.shape(x)) for x in jax.tree_util.tree_leaves(jg)))
        return jg

    def force(self, jg):
        out = self.step(jg, self.model)
        jax.block_until_ready(out)
        return np.asarray(out.forces), float(np.asarray(out.energy).reshape(-1)[0])

    def peak_mem_mib(self) -> float:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**20

    def status_for(self, exc: Exception) -> str:
        return f"FAIL: {type(exc).__name__}: {str(exc)[:200]}"

    def extra(self) -> dict:
        return {
            "e_cap": self.e_cap,
            "n_recompiles": len(self.sigs),
            "ckpt_mode": self.cfg.checkpoint,
            "framework": "jax",
        }

    def cleanup(self) -> None:
        # Drop this size's device-resident model + jitted step + compilation cache so
        # the next, larger size doesn't allocate its capacity bucket on top of these
        # (XLA, PREALLOCATE=false). A fresh process is still the cleaner guarantee.
        import gc

        self.model = None
        self.step = None
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    run_sizes(load_config(BenchMDConfig), JaxStepper)
