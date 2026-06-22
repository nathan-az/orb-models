"""torch (eager) arm of the MD-style inference benchmark on aqueous NaCl(aq).

See `_md_common.py` for the driver/config. Run ONE framework per process (torch
and JAX can't share a CUDA context). orbmol_v2 = conservative backbone + periodic
PME electrostatics (nvalchemiops Warp PME, eager). Random weights -- compute is
weight-independent. Run with TORCH_COMPILE_DISABLE=1 (no triton toolchain here).

  python benchmarks/inference/bench_md_torch.py n_side=4 steps=10
  python benchmarks/inference/bench_md_torch.py sweep=true n_side=3 n_side_max=10
"""

from __future__ import annotations

import numpy as np
import torch

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.pretrained import orb_v3_conservative_architecture

from _md_common import BenchMDConfig, load_config, run_sizes

_CKPT_OFF = frozenset({"", "none"})


class TorchStepper:
    def __init__(self, cfg: BenchMDConfig, atoms):
        self.cfg = cfg
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision(cfg.precision)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # has_charge_spin_cond stays False: charge/spin ride on atoms.info (read by
        # the adapter) -> conditioner + charge head.
        self.model = orb_v3_conservative_architecture(
            has_charge_spin_cond=False, has_stress=cfg.stress,
            has_electrostatics=True, device=self.dev,
            checkpoint=(None if cfg.checkpoint in _CKPT_OFF else cfg.checkpoint),
        ).eval()
        if cfg.checkpoint not in _CKPT_OFF:
            # CheckpointedSequential only remats when self.training is True
            # in eval(), activation checkpointing is a silent no-op even
            # though the force backward would benefit. Ungate it by flipping ONLY those
            # submodules to train mode (they're Linear+activation stacks; layer_norm is
            # a sibling, not inside them), and force any dropout within back to eval so
            # the result stays deterministic and we isolate the checkpoint effect.
            from orb_models.common.models.nn_util import CheckpointedSequential

            n_ckpt = 0
            for m in self.model.modules():
                if isinstance(m, CheckpointedSequential):
                    m.train()
                    for sub in m.modules():
                        if isinstance(sub, torch.nn.Dropout):
                            sub.eval()
                    n_ckpt += 1
            print(f"[ckpt] ungated eval-mode checkpointing on {n_ckpt} module(s)")
        if cfg.compile:
            # dynamic=True: one compile over symbolic shapes -> no recompile as the
            # edge count changes step to step (the torch answer to JAX's padding).
            self.model.compile(mode="default", dynamic=True)
        self.adapter = ForcefieldAtomsAdapter(radius=cfg.radius, max_num_neighbors=cfg.max_neighbors)
        atoms.info["charge"] = 0.0
        atoms.info["spin"] = 1.0
        self.atoms = atoms

    def reset_mem(self):
        if self.dev == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def prep(self, positions: np.ndarray):
        self.atoms.set_positions(positions)
        # knn_scipy is CPU-only; build on CPU then move the batch to the device.
        graph = self.adapter.from_ase_atoms(self.atoms, edge_method="knn_scipy", device="cpu")
        if self.dev == "cuda":
            graph = graph.to(torch.device("cuda"))
            torch.cuda.synchronize()
        return graph

    def force(self, graph):
        out = self.model(graph, compute_forces=True, compute_stress=self.cfg.stress)
        f = out["forces"].detach().cpu().numpy()
        e = float(out["energy"].detach().cpu().reshape(-1)[0])
        if self.dev == "cuda":
            torch.cuda.synchronize()
        return f, e

    def peak_mem_mib(self) -> float:
        return (torch.cuda.max_memory_allocated() if self.dev == "cuda" else 0) / 2**20

    def status_for(self, exc: Exception) -> str:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return "OOM"
        return f"FAIL: {type(exc).__name__}: {exc}"

    def extra(self) -> dict:
        return {
            "framework": "torch",
        }

    def cleanup(self) -> None:
        del self.model
        if self.dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_sizes(load_config(BenchMDConfig), TorchStepper)
