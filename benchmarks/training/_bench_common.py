"""Shared constants + dummy targets for the bucket-feasibility probes.

`check_torch_bucket.py` / `check_jax_bucket.py` ask a shape/memory/throughput
question, not a parity one: can one FFD-packed bucket fit, and how fast is a step?
So the targets are random and the optimiser config is fixed and identical on both
sides -- only the framework differs. These are benchmark-only knobs (not production
utilities), so they live here rather than in the library.
"""

from __future__ import annotations

import numpy as np

# orb-v3 conservative graph construction: 6 A cutoff, "effectively unlimited"
# neighbours (120 is empirically sufficient for the training distribution). The
# probes cap degree via their own --max-neighbors to hit a target bucket corner.
RADIUS = 6.0
MAX_NEIGHBORS = 120

# Optimiser config, shared so the optimiser update is the same work on both sides.
LR = 3e-4
TOTAL_STEPS = 1000
LOSS_WEIGHTS = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
PRECISION = "float32-high"  # fp32 + TF32 matmuls; orb-v3's real training precision.


def make_targets(n_atoms: int, n_graphs: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Random supervised targets in fp32, the same numbers for every framework."""
    rng = np.random.default_rng(seed)
    return {
        "energy": rng.standard_normal((n_graphs,)).astype(np.float32),
        "forces": (rng.standard_normal((n_atoms, 3)) * 0.1).astype(np.float32),
        "stress": (rng.standard_normal((n_graphs, 6)) * 0.01).astype(np.float32),
    }
