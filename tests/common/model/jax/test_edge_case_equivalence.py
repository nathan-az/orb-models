"""Padded JAX vs unpadded PyTorch on small-system edge cases.

The inputs where padding/masking bugs hide: a single periodic atom with self-loop
edges (sender == receiver, nonzero shift), a zero-edge graph (`segment_sum` over
nothing), and a mixed batch of tiny + normal graphs. Shared weights, float64 both
sides (conftest), so any disagreement is a real bug.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.forcefield_utils import (
    torch_full_3x3_to_voigt_6_stress,
)
from orb_models.forcefield.models.jax.conservative_regressor import predict

# Reuse the orb-v3-feature-set model + weight copy from the regressor test.
from tests.common.model.jax.test_conservative_regressor import _build_real_features

pytestmark = pytest.mark.equivalence

# A cell small enough that nonzero unit shifts land self-image neighbours INSIDE
# the 6 A cutoff: |shift @ (3*I)| is 3 A for an axis shift, ~5.2 A for [1,1,1].
SELF_CELL = np.eye(3) * 3.0


def _make_torch_batch(specs):
    """Assemble a disjoint multi-system periodic `AtomGraphs` from per-system specs.

    Each spec is (n_atoms, list_of_(local_sender, local_receiver, unit_shift)).
    Senders/receivers are offset to global node indices, mirroring AtomGraphs.batch.
    All edge `vectors` start at zero and are recomputed from positions/shifts/cell
    inside each framework's forward -- so self-loops (s == r) get vector shift@cell,
    nonzero as long as the shift is nonzero.
    """
    rng = np.random.default_rng(0)
    n_node, n_edge = [], []
    senders, receivers, unit_shifts = [], [], []
    offset = 0
    for n, edges in specs:
        n_node.append(n)
        n_edge.append(len(edges))
        for s, r, shift in edges:
            senders.append(offset + s)
            receivers.append(offset + r)
            unit_shifts.append(shift)
        offset += n

    N, G, E = offset, len(specs), len(senders)
    z = rng.integers(1, 30, size=N).astype(np.int64)
    cell = np.stack([SELF_CELL for _ in range(G)])
    return AtomGraphs(
        senders=torch.tensor(np.array(senders, dtype=np.int64)),
        receivers=torch.tensor(np.array(receivers, dtype=np.int64)),
        n_node=torch.tensor(np.array(n_node, dtype=np.int64)),
        n_edge=torch.tensor(np.array(n_edge, dtype=np.int64)),
        node_features={
            "positions": torch.tensor(rng.standard_normal((N, 3)) * 1.5),
            "atomic_numbers": torch.tensor(z),
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                torch.tensor(z - 1), num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((E, 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(
                np.array(unit_shifts, dtype=np.float64).reshape(E, 3)
            ),
        },
        system_features={
            "cell": torch.tensor(cell),
            "pbc": torch.ones((G, 3), dtype=torch.bool),
        },
        node_targets={},
        edge_targets={},
        system_targets={},
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([50]),
    )


# Per-system edge case specs (local indices). Shifts are always nonzero on
# self-loops so the recomputed edge vector is nonzero (||v|| -> finite grad).
_SINGLE_ATOM_SELF_IMAGES = (
    1,
    [
        (0, 0, [1, 0, 0]),
        (0, 0, [-1, 0, 0]),
        (0, 0, [0, 1, 0]),
        (0, 0, [0, -1, 0]),
        (0, 0, [1, 1, 0]),
        (0, 0, [-1, -1, 0]),
    ],
)
_SINGLE_ATOM_ZERO_EDGES = (1, [])  # nearest image beyond cutoff -> reference only
_NORMAL_TRIATOMIC = (
    3,
    [(0, 1, [0, 0, 0]), (1, 2, [0, 0, 0]), (2, 0, [0, 0, 0])],
)


def _assert_padded_jax_matches_unpadded_torch(helpers, torch_model, jax_model, batch):
    """torch forward on the unpadded batch == padded-JAX predict, sliced to reals."""
    n_node = batch.n_node
    N, G, E = (
        int(n_node.sum()),
        int(n_node.shape[0]),
        int(batch.n_edge.sum()),
    )
    # Snapshot jax graph BEFORE the torch forward mutates the batch in place.
    # Pad to a bucket strictly larger in every axis (forces a real padding graph
    # plus a trailing empty graph slot -- the harder masking case). This batch has no
    # targets, so `to_padded_numpy` returns an empty target dict (ignored here).
    padded_np, _ = jgb.to_padded_numpy(
        batch, n_pad=N + 5, e_pad=E + 7, g_pad=G + 2, has_stress=True
    )
    padded = jax.device_put(padded_np)

    out = torch_model(batch, fp64_energy=True)
    preds = predict(padded, jax_model, has_stress=True)

    # Real graphs are slots [0, G); real atoms are the first N rows (jraph order).
    helpers.assert_close(preds.energy[:G], out["interaction_energy"])
    jax_absolute = jax_model.energy_head.absolute_energy(preds.energy, padded)
    helpers.assert_close(jax_absolute[:G], out["energy"])
    helpers.assert_close(preds.forces[:N], out["forces"])
    jax_stress_voigt = torch_full_3x3_to_voigt_6_stress(
        torch.tensor(np.asarray(preds.stress[:G]))
    )
    helpers.assert_close(jnp.asarray(jax_stress_voigt.numpy()), out["stress"])

    # Padding must not leak into the real predictions.
    assert np.isfinite(np.asarray(preds.energy)).all()
    assert np.isfinite(np.asarray(preds.forces)).all()


def test_single_atom_self_image_edges(helpers, key):
    """A 1-atom cell whose only neighbours are its own periodic images."""
    torch_model, jax_model = _build_real_features(key)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    batch = _make_torch_batch([_SINGLE_ATOM_SELF_IMAGES])
    _assert_padded_jax_matches_unpadded_torch(helpers, torch_model, jax_model, batch)


def test_single_atom_zero_edges(helpers, key):
    """The degenerate cell: 1 atom, no edges -> reference-energy-only, zero force."""
    torch_model, jax_model = _build_real_features(key)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    batch = _make_torch_batch([_SINGLE_ATOM_ZERO_EDGES])
    _assert_padded_jax_matches_unpadded_torch(helpers, torch_model, jax_model, batch)


def test_mixed_tiny_and_normal_batch(helpers, key):
    """A batch mixing both single-atom edge cases with a normal system, padded to
    one bucket -- the real packing scenario where masking bugs surface."""
    torch_model, jax_model = _build_real_features(key)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    batch = _make_torch_batch(
        [_SINGLE_ATOM_SELF_IMAGES, _SINGLE_ATOM_ZERO_EDGES, _NORMAL_TRIATOMIC]
    )
    _assert_padded_jax_matches_unpadded_torch(helpers, torch_model, jax_model, batch)
