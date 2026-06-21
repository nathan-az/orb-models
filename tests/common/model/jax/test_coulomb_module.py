"""Equivalence tests for the JAX non-periodic CoulombModule against the torch
`_direct_coulomb` path. Periodic PME is deferred (the torch module's hard,
un-jittable branch), so every system here is non-periodic.

The JAX module differs from torch by design: it returns ONLY per-graph energy
(G,). For non-periodic systems torch's `explicit_forces`/`explicit_stress` are
all-zero -- the spatial forces come from autograd through the energy -- so in JAX
the total Coulomb forces are simply `-jax.grad(energy.sum(), positions)`, which
includes the charge-equilibration term dE/dr through q(r) for free.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.coulomb_module import CoulombModule as TorchCoulombModule
from orb_models.forcefield.models.jax.coulomb_module import CoulombModule

pytestmark = pytest.mark.equivalence


def _torch_graph(positions, n_node):
    """Minimal non-periodic AtomGraphs (only the fields CoulombModule reads)."""
    positions = np.asarray(positions, dtype=np.float64)
    n_node = np.asarray(n_node, dtype=np.int64)
    N, G = positions.shape[0], n_node.shape[0]
    z = np.ones(N, dtype=np.int64)
    return AtomGraphs(
        senders=torch.zeros(0, dtype=torch.long),
        receivers=torch.zeros(0, dtype=torch.long),
        n_node=torch.tensor(n_node),
        n_edge=torch.zeros(G, dtype=torch.long),
        node_features={
            "positions": torch.tensor(positions),
            "atomic_numbers": torch.tensor(z),
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                torch.tensor(z - 1), num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((0, 3), dtype=torch.float64),
            "unit_shifts": torch.zeros((0, 3), dtype=torch.float64),
        },
        system_features={
            "cell": torch.zeros((G, 3, 3), dtype=torch.float64),
            "pbc": torch.zeros((G, 3), dtype=torch.bool),
        },
        node_targets={}, edge_targets={}, system_targets={},
        system_id=None, fix_atoms=None, tags=None,
        radius=6.0, max_num_neighbors=torch.tensor([20]),
    )


SIGMAS = [None, 1.0, 0.5]


@pytest.mark.parametrize("sigma", SIGMAS, ids=["undamped", "sigma1.0", "sigma0.5"])
def test_energy_matches_torch(sigma):
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((5, 3)) * 1.5
    n_node = [3, 2]
    charges = rng.standard_normal((5, 1))

    tg = _torch_graph(positions, n_node)
    jg = jgb.to_jax(tg)

    torch_mod = TorchCoulombModule(direct_coulomb_erf_damping_sigma=sigma).double()
    t_energy, _, _ = torch_mod(torch.tensor(charges), tg)

    jax_mod = CoulombModule(direct_coulomb_erf_damping_sigma=sigma)
    j_energy = jax_mod(jnp.asarray(charges), jg)

    np.testing.assert_allclose(np.asarray(j_energy), t_energy.detach().numpy(), atol=1e-10)


def test_zero_charges_zero_energy():
    jg = jgb.to_jax(_torch_graph(np.random.randn(5, 3), [3, 2]))
    e = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)(jnp.zeros((5, 1)), jg)
    assert np.all(np.abs(np.asarray(e)) < 1e-12)


def test_opposite_charges_attract():
    jg = jgb.to_jax(_torch_graph([[0, 0, 0], [1, 0, 0]], [2]))
    e = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)(
        jnp.asarray([[1.0], [-1.0]]), jg
    )
    assert float(e[0]) < 0


def test_single_atom_zero_energy():
    jg = jgb.to_jax(_torch_graph([[0.3, 0.2, 0.1]], [1]))
    e = CoulombModule()(jnp.asarray([[1.0]]), jg)
    assert abs(float(e[0])) < 1e-12


def test_dEdq_nonzero():
    rng = np.random.default_rng(1)
    jg = jgb.to_jax(_torch_graph(rng.standard_normal((5, 3)), [3, 2]))
    charges = jnp.asarray(rng.standard_normal((5, 1)))
    mod = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)
    g = jax.grad(lambda q: mod(q, jg).sum())(charges)
    assert np.abs(np.asarray(g)).sum() > 0


@pytest.mark.parametrize("sigma", SIGMAS, ids=["undamped", "sigma1.0", "sigma0.5"])
def test_total_forces_match_torch(sigma):
    """Total Coulomb forces (autograd, including dq/dr) match torch.

    Charges depend on positions via a fixed linear map so dq/dr != 0; torch
    forces = explicit(=0) - autograd(dE/dr), JAX forces = -grad(E, positions).
    """
    rng = np.random.default_rng(2)
    positions = rng.standard_normal((5, 3)) * 1.5
    n_node = [3, 2]
    W = rng.standard_normal((5, 15))  # q = W @ flatten(positions)

    def torch_charges(pos):
        return (torch.tensor(W) @ pos.reshape(-1)).reshape(-1, 1)

    tg = _torch_graph(positions, n_node)
    torch_mod = TorchCoulombModule(direct_coulomb_erf_damping_sigma=sigma).double()
    pos_t = tg.node_features["positions"].clone().requires_grad_(True)
    tg.node_features["positions"] = pos_t
    e_t, explicit_f, _ = torch_mod(torch_charges(pos_t), tg)
    (autograd_f,) = torch.autograd.grad(e_t.sum(), pos_t)
    torch_forces = explicit_f - autograd_f

    jg = jgb.to_jax(_torch_graph(positions, n_node))
    jax_mod = CoulombModule(direct_coulomb_erf_damping_sigma=sigma)

    def jax_energy(pos):
        q = jnp.asarray(W) @ pos.reshape(-1)
        g = eqx.tree_at(lambda gg: gg.node_features["positions"], jg, pos)
        return jax_mod(q.reshape(-1, 1), g).sum()

    jax_forces = -jax.grad(jax_energy)(jnp.asarray(positions))
    np.testing.assert_allclose(
        np.asarray(jax_forces), torch_forces.detach().numpy(), atol=1e-9
    )


def test_batched_equals_individual():
    rng = np.random.default_rng(3)
    mols = [rng.standard_normal((3, 3)), rng.standard_normal((2, 3)), rng.standard_normal((4, 3))]
    charges = [rng.standard_normal((m.shape[0], 1)) for m in mols]
    mod = CoulombModule(direct_coulomb_erf_damping_sigma=0.7)

    individual = [
        float(mod(jnp.asarray(q), jgb.to_jax(_torch_graph(m, [m.shape[0]])))[0])
        for m, q in zip(mols, charges)
    ]
    batched = mod(
        jnp.asarray(np.concatenate(charges, 0)),
        jgb.to_jax(_torch_graph(np.concatenate(mols, 0), [m.shape[0] for m in mols])),
    )
    np.testing.assert_allclose(np.asarray(batched), individual, atol=1e-10)


def test_padding_no_nan_and_real_energy_preserved():
    """A padded fixed-bucket batch: real-graph energies are unchanged and the
    energy/forces are finite despite padding atoms coinciding at the origin."""
    rng = np.random.default_rng(4)
    positions = rng.standard_normal((5, 3)) * 1.5
    n_node = [3, 2]
    charges = rng.standard_normal((5, 1))
    mod = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)

    jg = jgb.to_jax(_torch_graph(positions, n_node))
    e_unpadded = mod(jnp.asarray(charges), jg)

    tg = _torch_graph(positions, n_node)
    graph_np, _ = jgb.to_padded_numpy(tg, n_pad=11, e_pad=4, g_pad=4, has_stress=False)
    jg_pad = jax.device_put(graph_np)
    q_pad = jnp.asarray(np.concatenate([charges, np.zeros((6, 1))], 0))

    e_pad = mod(q_pad, jg_pad)
    assert np.isfinite(np.asarray(e_pad)).all()
    np.testing.assert_allclose(np.asarray(e_pad[:2]), np.asarray(e_unpadded), atol=1e-10)

    # Forces finite through padding (coincident padding atoms must not NaN).
    def energy(pos):
        g = eqx.tree_at(lambda gg: gg.node_features["positions"], jg_pad, pos)
        return mod(q_pad, g).sum()

    f = jax.grad(energy)(jg_pad.node_features["positions"])
    assert np.isfinite(np.asarray(f)).all()


def test_jit_equals_eager():
    rng = np.random.default_rng(5)
    jg = jgb.to_jax(_torch_graph(rng.standard_normal((5, 3)), [3, 2]))
    charges = jnp.asarray(rng.standard_normal((5, 1)))
    mod = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)
    eager = mod(charges, jg)
    jitted = eqx.filter_jit(lambda m, q, g: m(q, g))(mod, charges, jg)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12)
