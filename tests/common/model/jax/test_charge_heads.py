"""Equivalence tests for the OrbMol-v2 per-atom heads against torch:
LatentChargeHead, LatentSpinHead, ChargeConditionedEnergyHead. All run in fp64
with shared weights, so a correct port agrees to ~1e-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.forcefield_heads import (
    ChargeConditionedEnergyHead as TorchChargeConditionedEnergyHead,
    LatentChargeHead as TorchLatentChargeHead,
    LatentSpinHead as TorchLatentSpinHead,
)
from orb_models.forcefield.models.jax.forcefield_heads import (
    ChargeConditionedEnergyHead,
    LatentChargeHead,
    LatentSpinHead,
)

LATENT = 8
N_NODE = [3, 2]
TOTAL_CHARGE = [1.0, -1.0]
SPIN_MULTIPLICITY = [2.0, 1.0]


def _torch_graph(with_charge=True, with_spin=True):
    rng = np.random.default_rng(0)
    n_node = np.array(N_NODE, dtype=np.int64)
    N, G = int(n_node.sum()), len(n_node)
    z = rng.integers(1, 30, size=(N,)).astype(np.int64)
    system_features = {
        "cell": torch.zeros((G, 3, 3), dtype=torch.float64),
        "pbc": torch.zeros((G, 3), dtype=torch.bool),
    }
    if with_charge:
        system_features["total_charge"] = torch.tensor(TOTAL_CHARGE)
    if with_spin:
        system_features["spin_multiplicity"] = torch.tensor(SPIN_MULTIPLICITY)
    return AtomGraphs(
        senders=torch.zeros(0, dtype=torch.long),
        receivers=torch.zeros(0, dtype=torch.long),
        n_node=torch.tensor(n_node),
        n_edge=torch.zeros(G, dtype=torch.long),
        node_features={
            "positions": torch.tensor(rng.standard_normal((N, 3))),
            "atomic_numbers": torch.tensor(z),
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                torch.tensor(z - 1), num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((0, 3), dtype=torch.float64),
            "unit_shifts": torch.zeros((0, 3), dtype=torch.float64),
        },
        system_features=system_features,
        node_targets={}, edge_targets={}, system_targets={},
        system_id=None, fix_atoms=None, tags=None,
        radius=6.0, max_num_neighbors=torch.tensor([20]),
    )


def _node_features(N, seed=1):
    return np.random.default_rng(seed).standard_normal((N, LATENT))


# ---- LatentChargeHead ----


@pytest.mark.parametrize("with_charge", [True, False], ids=["total_charge", "no_total_charge"])
@pytest.mark.parametrize("enforce", [True, False], ids=["enforce", "raw"])
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_latent_charge_head_matches_torch(helpers, key, with_charge, enforce, scale):
    tg = _torch_graph(with_charge=with_charge)
    jg = jgb.to_jax(tg)
    N = int(np.sum(N_NODE))
    nf = _node_features(N)

    torch_head = TorchLatentChargeHead(
        LATENT, num_mlp_layers=2, mlp_hidden_dim=16,
        enforce_total_charge=enforce, charge_scale=scale, activation="silu",
    ).double().eval()
    jax_head = LatentChargeHead(
        LATENT, num_mlp_layers=2, mlp_hidden_dim=16, key=key,
        enforce_total_charge=enforce, charge_scale=scale,
    )
    jax_head = helpers.copy_latent_charge_head(jax_head, torch_head)

    t_out = torch_head(torch.tensor(nf), tg)
    j_out = jax_head(jnp.asarray(nf), jg)
    helpers.assert_close(j_out, t_out)

    if enforce and with_charge:
        # Each system sums to its total charge / scale.
        sums = np.asarray(jax.ops.segment_sum(j_out[:, 0], jg.per_node_graph_index, 2))
        np.testing.assert_allclose(sums / scale, TOTAL_CHARGE, atol=1e-9)


def test_latent_spin_head_matches_torch(helpers, key):
    tg = _torch_graph(with_spin=True)
    jg = jgb.to_jax(tg)
    N = int(np.sum(N_NODE))
    nf = _node_features(N, seed=2)

    torch_head = TorchLatentSpinHead(LATENT, num_mlp_layers=2, mlp_hidden_dim=16, activation="silu").double().eval()
    jax_head = LatentSpinHead(LATENT, num_mlp_layers=2, mlp_hidden_dim=16, key=key)
    jax_head = helpers.copy_latent_charge_head(jax_head, torch_head)

    t_out = torch_head(torch.tensor(nf), tg)
    j_out = jax_head(jnp.asarray(nf), jg)
    helpers.assert_close(j_out, t_out)

    # Each system sums to 2S = multiplicity - 1.
    sums = np.asarray(jax.ops.segment_sum(j_out[:, 0], jg.per_node_graph_index, 2))
    np.testing.assert_allclose(sums, np.array(SPIN_MULTIPLICITY) - 1, atol=1e-9)


# ---- ChargeConditionedEnergyHead ----


@pytest.mark.parametrize("use_spins", [False, True], ids=["charge_only", "charge_spin"])
def test_charge_conditioned_energy_head_matches_torch(helpers, key, use_spins):
    tg = _torch_graph()
    jg = jgb.to_jax(tg)
    N = int(np.sum(N_NODE))
    nf = _node_features(N, seed=3)
    rng = np.random.default_rng(4)
    charges = rng.standard_normal((N, 1))
    spins = rng.standard_normal((N, 1)) if use_spins else None

    torch_head = TorchChargeConditionedEnergyHead(
        LATENT, num_mlp_layers=2, mlp_hidden_dim=16, use_spins=use_spins, activation="silu",
    ).double().eval()
    jax_head = ChargeConditionedEnergyHead(
        LATENT, num_mlp_layers=2, mlp_hidden_dim=16, key=key, use_spins=use_spins,
    )
    jax_head = helpers.copy_charge_conditioned_energy_head(jax_head, torch_head)

    t_charges = torch.tensor(charges)
    t_spins = torch.tensor(spins) if use_spins else None
    t_out = torch_head(torch.tensor(nf), tg, t_charges, t_spins)

    j_out = jax_head(
        jnp.asarray(nf), jg, jnp.asarray(charges),
        jnp.asarray(spins) if use_spins else None,
    )
    helpers.assert_close(j_out, t_out)


def _graph_with_n_node(n_node):
    """Minimal non-periodic jax graph with the given per-system node counts."""
    n_node = np.asarray(n_node, dtype=np.int64)
    N, G = int(n_node.sum()), n_node.shape[0]
    z = np.ones(N, dtype=np.int64)
    tg = AtomGraphs(
        senders=torch.zeros(0, dtype=torch.long),
        receivers=torch.zeros(0, dtype=torch.long),
        n_node=torch.tensor(n_node),
        n_edge=torch.zeros(G, dtype=torch.long),
        node_features={
            "positions": torch.zeros((N, 3), dtype=torch.float64),
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
    return jgb.to_jax(tg)


def test_charge_conditioned_energy_head_size_consistency(key):
    """Two copies of a system give exactly 2x the single energy (sum-pooling =>
    size-consistent), independent of charge values."""
    jax_head = ChargeConditionedEnergyHead(
        LATENT, num_mlp_layers=2, mlp_hidden_dim=16, key=key,
    )
    nf = jnp.asarray(_node_features(3, seed=5))
    charges = jnp.asarray(np.random.default_rng(6).standard_normal((3, 1)))

    e1 = jax_head(nf, _graph_with_n_node([3]), charges)
    e2 = jax_head(
        jnp.concatenate([nf, nf], 0),
        _graph_with_n_node([3, 3]),
        jnp.concatenate([charges, charges], 0),
    )
    np.testing.assert_allclose(np.asarray(e2.sum()), 2 * np.asarray(e1.sum()), atol=1e-9)
