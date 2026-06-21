"""Compare the jax EnergyHead to the PyTorch reference, with shared weights.

Checks both the prediction (interaction + absolute energy) and the gradient
w.r.t. the incoming node features -- the latter is the head's slice of the force
autograd path (forces = -dE/dpos flows backbone <- head).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.forcefield.models.forcefield_heads import EnergyHead as TorchEnergyHead
from orb_models.forcefield.models.jax.forcefield_heads import EnergyHead

pytestmark = pytest.mark.equivalence

LATENT, N_LAYERS, HIDDEN = 8, 2, 16
N_NODE = [3, 4]  # two graphs -> N = 7


def _graphs():
    """Per-node graph index + n_node + atomic numbers for a 2-graph batch."""
    rng = np.random.default_rng(2)
    n_node = np.array(N_NODE, dtype=np.int64)
    per_node = np.repeat(np.arange(len(N_NODE)), N_NODE).astype(np.int64)
    atomic_numbers = rng.integers(1, 118, size=(per_node.shape[0],)).astype(np.int64)
    node_features = rng.standard_normal((per_node.shape[0], LATENT))
    return n_node, per_node, atomic_numbers, node_features


def _build(key):
    torch_head = TorchEnergyHead(
        latent_dim=LATENT,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        predict_atom_avg=True,
        activation="silu",
    ).eval()
    jax_head = EnergyHead(
        latent_dim=LATENT,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        predict_atom_avg=True,
        activation="silu",
        key=key,
    )
    return torch_head, jax_head


def test_energy_head_matches_torch(helpers, key):
    torch_head, jax_head = _build(key)
    jax_head = helpers.copy_energy_head(jax_head, torch_head)

    n_node, per_node, atomic_numbers, node_features = _graphs()
    jax_graph = types.SimpleNamespace(
        n_node=jnp.asarray(n_node),
        per_node_graph_index=jnp.asarray(per_node),
        node_features={"atomic_numbers": jnp.asarray(atomic_numbers)},
    )
    torch_batch = types.SimpleNamespace(
        n_node=torch.tensor(n_node),
        atomic_numbers=torch.tensor(atomic_numbers),
    )

    # --- predictions: interaction + absolute energy ---
    jax_interaction = jax_head(jnp.asarray(node_features), jax_graph)
    with torch.no_grad():
        torch_interaction = torch_head(torch.tensor(node_features), torch_batch)
    helpers.assert_close(jax_interaction, torch_interaction)

    jax_absolute = jax_head.absolute_energy(jax_interaction, jax_graph)
    with torch.no_grad():
        torch_absolute = torch_head.absolute_energy(torch_interaction, torch_batch)
    helpers.assert_close(jax_absolute, torch_absolute)


def test_energy_head_grad_matches_torch(helpers, key):
    """dE/d(node_features): the head's contribution to the force path."""
    torch_head, jax_head = _build(key)
    jax_head = helpers.copy_energy_head(jax_head, torch_head)

    n_node, per_node, atomic_numbers, node_features = _graphs()
    jax_graph = types.SimpleNamespace(
        n_node=jnp.asarray(n_node),
        per_node_graph_index=jnp.asarray(per_node),
        node_features={"atomic_numbers": jnp.asarray(atomic_numbers)},
    )
    torch_batch = types.SimpleNamespace(n_node=torch.tensor(n_node))

    jax_grad = jax.grad(lambda nf: jax_head(nf, jax_graph).sum())(jnp.asarray(node_features))

    nf_torch = torch.tensor(node_features, requires_grad=True)
    torch_head(nf_torch, torch_batch).sum().backward()

    helpers.assert_close(jax_grad, nf_torch.grad)
