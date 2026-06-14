"""Compare jax Encoder / AttentionInteractionNetwork to their PyTorch refs."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models import gns as torch_gns
from orb_models.common.models.jax.gnn import (
    AttentionInteractionNetwork,
    Encoder,
)

LATENT, N_LAYERS, HIDDEN = 8, 2, 16
N_NODES, N_EDGES = 6, 10


@pytest.fixture
def graph_arrays():
    """Random node/edge latents and a valid sender/receiver/cutoff topology."""
    rng = np.random.default_rng(0)
    senders = rng.integers(0, N_NODES, size=(N_EDGES,)).astype(np.int64)
    receivers = rng.integers(0, N_NODES, size=(N_EDGES,)).astype(np.int64)
    cutoff = rng.uniform(0.0, 1.0, size=(N_EDGES, 1))
    return rng, senders, receivers, cutoff


def test_encoder(helpers, key):
    rng = np.random.default_rng(0)
    n_node_in, n_edge_in = 7, 5
    torch_enc = torch_gns.Encoder(n_node_in, n_edge_in, LATENT, N_LAYERS, HIDDEN, activation="silu")
    jax_enc = helpers.copy_encoder(
        Encoder(n_node_in, n_edge_in, LATENT, N_LAYERS, HIDDEN, key, activation="silu"), torch_enc
    )

    node_feats = rng.standard_normal((N_NODES, n_node_in))
    edge_feats = rng.standard_normal((N_EDGES, n_edge_in))
    jax_nodes, jax_edges = jax_enc(jnp.asarray(node_feats), jnp.asarray(edge_feats))
    torch_nodes, torch_edges = torch_enc(torch.tensor(node_feats), torch.tensor(edge_feats))
    helpers.assert_close(jax_nodes, torch_nodes)
    helpers.assert_close(jax_edges, torch_edges)


@pytest.mark.parametrize("attention_gate", ["sigmoid", "softmax"])
@pytest.mark.parametrize("distance_cutoff", [False, True])
def test_attention_network_no_conditioning(
    helpers, key, graph_arrays, attention_gate, distance_cutoff
):
    rng, senders, receivers, cutoff = graph_arrays
    torch_ain = torch_gns.AttentionInteractionNetwork(
        LATENT, N_LAYERS, HIDDEN,
        attention_gate=attention_gate, distance_cutoff=distance_cutoff, activation="silu",
    )
    jax_ain = helpers.copy_attention_network(
        AttentionInteractionNetwork(
            LATENT, N_LAYERS, HIDDEN, key,
            attention_gate=attention_gate, distance_cutoff=distance_cutoff, activation="silu",
        ),
        torch_ain,
    )

    nodes = rng.standard_normal((N_NODES, LATENT))
    edges = rng.standard_normal((N_EDGES, LATENT))
    jax_out = jax_ain.forward(
        jnp.asarray(nodes), jnp.asarray(edges),
        jnp.asarray(senders), jnp.asarray(receivers), jnp.asarray(cutoff),
    )
    torch_out = torch_ain.forward(
        torch.tensor(nodes), torch.tensor(edges),
        torch.tensor(senders), torch.tensor(receivers), torch.tensor(cutoff),
    )
    helpers.assert_close(jax_out[0], torch_out[0])
    helpers.assert_close(jax_out[1], torch_out[1])


@pytest.mark.parametrize("conditioning", ["additive", "concatenative"])
def test_attention_network_conditioning(helpers, key, graph_arrays, conditioning):
    rng, senders, receivers, cutoff = graph_arrays
    torch_ain = torch_gns.AttentionInteractionNetwork(
        LATENT, N_LAYERS, HIDDEN, conditioning=conditioning, activation="silu"
    )
    jax_ain = helpers.copy_attention_network(
        AttentionInteractionNetwork(
            LATENT, N_LAYERS, HIDDEN, key, conditioning=conditioning, activation="silu"
        ),
        torch_ain,
    )

    nodes = rng.standard_normal((N_NODES, LATENT))
    edges = rng.standard_normal((N_EDGES, LATENT))
    cond_nodes = rng.standard_normal((N_NODES, LATENT))
    cond_edges = rng.standard_normal((N_EDGES, LATENT))
    jax_out = jax_ain.forward(
        jnp.asarray(nodes), jnp.asarray(edges),
        jnp.asarray(senders), jnp.asarray(receivers), jnp.asarray(cutoff),
        cond_nodes=jnp.asarray(cond_nodes), cond_edges=jnp.asarray(cond_edges),
    )
    torch_out = torch_ain.forward(
        torch.tensor(nodes), torch.tensor(edges),
        torch.tensor(senders), torch.tensor(receivers), torch.tensor(cutoff),
        cond_nodes=torch.tensor(cond_nodes), cond_edges=torch.tensor(cond_edges),
    )
    helpers.assert_close(jax_out[0], torch_out[0])
    helpers.assert_close(jax_out[1], torch_out[1])
