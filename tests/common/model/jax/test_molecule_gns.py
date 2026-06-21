"""Compare the full jax MoleculeGNS.forward to the PyTorch reference.

This is the end-to-end check: featurize_nodes/edges -> encoder -> the stack of
attention interaction networks -> decoder, with every weight shared across the
two frameworks. It covers both node-featurization paths (learned atom-type
embedding vs. one-hot input) and confirms the whole stack is differentiable
w.r.t. edge vectors (the input the force path differentiates).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models import gns as torch_gns
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.common.models.jax.rbf import BesselBasis
from orb_models.common.models.rbf import BesselBasis as TorchBesselBasis

LATENT, STEPS, N_LAYERS, HIDDEN = 8, 2, 2, 16
NUM_BASES, N_NODES, N_EDGES, N_OUT = 8, 6, 12, 3


def _dummy_data(use_embedding):
    """Random but valid node/edge arrays as plain numpy (framework-agnostic)."""
    rng = np.random.default_rng(1)
    data = {
        "vectors": rng.standard_normal((N_EDGES, 3)),  # norms ~1.7, well under r_max=6
        "senders": rng.integers(0, N_NODES, size=(N_EDGES,)).astype(np.int64),
        "receivers": rng.integers(0, N_NODES, size=(N_EDGES,)).astype(np.int64),
    }
    if use_embedding:
        data["atomic_numbers"] = rng.integers(1, 118, size=(N_NODES,)).astype(np.int64)
    else:
        z = rng.integers(0, 118, size=(N_NODES,))
        data["atomic_numbers_embedding"] = np.eye(118)[z]
    return data


def _batch(data, convert):
    """Pack arrays into the attribute/dict shape both forwards expect."""
    node_key = "atomic_numbers" if "atomic_numbers" in data else "atomic_numbers_embedding"
    return types.SimpleNamespace(
        node_features={node_key: convert(data[node_key])},
        edge_features={"vectors": convert(data["vectors"])},
        senders=convert(data["senders"]),
        receivers=convert(data["receivers"]),
    )


def _build(use_embedding, key):
    kw = dict(
        latent_dim=LATENT,
        num_message_passing_steps=STEPS,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        use_embedding=use_embedding,
        num_node_out_features=N_OUT,
        activation="silu",
    )
    torch_model = torch_gns.MoleculeGNS(
        rbf_transform=TorchBesselBasis(6.0, num_bases=NUM_BASES), **kw
    )
    jax_model = MoleculeGNS(
        rbf_transform=BesselBasis(6.0, num_bases=NUM_BASES), key=key, **kw
    )
    return torch_model, jax_model


@pytest.mark.equivalence
@pytest.mark.parametrize("use_embedding", [False, True])
def test_molecule_gns_forward_matches_torch(helpers, key, use_embedding):
    torch_model, jax_model = _build(use_embedding, key)
    jax_model = helpers.copy_molecule_gns(jax_model, torch_model)

    data = _dummy_data(use_embedding)
    jax_out = jax_model(_batch(data, jnp.asarray))
    with torch.no_grad():
        torch_out = torch_model(_batch(data, torch.tensor))

    for k in ("node_features", "edge_features", "pred"):
        helpers.assert_close(jax_out[k], torch_out[k])


def test_molecule_gns_differentiable_wrt_vectors(key):
    """grad of the output through the whole stack (incl. the StableNormalize
    custom jvp) must be finite -- this is the autodiff path forces ride on."""
    _, jax_model = _build(use_embedding=True, key=key)
    batch = _batch(_dummy_data(use_embedding=True), jnp.asarray)

    def scalar(vectors):
        b = types.SimpleNamespace(
            node_features=batch.node_features,
            edge_features={"vectors": vectors},
            senders=batch.senders,
            receivers=batch.receivers,
        )
        return jnp.sum(jax_model(b)["pred"])

    grad = jax.grad(scalar)(batch.edge_features["vectors"])
    assert grad.shape == (N_EDGES, 3)
    assert np.isfinite(np.asarray(grad)).all()
