"""End-to-end MoleculeGNS equivalence *with* the charge/spin conditioner attached.

Same shape as test_molecule_gns, but a multi-graph batch carrying system-level
total_charge/spin_multiplicity, conditioning the backbone additively on nodes
(the OrbMol-v1 configuration). Confirms the conditioned forward matches torch and
is differentiable w.r.t. edge vectors (the force path).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models import gns as torch_gns
from orb_models.common.models.jax.conditioner import ChargeSpinConditioner
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.common.models.jax.rbf import BesselBasis
from orb_models.common.models.nn_util import (
    ChargeSpinConditioner as TorchChargeSpinConditioner,
)
from orb_models.common.models.rbf import BesselBasis as TorchBesselBasis

LATENT, STEPS, N_LAYERS, HIDDEN = 8, 2, 2, 16
NUM_BASES, N_OUT = 8, 3
N_NODE = [2, 3, 1]  # 6 nodes across 3 graphs
N_EDGES = 12


def _dummy_data():
    rng = np.random.default_rng(2)
    n_nodes = sum(N_NODE)
    per_node = np.concatenate([np.full(n, g) for g, n in enumerate(N_NODE)]).astype(np.int64)
    senders = rng.integers(0, n_nodes, size=(N_EDGES,)).astype(np.int64)
    receivers = rng.integers(0, n_nodes, size=(N_EDGES,)).astype(np.int64)
    return {
        "vectors": rng.standard_normal((N_EDGES, 3)),
        "senders": senders,
        "receivers": receivers,
        "atomic_numbers": rng.integers(1, 118, size=(n_nodes,)).astype(np.int64),
        "n_node": np.array(N_NODE, dtype=np.int64),
        "n_edge": np.array([4, 4, 4], dtype=np.int64),
        "per_node_graph_index": per_node,
        "per_edge_graph_index": per_node[senders],  # arbitrary but consistent
        "total_charge": np.array([1.0, -1.0, 0.0]),
        "spin_multiplicity": np.array([2.0, 1.0, 0.0]),
    }


def _batch(data, convert):
    return types.SimpleNamespace(
        node_features={"atomic_numbers": convert(data["atomic_numbers"])},
        edge_features={"vectors": convert(data["vectors"])},
        senders=convert(data["senders"]),
        receivers=convert(data["receivers"]),
        n_node=convert(data["n_node"]),
        n_edge=convert(data["n_edge"]),
        per_node_graph_index=convert(data["per_node_graph_index"]),
        per_edge_graph_index=convert(data["per_edge_graph_index"]),
        system_features={
            "total_charge": convert(data["total_charge"]),
            "spin_multiplicity": convert(data["spin_multiplicity"]),
        },
    )


def _build(key):
    kw = dict(
        latent_dim=LATENT,
        num_message_passing_steps=STEPS,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        use_embedding=True,
        num_node_out_features=N_OUT,
        activation="silu",
        conditioning_type="additive",
    )
    torch_model = torch_gns.MoleculeGNS(
        rbf_transform=TorchBesselBasis(6.0, num_bases=NUM_BASES),
        conditioner=TorchChargeSpinConditioner(LATENT),
        **kw,
    )
    jax_model = MoleculeGNS(
        rbf_transform=BesselBasis(6.0, num_bases=NUM_BASES),
        conditioner=ChargeSpinConditioner(LATENT, key=key),
        key=key,
        **kw,
    )
    return torch_model, jax_model


@pytest.mark.equivalence
def test_conditioned_gns_matches_torch(helpers, key):
    torch_model, jax_model = _build(key)
    jax_model = helpers.copy_molecule_gns(jax_model, torch_model)

    data = _dummy_data()
    jax_out = jax_model(_batch(data, jnp.asarray))
    with torch.no_grad():
        torch_out = torch_model(_batch(data, torch.tensor))

    for k in ("node_features", "edge_features", "pred"):
        helpers.assert_close(jax_out[k], torch_out[k])


def test_conditioned_gns_differentiable_wrt_vectors(key):
    _, jax_model = _build(key)
    base = _batch(_dummy_data(), jnp.asarray)

    def scalar(vectors):
        b = types.SimpleNamespace(
            node_features=base.node_features,
            edge_features={"vectors": vectors},
            senders=base.senders,
            receivers=base.receivers,
            n_node=base.n_node,
            n_edge=base.n_edge,
            per_node_graph_index=base.per_node_graph_index,
            per_edge_graph_index=base.per_edge_graph_index,
            system_features=base.system_features,
        )
        return jnp.sum(jax_model(b)["pred"])

    grad = jax.grad(scalar)(base.edge_features["vectors"])
    assert grad.shape == (N_EDGES, 3)
    assert np.isfinite(np.asarray(grad)).all()
