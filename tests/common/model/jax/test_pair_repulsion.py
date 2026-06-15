"""Compare the jax ZBL energy (and its grad w.r.t. edge vectors) to torch.

All ZBL tensors are fixed physical constants, so there is nothing to copy -- both
sides construct identical buffers. The grad-through-vectors check is ZBL's slice
of the conservative force/stress path.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import torch

from orb_models.forcefield.models.jax.pair_repulsion import ZBLBasis
from orb_models.forcefield.models.pair_repulsion import ZBLBasis as TorchZBLBasis

N_NODE = [3, 4]  # two graphs -> N = 7
N_EDGES = 12


def _data():
    rng = np.random.default_rng(3)
    n_node = np.array(N_NODE, dtype=np.int64)
    per_node = np.repeat(np.arange(len(N_NODE)), N_NODE).astype(np.int64)
    n = per_node.shape[0]
    z = rng.integers(1, 30, size=(n,)).astype(np.int64)  # physical atomic numbers
    # edges within the same graph so aggregation is physically sensible
    senders = rng.integers(0, n, size=(N_EDGES,)).astype(np.int64)
    receivers = ((senders + rng.integers(1, n, size=(N_EDGES,))) % n).astype(np.int64)
    vectors = rng.standard_normal((N_EDGES, 3)) * 1.5
    return n_node, per_node, z, senders, receivers, vectors


def _jax_graph(n_node, per_node, z, senders, receivers, vectors):
    one_hot = np.eye(118)[z - 1]  # argmax(one_hot)+1 == physical Z, matching torch
    return types.SimpleNamespace(
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        n_node=jnp.asarray(n_node),
        per_node_graph_index=jnp.asarray(per_node),
        node_features={
            "atomic_numbers": jnp.asarray(z),
            "atomic_numbers_embedding": jnp.asarray(one_hot),
        },
        edge_features={"vectors": jnp.asarray(vectors)},
    )


def _torch_batch(n_node, z, senders, receivers, vectors):
    one_hot = np.eye(118)[z - 1]  # argmax(one_hot)+1 == physical Z
    return types.SimpleNamespace(
        senders=torch.tensor(senders),
        receivers=torch.tensor(receivers),
        n_node=torch.tensor(n_node),
        node_features={"atomic_numbers_embedding": torch.tensor(one_hot)},
        edge_features={"vectors": torch.tensor(vectors)},
    )


def test_zbl_energy_matches_torch(helpers):
    n_node, per_node, z, senders, receivers, vectors = _data()
    jax_zbl = ZBLBasis(p=6, node_aggregation="sum")
    torch_zbl = TorchZBLBasis(p=6, node_aggregation="sum", compute_gradients=False)

    jax_e = jax_zbl(_jax_graph(n_node, per_node, z, senders, receivers, vectors))
    with torch.no_grad():
        torch_e = torch_zbl(_torch_batch(n_node, z, senders, receivers, vectors))["energy"]
    helpers.assert_close(jax_e, torch_e)


def test_zbl_grad_wrt_vectors_matches_torch(helpers):
    """dE_ZBL/d(vectors): ZBL's contribution to the autograd force/stress path."""
    n_node, per_node, z, senders, receivers, vectors = _data()
    jax_zbl = ZBLBasis(p=6, node_aggregation="sum")
    torch_zbl = TorchZBLBasis(p=6, node_aggregation="sum", compute_gradients=False)

    def jax_scalar(v):
        g = _jax_graph(n_node, per_node, z, senders, receivers, v)
        return jax_zbl(g).sum()

    jax_grad = jax.grad(jax_scalar)(jnp.asarray(vectors))

    v_torch = torch.tensor(vectors, requires_grad=True)
    batch = _torch_batch(n_node, z, senders, receivers, vectors)
    batch.edge_features["vectors"] = v_torch
    torch_zbl(batch)["energy"].sum().backward()

    helpers.assert_close(jax_grad, v_torch.grad)
