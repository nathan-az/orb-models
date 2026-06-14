"""Compare the JAX differentiable-geometry boundary to the torch AtomGraphs.

Covers orb_models.common.atoms.jax.graph_batch:
  * to_jax                              (the torch -> pytree adapter)
  * apply_stress_displacement           (strain applied to positions + cell)
  * rotation_from_generator             (skew generator -> rotation matrix)
  * compute_differentiable_edge_vectors (forward PBC edge vectors)
  * forces / stress / equigrad          (jax.grad vs torch.autograd)

The geometry tests build the JAX inputs straight from numpy so they isolate the
maths from `to_jax`; a single dedicated test exercises `to_jax`.
"""

import numpy as np
import pytest
import torch

from orb_models.common.atoms import featurization
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb

# Two systems so per-graph batching (cell/displacement scatter) is exercised.
N_NODE = [3, 2]
# edges live within each graph; indices are already offset into the flat batch.
SENDERS = [0, 1, 2, 0, 3, 4]
RECEIVERS = [1, 2, 0, 2, 4, 3]
N_EDGE = [4, 2]


@pytest.fixture
def arrays():
    """Raw float64 numpy geometry for a 2-system periodic batch."""
    rng = np.random.default_rng(0)
    n_node = np.array(N_NODE, dtype=np.int64)
    n_edge = np.array(N_EDGE, dtype=np.int64)
    N, G, E = int(n_node.sum()), len(n_node), int(n_edge.sum())

    positions = rng.standard_normal((N, 3)) * 2.0
    # well-conditioned, positive-volume cells
    cell = np.stack(
        [np.eye(3) * 4.0 + 0.5 * rng.standard_normal((3, 3)) for _ in range(G)]
    )
    unit_shifts = rng.integers(-1, 2, size=(E, 3)).astype(np.float64)
    senders = np.array(SENDERS, dtype=np.int64)
    receivers = np.array(RECEIVERS, dtype=np.int64)
    per_node = np.repeat(np.arange(G), n_node)
    per_edge = np.repeat(np.arange(G), n_edge)
    return dict(
        n_node=n_node,
        n_edge=n_edge,
        N=N,
        G=G,
        E=E,
        positions=positions,
        cell=cell,
        unit_shifts=unit_shifts,
        senders=senders,
        receivers=receivers,
        per_node=per_node,
        per_edge=per_edge,
    )


def _torch_graph(a):
    """Build a torch AtomGraphs from the numpy `arrays` dict."""
    N = a["N"]
    atomic_numbers = torch.arange(N, dtype=torch.long)
    return AtomGraphs(
        senders=torch.tensor(a["senders"]),
        receivers=torch.tensor(a["receivers"]),
        n_node=torch.tensor(a["n_node"]),
        n_edge=torch.tensor(a["n_edge"]),
        node_features={
            "positions": torch.tensor(a["positions"]),
            "atomic_numbers": atomic_numbers,
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                atomic_numbers % 118, num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((a["E"], 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(a["unit_shifts"]),
        },
        system_features={
            "cell": torch.tensor(a["cell"]),
            "pbc": torch.ones((a["G"], 3), dtype=torch.bool),
        },
        node_targets={},
        edge_targets={},
        system_targets={},
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
    )


# --- to_jax -----------------------------------------------------------------
def test_to_jax(arrays):
    graph = _torch_graph(arrays)
    jg = jgb.to_jax(graph)

    np.testing.assert_array_equal(np.asarray(jg.senders), arrays["senders"])
    np.testing.assert_array_equal(np.asarray(jg.receivers), arrays["receivers"])
    np.testing.assert_array_equal(np.asarray(jg.n_node), arrays["n_node"])
    np.testing.assert_array_equal(np.asarray(jg.n_edge), arrays["n_edge"])
    np.testing.assert_array_equal(
        np.asarray(jg.per_node_graph_index), arrays["per_node"]
    )
    np.testing.assert_array_equal(
        np.asarray(jg.per_edge_graph_index), arrays["per_edge"]
    )
    np.testing.assert_allclose(
        np.asarray(jg.node_features["positions"]), arrays["positions"]
    )
    np.testing.assert_allclose(
        np.asarray(jg.system_features["cell"]), arrays["cell"]
    )
    assert jg.radius == 6.0


# --- rotation_from_generator ------------------------------------------------
def test_rotation_from_generator_matches_torch(helpers, arrays):
    rng = np.random.default_rng(1)
    gen = rng.standard_normal((arrays["G"], 3, 3))
    jax_R = jgb.rotation_from_generator(jnp_(gen))
    torch_R = featurization.rotation_from_generator(torch.tensor(gen))
    helpers.assert_close(jax_R, torch_R)


def test_rotation_from_generator_is_orthogonal(arrays):
    rng = np.random.default_rng(2)
    gen = rng.standard_normal((arrays["G"], 3, 3))
    R = np.asarray(jgb.rotation_from_generator(jnp_(gen)))
    eye = np.broadcast_to(np.eye(3), R.shape)
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), eye, atol=1e-10)
    # generator = 0 must give the identity (no rotation)
    R0 = np.asarray(
        jgb.rotation_from_generator(jnp_(np.zeros((arrays["G"], 3, 3))))
    )
    np.testing.assert_allclose(R0, eye, atol=1e-12)


# --- apply_stress_displacement ----------------------------------------------
@pytest.mark.parametrize("zero_displacement", [True, False])
def test_apply_stress_displacement(helpers, arrays, zero_displacement):
    rng = np.random.default_rng(3)
    disp = (
        np.zeros((arrays["G"], 3, 3))
        if zero_displacement
        else rng.standard_normal((arrays["G"], 3, 3)) * 0.05
    )
    jax_pos, jax_cell = jgb.apply_stress_displacement(
        jnp_(arrays["positions"]),
        jnp_(arrays["cell"]),
        jnp_(disp),
        jnp_(arrays["per_node"]),
    )

    # reference: the exact torch formula (positions + ..., cell + ...)
    d = torch.tensor(disp)
    sym = 0.5 * (d + d.transpose(-1, -2))
    pos = torch.tensor(arrays["positions"])
    cell = torch.tensor(arrays["cell"])
    per_node = torch.tensor(arrays["per_node"])
    ref_pos = pos + torch.bmm(pos.unsqueeze(1), sym[per_node]).squeeze(1)
    ref_cell = cell + torch.bmm(cell, sym)

    helpers.assert_close(jax_pos, ref_pos)
    helpers.assert_close(jax_cell, ref_cell)


# --- compute_differentiable_edge_vectors: forward ---------------------------
def test_edge_vectors_forward_matches_torch(helpers, arrays):
    graph = _torch_graph(arrays)
    torch_vectors, _, _ = graph.compute_differentiable_edge_vectors()

    jax_vectors = jgb.compute_differentiable_edge_vectors(
        jnp_(arrays["positions"]),
        jnp_(arrays["unit_shifts"]),
        jnp_(arrays["cell"]),
        jnp_(arrays["senders"]),
        jnp_(arrays["receivers"]),
        jnp_(arrays["per_node"]),
        jnp_(arrays["per_edge"]),
        jnp_(np.zeros((arrays["G"], 3, 3))),  # stress_displacement
        jnp_(np.zeros((arrays["G"], 3, 3))),  # generator
    )
    helpers.assert_close(jax_vectors, torch_vectors)


# --- forces / stress / equigrad: jax.grad vs torch.autograd -----------------
def test_force_stress_equigrad_match_torch(helpers, arrays):
    """The actual contract: d(scalar of vectors) w.r.t. positions/displacement/
    generator must agree, since forces/stress/rotational_grad are exactly those.
    """
    import jax

    # ---- torch reference (autograd through the AtomGraphs method) ----
    graph = _torch_graph(arrays)
    vectors, displacement, generator = (
        graph.compute_differentiable_edge_vectors()
    )
    loss = (vectors**2).sum()
    loss.backward()
    t_force = graph.node_features["positions"].grad
    t_stress = displacement.grad
    t_gen = generator.grad

    # ---- jax: differentiate the same scalar of the same function ----
    def energy(positions, disp, gen):
        v = jgb.compute_differentiable_edge_vectors(
            positions,
            jnp_(arrays["unit_shifts"]),
            jnp_(arrays["cell"]),
            jnp_(arrays["senders"]),
            jnp_(arrays["receivers"]),
            jnp_(arrays["per_node"]),
            jnp_(arrays["per_edge"]),
            disp,
            gen,
        )
        return (v**2).sum()

    zeros = jnp_(np.zeros((arrays["G"], 3, 3)))
    e, (g_pos, g_disp, g_gen) = jax.value_and_grad(energy, argnums=(0, 1, 2))(
        jnp_(arrays["positions"]), zeros, zeros
    )

    helpers.assert_close(e, loss)
    helpers.assert_close(g_pos, t_force)
    helpers.assert_close(g_disp, t_stress)
    helpers.assert_close(g_gen, t_gen)


def jnp_(x):
    import jax.numpy as jnp

    return jnp.asarray(x)
