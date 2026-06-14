"""JAX side of the AtomGraphs

  * The torch `AtomGraphs` + `graph_featurization` pipeline (PBC wrap -> supercell
    -> KNN neighbour list) STAYS in torch. It is one-time CPU preprocessing,
    is non-differentiable, and relies on scipy/CUDA kernels with no JAX analogue.
  * This module only owns (1) a pytree container of the arrays the model needs,
    (2) a thin boundary adapter from the torch container, and (3) the *one*
    differentiable piece: edge-vector / stress-displacement / rotation geometry,
    which must be JAX so forces/stress = jax.grad of energy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    # Only for type hints in `to_jax`. The torch AtomGraphs is the *input* and is
    # never imported at runtime by the JAX path.
    from torch import Tensor

    from orb_models.common.atoms.batch.graph_batch import AtomGraphs


class JaxAtomGraphs(eqx.Module):
    """Pytree mirror of the arrays the JAX model + force path consume.

    A NamedTuple is automatically a JAX pytree, so jax.tree_util / device_put /
    grad-through-positions all work. Every field is a leaf:

    NOTE on jit: `radius` (and arguably n_node/n_edge, which set output shapes)
    are effectively static. For a first pass you are not jitting the whole graph,
    so leave them as leaves. When you do jit, either mark them static or pull the
    flat int arrays out and pass radius as a Python float closed over the fn.
    """

    # --- connectivity (the disjoint-graph batch: one big graph, indices offset) -
    senders: jax.Array  # (E,) int   source node per directed edge
    receivers: jax.Array  # (E,) int   dest   node per directed edge
    n_node: jax.Array  # (G,) int   atoms per system
    n_edge: jax.Array  # (G,) int   edges per system

    # --- features (see key/shape table above) -------------------------------
    node_features: dict[str, jax.Array]
    edge_features: dict[str, jax.Array]
    system_features: dict[str, jax.Array]

    # --- precomputed index helpers (cheap; compute once in `to_jax`) ---------
    # Replaces torch's _get_per_node/edge_graph_indices (searchsorted). Needed by
    # the stress-displacement / rotation maths to scatter per-graph 3x3 matrices
    # onto per-node / per-edge rows.
    per_node_graph_index: jax.Array  # (N,) int   graph id of each node
    per_edge_graph_index: jax.Array  # (E,) int   graph id of each edge

    # --- static-ish scalars --------------------------------------------------
    radius: float = eqx.field(static=True)


def torch_to_jax(tensor: "Tensor") -> jax.Array:
    return jnp.asarray(tensor.detach().cpu().numpy())


def to_jax(graph: "AtomGraphs") -> JaxAtomGraphs:
    return JaxAtomGraphs(
        senders=torch_to_jax(graph.senders),
        receivers=torch_to_jax(graph.receivers),
        n_node=torch_to_jax(graph.n_node),
        n_edge=torch_to_jax(graph.n_edge),
        node_features=jax.tree.map(torch_to_jax, graph.node_features),
        edge_features=jax.tree.map(torch_to_jax, graph.edge_features),
        system_features=jax.tree.map(torch_to_jax, graph.system_features),
        per_node_graph_index=torch_to_jax(graph.node_batch_index),
        per_edge_graph_index=torch_to_jax(graph._get_per_edge_graph_indices()),
        radius=graph.radius,
    )


def compute_differentiable_edge_vectors(
    positions: jax.Array,  # (N, 3)  DIFFERENTIATE w.r.t. this -> forces
    unit_shifts: jax.Array,  # (E, 3)  integer image offsets per edge (static)
    cell: jax.Array,  # (G, 3, 3)
    senders: jax.Array,  # (E,)
    receivers: jax.Array,  # (E,)
    per_node_graph_index: jax.Array,  # (N,)
    per_edge_graph_index: jax.Array,  # (E,)
    stress_displacement: jax.Array,  # (G, 3, 3) DIFFERENTIATE w.r.t. this -> stress
    generator: jax.Array,  # (G, 3, 3) DIFFERENTIATE w.r.t. this -> rotational_grad
) -> jax.Array:
    positions, cell = apply_stress_displacement(
        positions, cell, stress_displacement, per_node_graph_index
    )
    rotation = rotation_from_generator(generator)
    positions = jnp.einsum(
        "ni,nij->nj", positions, rotation[per_node_graph_index]
    )
    cell = jnp.einsum("gij,gjk->gik", cell, rotation)

    shifts = jnp.einsum("ei,eij->ej", unit_shifts, cell[per_edge_graph_index])
    vectors = positions[receivers] - positions[senders] + shifts
    return vectors


def apply_stress_displacement(
    positions: jax.Array,  # (N, 3)
    cell: jax.Array,  # (G, 3, 3)
    displacement: jax.Array,  # (G, 3, 3)  caller passes zeros; grad target = stress
    per_node_graph_index: jax.Array,  # (N,)
) -> tuple[jax.Array, jax.Array]:
    symmetric_displacement = 0.5 * (
        displacement + displacement.swapaxes(-1, -2)
    )
    positions = positions + jnp.einsum(
        "ni,nij->nj", positions, symmetric_displacement[per_node_graph_index]
    )
    cell = cell + jnp.einsum("gij,gjk->gik", cell, symmetric_displacement)
    return positions, cell


def rotation_from_generator(generator: jax.Array) -> jax.Array:
    """(G, 3, 3) skew/antisymmetric-ish generator -> (G, 3, 3) rotation matrices."""
    return jax.scipy.linalg.expm(generator - generator.swapaxes(-1, -2))
