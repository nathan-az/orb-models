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

    # --- padding bookkeeping (None on an unpadded graph) ---------------------
    # Number of *real* graphs once `pad_to_bucket` has appended a dummy padding
    # graph (jraph `pad_with_graphs` convention). Real graphs occupy slots
    # [0, n_real_graph); the absorbing padding graph sits at slot `n_real_graph`
    # and any further slots are empty padding graphs. Every loss/aggregation mask
    # derives from this single scalar (see `real_*_mask`). A traced () array, NOT
    # static: it varies per batch but never changes a shape, so jit caches once.
    # `None` means "not padded" -> all masks are all-True (legacy / single-graph).
    n_real_graph: jax.Array | None = None


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
        # Every graph the adapter produces is fully real until `pad_to_bucket`
        # appends a padding graph; record the count so the masks have the right
        # cutoff even on an already-disjoint-batched torch graph.
        n_real_graph=jnp.asarray(graph.n_node.shape[0], dtype=jnp.int32),
    )


# --- masking: which rows are real vs padding ---------------------------------
# All three derive from the single `n_real_graph` scalar (jraph convention: real
# graphs are slots [0, n_real_graph), everything padded is >= n_real_graph). A
# `None` count means the graph was never padded, so every row is real.


def real_graph_mask(graph: JaxAtomGraphs) -> jax.Array:
    """(G,) bool. True on real graphs, False on the absorbing/empty padding graphs."""
    g = graph.n_node.shape[0]
    if graph.n_real_graph is None:
        return jnp.ones((g,), dtype=bool)
    return jnp.arange(g) < graph.n_real_graph


def real_node_mask(graph: JaxAtomGraphs) -> jax.Array:
    """(N,) bool. True on real atoms; padding atoms live in the padding graph."""
    if graph.n_real_graph is None:
        return jnp.ones((graph.per_node_graph_index.shape[0],), dtype=bool)
    return graph.per_node_graph_index < graph.n_real_graph


# --- packing: the grouping POLICY (the disjoint concat is torch's) ------------


def pack_graphs(graphs: list["AtomGraphs"], n_max: int, e_max: int) -> list["AtomGraphs"]:
    """Greedily group single-system torch `AtomGraphs` into disjoint batches under
    (n_max, e_max). Only the grouping *policy* is the JAX path's concern; the actual
    disjoint concatenation (index offsets, `per_*_graph_index`, `n_node`/`n_edge`) is
    delegated to torch `AtomGraphs.batch`, which already does exactly that -- we do
    NOT re-implement offsetting on the JAX side.

    First-fit to a budget: walk the list (caller shuffles first each epoch for
    unbiased gradients), accumulating systems until the next would push either the
    node total over `n_max` or the edge total over `e_max`, then start a new batch.
    Because a cutoff graph has `E ~ avg_degree * N`, `e_max` is the binding budget
    and `n_max` is mostly a guard; both caps leave headroom for `pad_to_bucket` to
    top each batch up to the fixed bucket shape. A single system larger than a cap
    becomes its own (over-budget) batch -- pick the bucket to dominate it.

    Returns batched torch `AtomGraphs`, each ready for `to_jax` then `pad_to_bucket`.
    """
    # Deferred import keeps this module torch-free at import time; torch is only
    # touched when the packer actually runs (host-side, before `to_jax`).
    from orb_models.common.atoms.batch.graph_batch import AtomGraphs

    batches: list["AtomGraphs"] = []
    current: list["AtomGraphs"] = []
    cur_n = cur_e = 0
    for g in graphs:
        n = int(g.n_node.sum())
        e = int(g.n_edge.sum())
        if current and (cur_n + n > n_max or cur_e + e > e_max):
            batches.append(AtomGraphs.batch(current))
            current, cur_n, cur_e = [], 0, 0
        current.append(g)
        cur_n += n
        cur_e += e
    if current:
        batches.append(AtomGraphs.batch(current))
    return batches


# --- padding: top a packed batch up to a fixed bucket shape ------------------


def _pad_axis0(x: jax.Array, target: int, fill=0) -> jax.Array:
    """Pad `x` along axis 0 up to `target` rows with constant `fill`."""
    pad = target - x.shape[0]
    if pad == 0:
        return x
    widths = [(0, pad)] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, widths, constant_values=fill)


def pad_to_bucket(
    graph: JaxAtomGraphs, n_pad: int, e_pad: int, g_pad: int
) -> JaxAtomGraphs:
    """Top a packed batch up to fixed `(n_pad, e_pad, g_pad)` so jit compiles ONCE.

    Appends one dummy padding graph (jraph `pad_with_graphs` convention) at slot
    `g = n_real_graph` that absorbs every slack node/edge; remaining graph slots are
    empty (`n_node = n_edge = 0`). The bucket shape -- not the system contents --
    is what `eqx.filter_jit` keys on, so every batch that fits one bucket reuses the
    same compiled step.

    Two padding choices keep the autodiff clean despite the masked-out padding:
      * padding edges are self-loops on the first padding atom (`senders =
        receivers = N`) so `segment_sum(num_segments=n_pad)` deposits their message
        on a padding atom, never a real one;
      * those edges get a nonzero `unit_shifts` ([1,0,0]) and the padding graphs get
        an identity `cell`, so the recomputed edge `vectors` are nonzero. A zero
        vector would make `||v||` -- and hence the force grad through it -- NaN, and
        `0 * NaN` survives the loss mask. Identity cells also give the padding graphs
        a finite (unit) volume, so the `stress = dE/d(disp) / volume` divide is safe.
    """
    n = graph.per_node_graph_index.shape[0]
    e = graph.per_edge_graph_index.shape[0]
    g = graph.n_node.shape[0]
    if n_pad < n or e_pad < e or g_pad < g + 1:
        raise ValueError(
            f"bucket ({n_pad},{e_pad},{g_pad}) too small for packed batch "
            f"({n},{e},{g}); need n_pad>=n, e_pad>=e, g_pad>=n_graphs+1."
        )

    # The absorbing padding graph occupies slot `g`; padding atoms/edges point at it.
    node_idx = _pad_axis0(graph.per_node_graph_index, n_pad, fill=g)
    edge_idx = _pad_axis0(graph.per_edge_graph_index, e_pad, fill=g)
    # Padding edges are self-loops on the first padding atom (index n).
    senders = _pad_axis0(graph.senders, e_pad, fill=n)
    receivers = _pad_axis0(graph.receivers, e_pad, fill=n)

    # n_node/n_edge: real counts, then the slack lands on the absorbing graph, then
    # zeros for any trailing empty graphs.
    extra = jnp.zeros((g_pad - g,), dtype=graph.n_node.dtype)
    n_node = jnp.concatenate([graph.n_node, extra]).at[g].set(n_pad - n)
    n_edge = jnp.concatenate([graph.n_edge, extra]).at[g].set(e_pad - e)

    def pad_nodes(d):
        return {k: _pad_axis0(v, n_pad) for k, v in d.items()}

    def pad_edges(d):
        out = {}
        for k, v in d.items():
            # Nonzero shift on padding edges -> nonzero vectors -> no NaN force grad.
            fill = 1 if k == "unit_shifts" else 0
            out[k] = _pad_axis0(v, e_pad, fill=fill)
            if k == "unit_shifts":
                out[k] = out[k].at[e:, 1:].set(0)  # exactly [1,0,0] per padding edge
        return out

    def pad_graphs(d):
        out = {}
        for k, v in d.items():
            if k == "cell":  # identity cell on padding graphs -> finite unit volume
                eye = jnp.broadcast_to(jnp.eye(3, dtype=v.dtype), (g_pad - g, 3, 3))
                out[k] = jnp.concatenate([v, eye], axis=0)
            else:
                out[k] = _pad_axis0(v, g_pad)
        return out

    return JaxAtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        node_features=pad_nodes(graph.node_features),
        edge_features=pad_edges(graph.edge_features),
        system_features=pad_graphs(graph.system_features),
        per_node_graph_index=node_idx,
        per_edge_graph_index=edge_idx,
        radius=graph.radius,
        n_real_graph=jnp.asarray(g, dtype=jnp.int32),
    )


def pad_targets(
    targets: dict[str, jax.Array], n_pad: int, g_pad: int
) -> dict[str, jax.Array]:
    """Pad supervised targets to a graph's bucket shape (zeros; masked out of loss).

    Per-node targets (`forces`, (N,3)) grow to `n_pad`; per-graph targets (`energy`
    (G,), `stress` (G,6)) grow to `g_pad`. The fill value is irrelevant -- the loss
    mask drops every padded row -- but it must be finite to keep grads clean.
    """
    out = {}
    for k, v in targets.items():
        target = n_pad if k == "forces" else g_pad
        out[k] = _pad_axis0(v, target)
    return out


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
