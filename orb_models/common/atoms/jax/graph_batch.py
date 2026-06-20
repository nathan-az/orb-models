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
import numpy as np

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
    # Number of *real* graphs once `to_padded_numpy` has appended a dummy padding
    # graph (jraph `pad_with_graphs` convention). Real graphs occupy slots
    # [0, n_real_graph); the absorbing padding graph sits at slot `n_real_graph`
    # and any further slots are empty padding graphs. Every loss/aggregation mask
    # derives from this single scalar (see `real_*_mask`). A traced () array, NOT
    # static: it varies per batch but never changes a shape, so jit caches once.
    # `None` means "not padded" -> all masks are all-True (legacy / single-graph).
    n_real_graph: jax.Array | None = None

    # --- OrbMol-v2 periodic electrostatics prep (None unless periodic Coulomb) ----
    # Host-prepped jax-pme batch containers `(sr_batch, nonperiodic_batch,
    # periodic_batch)` from `forcefield.models.jax.pme.build_pme_batch`, sized to this
    # graph's bucket. Carries the fixed PME real-space neighbour list + k-grid + masks
    # (the non-jittable boundary). Read only by the periodic branch of
    # `conservative_regressor.energy_fn`; a tuple of namedtuple pytrees, so it threads
    # through jit. `None` -> non-periodic (direct dense Coulomb) or no electrostatics.
    pme_prep: tuple | None = None


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
        # Every graph the adapter produces is fully real until `to_padded_numpy`
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


def extract_targets(
    graph: "AtomGraphs", *, has_stress: bool = True
) -> dict[str, jax.Array]:
    """Pull energy/forces/stress off a torch `AtomGraphs` batch into a JAX dict.

    Keys match the loss/`property_definitions` fullnames. Per `_total_loss`: energy
    is ``(G,)`` absolute, forces ``(N, 3)``, stress ``(G, 6)`` Voigt. The graph
    adapter stores per-graph targets with a trailing dim, so energy/stress are
    reshaped. (The training path uses `to_padded_numpy`, which extracts + pads in one
    pass; this standalone helper is for probes/tests that want the unpadded targets.)
    """
    targets = {
        "energy": torch_to_jax(graph.system_targets["energy"]).reshape(-1),  # (G,)
        "forces": torch_to_jax(graph.node_targets["forces"]),  # (N, 3)
    }
    if has_stress:
        targets["stress"] = torch_to_jax(graph.system_targets["stress"]).reshape(-1, 6)
    return targets


# --- host-side (numpy) padded prep -------------------------------------------
# THE conversion+padding path: torch `AtomGraphs` -> fixed-bucket padded arrays,
# built on the host with np.pad/np.concatenate then handed to the model via a single
# `jax.device_put`. Doing it in numpy (rather than eager `jnp` ops) matters: the jnp
# route dispatched ~20 tiny XLA ops per batch whose eager compile-cache missed on the
# varying real shape -- ~300ms/batch. numpy is ~3ms and parallelises across DataLoader
# workers (where there is no device anyway). See training/profile_prep.py.


def _np_pad_axis0(x: np.ndarray, target: int, fill=0) -> np.ndarray:
    """np.pad along axis 0 up to `target` rows with constant `fill`."""
    pad = target - x.shape[0]
    if pad == 0:
        return x
    widths = [(0, pad)] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, widths, constant_values=fill)


def to_padded_numpy(
    graph: "AtomGraphs",
    n_pad: int,
    e_pad: int,
    g_pad: int,
    *,
    has_stress: bool,
    reference_coefficients: np.ndarray | None = None,
) -> tuple[JaxAtomGraphs, dict[str, np.ndarray]]:
    """Build the fixed-bucket padded ``(JaxAtomGraphs, targets)`` entirely in numpy.

    Tops a packed batch up to fixed ``(n_pad, e_pad, g_pad)`` so ``eqx.filter_jit``
    compiles ONCE: appends one absorbing padding graph (jraph ``pad_with_graphs``
    convention) at slot ``g`` that soaks up the slack nodes/edges. The padding is
    autodiff-safe -- padding edges are self-loops on the first padding atom (so
    ``segment_sum`` never deposits on a real atom), get a nonzero ``[1,0,0]``
    ``unit_shifts`` (nonzero edge ``vectors`` -> no ``||v||`` NaN force grad), and the
    padding graphs get an identity ``cell`` (finite unit volume for the stress divide).
    Every leaf is a host ``np.ndarray``; intended as a DataLoader ``collate_fn`` step so
    the cost runs in workers, with the train loop doing one ``jax.device_put``.
    """
    npf = lambda t: t.detach().cpu().numpy()
    # Targets first (read-only, before anything touches the torch batch). A batch with
    # no supervised targets (e.g. forward-only equivalence tests) -> empty dict.
    targets: dict[str, np.ndarray] = {}
    if "energy" in graph.system_targets:
        targets["energy"] = npf(graph.system_targets["energy"]).reshape(-1)
        targets["forces"] = npf(graph.node_targets["forces"])
        if has_stress:
            targets["stress"] = npf(graph.system_targets["stress"]).reshape(-1, 6)
        # fp64 reference subtraction on the HOST, before device_put downcasts to
        # fp32. `energy` (and `reference_coefficients`) must still be fp64 here --
        # build the dataset with dtype=float64 so the label survives the torch
        # graph-construction `.to(dtype)` cast. The small interaction (~eV) casts
        # to fp32 losslessly; doing `raw(~1e5) - ref(~1e5)` in fp32 instead loses
        # ~meV to catastrophic cancellation (see conservative_regressor._total_loss).
        # `reference_coefficients` is upcast to fp64 to mirror torch `reference.double()`.
        if reference_coefficients is not None:
            # fp64 upcast forces the per-graph accumulation + subtraction below to run
            # in fp64 (NOT to recover the coefficients' own precision -- they are fp32
            # in the model, same as torch). bincount sums the per-atom coefficients into
            # their graph bins (fp64 weights -> fp64 output), the scatter-add we need
            # since many atoms share a graph index.
            ref_coeffs = np.asarray(reference_coefficients, dtype=np.float64)
            atomic_numbers = npf(graph.node_features["atomic_numbers"]).astype(np.int64)
            pgi = npf(graph.node_batch_index).astype(np.int64)
            n_real = graph.n_node.shape[0]
            ref_per_graph = np.bincount(
                pgi, weights=ref_coeffs[atomic_numbers], minlength=n_real
            )
            energy_f64 = npf(graph.system_targets["energy"]).reshape(-1).astype(np.float64)
            targets["interaction_energy"] = (energy_f64 - ref_per_graph).astype(np.float32)

    n = graph.node_batch_index.shape[0]
    e = graph.senders.shape[0]
    g = graph.n_node.shape[0]
    if n_pad < n or e_pad < e or g_pad < g + 1:
        raise ValueError(
            f"bucket ({n_pad},{e_pad},{g_pad}) too small for packed batch "
            f"({n},{e},{g}); need n_pad>=n, e_pad>=e, g_pad>=n_graphs+1."
        )

    # Connectivity: padding atoms/edges point at the absorbing graph (slot g);
    # padding edges are self-loops on the first padding atom (index n).
    node_idx = _np_pad_axis0(npf(graph.node_batch_index), n_pad, fill=g)
    edge_idx = _np_pad_axis0(npf(graph._get_per_edge_graph_indices()), e_pad, fill=g)
    senders = _np_pad_axis0(npf(graph.senders), e_pad, fill=n)
    receivers = _np_pad_axis0(npf(graph.receivers), e_pad, fill=n)

    n_node = _np_pad_axis0(npf(graph.n_node), g_pad)
    n_node[g] = n_pad - n
    n_edge = _np_pad_axis0(npf(graph.n_edge), g_pad)
    n_edge[g] = e_pad - e

    node_features = {k: _np_pad_axis0(npf(v), n_pad) for k, v in graph.node_features.items()}
    edge_features = {}
    for k, v in graph.edge_features.items():
        fill = 1 if k == "unit_shifts" else 0  # nonzero shift -> nonzero vectors
        a = _np_pad_axis0(npf(v), e_pad, fill=fill)
        if k == "unit_shifts":
            a[e:, 1:] = 0  # exactly [1,0,0] per padding edge
        edge_features[k] = a
    system_features = {}
    for k, v in graph.system_features.items():
        a = npf(v)
        if k == "cell":  # identity cell on padding graphs -> finite unit volume
            eye = np.broadcast_to(np.eye(3, dtype=a.dtype), (g_pad - g, 3, 3))
            system_features[k] = np.concatenate([a, eye], axis=0)
        else:
            system_features[k] = _np_pad_axis0(a, g_pad)

    padded_graph = JaxAtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        node_features=node_features,
        edge_features=edge_features,
        system_features=system_features,
        per_node_graph_index=node_idx,
        per_edge_graph_index=edge_idx,
        radius=graph.radius,
        n_real_graph=np.int32(g),
    )
    padded_targets = {
        k: _np_pad_axis0(v, n_pad if k == "forces" else g_pad)
        for k, v in targets.items()
    }
    return padded_graph, padded_targets


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
