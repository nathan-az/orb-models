"""JAX ZBL pair repulsion. Energy-only port of pair_repulsion.ZBLBasis.

orb-v3 constructs this with compute_gradients=False: forces/stress come from
autograd of the total energy, so only the ENERGY term is needed here. Adding it
inside `energy_fn` means it flows into forces/stress through the same jax.grad as
the network energy -- no analytic force/stress code required.

Buffers: every array below (screening coefficients c/d, covalent radii, a_exp,
a_prefactor) is a FIXED physical constant, not a trainable param. Equinox has no
`requires_grad=False`, so we defend in two layers:
  1. `jax.lax.stop_gradient` at use -> zero gradient on the grad path (here).
  2. partition them out of the optimised params at the train step (caller's job;
     the real equivalent of "not in the optimizer"). With plain Adam (what orb
     uses) layer 1 alone suffices -- zero grad => zero update. With AdamW/weight
     decay, layer 2 is REQUIRED: decay shrinks params regardless of gradient.
`p` and `node_aggregation` are static (hashable) so they never enter grads at all.
"""

from __future__ import annotations

import ase.data
import equinox as eqx
import jax
import jax.numpy as jnp

from orb_models.common.models.jax.nn_utils import polynomial_cutoff


class ZBLBasis(eqx.Module):
    c: jax.Array  # (4, 1) screening prefactors
    d: jax.Array  # (4, 1) screening exponents
    covalent_radii: jax.Array  # (119,) indexed by physical Z
    a_exp: jax.Array  # scalar
    a_prefactor: jax.Array  # scalar
    p: int = eqx.field(static=True)
    node_aggregation: str = eqx.field(static=True)

    def __init__(self, p: int = 6, node_aggregation: str = "sum"):
        self.c = jnp.array([0.1818, 0.5099, 0.2802, 0.02817])[:, None]
        self.d = jnp.array([3.2, 0.9423, 0.4028, 0.2016])[:, None]
        self.covalent_radii = jnp.asarray(ase.data.covalent_radii)
        self.a_exp = jnp.asarray(0.300)
        self.a_prefactor = jnp.asarray(0.4543)
        self.p = p
        self.node_aggregation = node_aggregation

    def __call__(self, graph) -> jax.Array:
        """Per-graph ZBL repulsion energy (G,). Differentiable w.r.t. edge vectors."""
        # Freeze buffers on the grad path (see module docstring).
        c, d, cov = (jax.lax.stop_gradient(t) for t in (self.c, self.d, self.covalent_radii))
        a_exp = jax.lax.stop_gradient(self.a_exp)
        a_prefactor = jax.lax.stop_gradient(self.a_prefactor)

        senders, receivers = graph.senders, graph.receivers
        Z = graph.node_features["atomic_numbers"]  # physical Z (1..118)
        Z_u = Z[senders].astype(c.dtype)
        Z_v = Z[receivers].astype(c.dtype)

        # Screening distance.
        a = a_prefactor * 0.529 / (jnp.power(Z_u, a_exp) + jnp.power(Z_v, a_exp))

        vectors = graph.edge_features["vectors"]  # differentiable -> forces/stress
        x = jnp.linalg.norm(vectors, axis=1)  # (E,)
        r_over_a = x / a

        exp_term = jnp.exp(-d * r_over_a[None, :])  # (4, E)
        phi = jnp.sum(c * exp_term, axis=0)  # (E,)

        coulomb_term = 14.3996 * Z_u * Z_v / x
        v_edges_raw = coulomb_term * phi

        r_max = cov[Z[senders]] + cov[Z[receivers]]  # (E,)
        envelope = polynomial_cutoff(x, r_max, p=self.p)
        v_edges = 0.5 * v_edges_raw * envelope  # (E,)

        n_nodes = Z.shape[0]
        v_nodes = jax.ops.segment_sum(v_edges, senders, n_nodes)  # (N,)

        n_graphs = graph.n_node.shape[0]
        energy = jax.ops.segment_sum(v_nodes, graph.per_node_graph_index, n_graphs)  # (G,)
        if self.node_aggregation == "mean":
            energy = energy / graph.n_node
        return energy
