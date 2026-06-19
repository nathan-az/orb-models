"""JAX CoulombModule -- non-periodic direct Coulomb sum.

Port of the non-periodic (`_direct_coulomb`) branch of
`orb_models.forcefield.models.coulomb_module.CoulombModule`. The periodic
Particle-Mesh-Ewald branch is intentionally NOT ported: it wraps the
nvalchemiops CUDA op behind `@torch.compiler.disable` precisely because it is
un-jittable, and has no JAX analogue.

Design difference from torch
----------------------------
Torch returns `(energy, explicit_forces, explicit_stress)`. For non-periodic
systems the explicit force/stress are all-zero -- the spatial forces come from
autograd through the energy, and the charge-equilibration term dE/dr-through-q(r)
likewise. So this module returns ONLY the per-graph energy ``(G,)``; the
consumer (the conservative regressor) folds it into the differentiated
interaction energy and gets all forces/stress from the outer `jax.grad`.

The torch implementation builds an explicit, data-dependent fully-connected
sender/receiver list. Here we instead form the dense ``(N, N)`` pairwise matrix
and mask it (same-system, non-self, both-real). This is fixed-shape -> jit-
friendly (and pad-safe), and is mathematically the same O(N^2) pair sum.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from orb_models.common.atoms.jax.graph_batch import JaxAtomGraphs, real_node_mask

COULOMB_CONSTANT = 14.3996  # eV*A/e^2


class CoulombModule(eqx.Module):
    """Long-range electrostatic energy from predicted per-atom charges (non-periodic).

        E = k/2 * sum_{i != j} q_i q_j erf(r_ij / (sigma*sqrt2)) / r_ij

    with the erf damping dropped (-> plain 1/r) when
    ``direct_coulomb_erf_damping_sigma`` is None (the released orbmol-v2 default).

    `coulomb_constant` is a fixed physical constant (a torch buffer), not a
    trainable param -- exclude it from the optimiser partition.
    """

    coulomb_constant: jax.Array
    direct_coulomb_erf_damping_sigma: float | None = eqx.field(static=True)

    def __init__(self, direct_coulomb_erf_damping_sigma: float | None = None):
        self.direct_coulomb_erf_damping_sigma = direct_coulomb_erf_damping_sigma
        self.coulomb_constant = jnp.asarray(COULOMB_CONSTANT)

    def __call__(self, latent_charges: jax.Array, graph: JaxAtomGraphs) -> jax.Array:
        """Per-graph electrostatic energy ``(G,)``.

        Args:
            latent_charges: ``(N, 1)`` (or ``(N,)``) predicted per-atom charges.
            graph: the (possibly padded) batch -- reads positions, the per-node
                graph index, and the real-node mask.
        """
        charges = latent_charges.reshape(-1)  # (N,)
        positions = graph.node_features["positions"]  # (N, 3)
        pgi = graph.per_node_graph_index  # (N,)
        n_graphs = graph.n_node.shape[0]
        real = real_node_mask(graph)  # (N,)

        # Pairwise mask: same system, no self-loops, both atoms real (padding atoms
        # -- which coincide at the origin -- are dropped so they contribute neither
        # energy nor a 0/0 force NaN).
        same_system = pgi[:, None] == pgi[None, :]
        not_self = ~jnp.eye(positions.shape[0], dtype=bool)
        pair_mask = same_system & not_self & real[:, None] & real[None, :]

        # Squared distances, but force the masked-out entries to a safe nonzero
        # value BEFORE the sqrt so coincident/self pairs never produce a sqrt(0)
        # NaN that would survive the multiplicative mask as 0 * NaN.
        diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
        sq = jnp.sum(diff * diff, axis=-1)  # (N, N)
        safe_sq = jnp.where(pair_mask, sq, 1.0)
        dist = jnp.sqrt(safe_sq)

        qq = charges[:, None] * charges[None, :]  # (N, N)
        if self.direct_coulomb_erf_damping_sigma is None:
            pair = qq / dist
        else:
            convergence = jsp.erf(
                dist / (self.direct_coulomb_erf_damping_sigma * math.sqrt(2.0))
            )
            pair = qq * convergence / dist

        pair = jnp.where(pair_mask, pair, 0.0)
        per_atom = jnp.sum(pair, axis=1)  # (N,)
        per_graph = jax.ops.segment_sum(per_atom, pgi, n_graphs)  # (G,)
        return 0.5 * self.coulomb_constant * per_graph

    def periodic_energy(
        self,
        latent_charges: jax.Array,  # (N, 1) or (N,)
        positions: jax.Array,  # (N, 3) LIVE differentiable positions
        cell: jax.Array,  # (G, 3, 3) LIVE differentiable cells
        pme_prep,  # (sr_batch, nonperiodic_batch, periodic_batch) from pme.build_pme_batch
    ) -> jax.Array:
        """Per-graph periodic (Ewald/PME) energy ``(G,)`` via the jax-pme engine.

        The host-prepped `pme_prep` carries the fixed real-space neighbour list,
        k-grid, and padding masks (see `jax/pme.py`); the LIVE `positions`/`cell`
        flow through so forces/stress come from the outer `jax.grad`. jax-pme's
        `batched_mixed` handles mixed periodic / non-periodic systems in one batch
        (its non-periodic path is the same bare 1/r sum as `__call__`), so a single
        `pme_prep` covers a whole bucket. Same 0.5*k convention as `__call__`.
        """
        from orb_models.forcefield.models.jax.pme import periodic_coulomb_energy

        sr_batch, nonperiodic_batch, periodic_batch = pme_prep
        return periodic_coulomb_energy(
            latent_charges.reshape(-1),
            positions,
            cell,
            sr_batch,
            nonperiodic_batch,
            periodic_batch,
            prefactor=self.coulomb_constant,
        )
