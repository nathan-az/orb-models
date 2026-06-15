"""JAX forcefield heads. Port of orb_models.forcefield.models.forcefield_heads.

Only the pieces on the energy -> forces/stress critical path are implemented:
EnergyHead (+ its ScalarNormalizer and LinearReferenceEnergy). ConfidenceHead,
stress/charge heads etc. are deferred -- they do not feed the conservative
force/stress path.

Buffer note: `ScalarNormalizer` (mean/std) and `LinearReferenceEnergy`
(coefficients) hold FIXED statistics, not trainable params. They are float
arrays, so `eqx.filter_grad`'s default `is_inexact_array` filter would compute
gradients for them -- partition them out before the optimiser step once training
for real. They are constant w.r.t. positions, so they never affect forces/stress.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from orb_models.common.models.jax.nn_utils import MLP


def aggregate_nodes(
    node_features: jax.Array,  # (N, F)
    per_node_graph_index: jax.Array,  # (N,) graph id per node
    n_node: jax.Array,  # (G,) atoms per graph
    *,
    reduction: str,
) -> jax.Array:
    """Per-graph node aggregation. Mirrors torch segment_ops.aggregate_nodes."""
    n_graphs = n_node.shape[0]
    summed = jax.ops.segment_sum(node_features, per_node_graph_index, n_graphs)
    if reduction == "sum":
        return summed
    if reduction == "mean":
        return summed / n_node[:, None]
    raise ValueError(f"unsupported reduction: {reduction}")


class ScalarNormalizer(eqx.Module):
    """Fixed affine (de)normalizer. torch ScalarNormalizer in eval mode.

    forward:  (x - mean) / std            (normalize a target)
    inverse:  x * std + mean              (denormalize a prediction)
    """

    mean: jax.Array  # (1,)
    std: jax.Array  # (1,)

    def __call__(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / self.std

    def inverse(self, x: jax.Array) -> jax.Array:
        return x * self.std + self.mean


class LinearReferenceEnergy(eqx.Module):
    """Fixed per-element reference energy, summed over a graph's atoms.

    torch stores a Linear(118, 1, bias=False) applied to a one-hot sum; the
    equivalent here is `sum_{atoms} coefficients[Z_atom]` per graph.
    """

    coefficients: jax.Array  # (118,)

    def __call__(
        self,
        atomic_numbers: jax.Array,  # (N,)
        per_node_graph_index: jax.Array,  # (N,)
        n_graphs: int,
    ) -> jax.Array:
        per_atom = self.coefficients[atomic_numbers]  # (N,)
        return jax.ops.segment_sum(per_atom, per_node_graph_index, n_graphs)  # (G,)


class EnergyHead(eqx.Module):
    """Per-graph interaction energy in physical units. Port of torch EnergyHead."""

    mlp: MLP
    normalizer: ScalarNormalizer
    reference: LinearReferenceEnergy
    atom_avg: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        *,
        key,
        predict_atom_avg: bool = True,
        activation: str = "silu",
    ):
        self.mlp = MLP(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=1,
            activation=activation,
            key=key,
        )
        # Placeholders; real values are copied in from the torch ref (tests) or a
        # checkpoint. mean/std identity-ish, zero reference.
        self.normalizer = ScalarNormalizer(mean=jnp.zeros(1), std=jnp.ones(1))
        self.reference = LinearReferenceEnergy(coefficients=jnp.zeros(118))
        self.atom_avg = predict_atom_avg

    def __call__(self, node_features: jax.Array, graph) -> jax.Array:
        """Interaction energy (G,) -- the quantity forces/stress differentiate."""
        reduction = "mean" if self.atom_avg else "sum"
        aggregated = aggregate_nodes(
            node_features, graph.per_node_graph_index, graph.n_node, reduction=reduction
        )
        mlp_out = self.mlp(aggregated).squeeze(-1)  # (G,)
        energy = self.normalizer.inverse(mlp_out)
        if self.atom_avg:
            energy = energy * graph.n_node
        return energy

    def absolute_energy(self, interaction_energy: jax.Array, graph) -> jax.Array:
        """interaction energy + fixed reference. Constant w.r.t. positions/params."""
        n_graphs = graph.n_node.shape[0]
        ref = self.reference(
            graph.node_features["atomic_numbers"], graph.per_node_graph_index, n_graphs
        )
        return interaction_energy + ref
