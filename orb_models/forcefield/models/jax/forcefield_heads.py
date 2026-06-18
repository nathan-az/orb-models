"""JAX forcefield heads. Port of orb_models.forcefield.models.forcefield_heads.

Only the pieces on the energy -> forces/stress critical path are implemented:
EnergyHead (+ its ScalarNormalizer and LinearReferenceEnergy). ConfidenceHead,
stress/charge heads etc. are deferred -- they do not feed the conservative
force/stress path.

Buffer note: `ScalarNormalizer` (mean/std/count) and `LinearReferenceEnergy`
(coefficients) hold STATISTICS, not gradient-trained params. The normalizer mean/std
track the target distribution via an online running average (torch BatchNorm
momentum=None), updated OUTSIDE backprop -- never by the optimiser. They are float
arrays, so `eqx.filter_grad`'s default `is_inexact_array` filter would still compute
gradients for them; exclude them with `conservative_regressor.trainable_filter`
before the optimiser step. They are constant w.r.t. positions, so they never affect
forces/stress.
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
        # clamp >=1 so empty padding graphs (n_node=0) give 0/1=0, not 0/0=NaN.
        # The differentiated energy is sum_g E_g, so a NaN here -- even on a graph
        # later masked out of the loss -- would poison every gradient. Identity for
        # real graphs (n_node>=1), so parity is untouched.
        return summed / jnp.maximum(n_node, 1)[:, None]
    raise ValueError(f"unsupported reduction: {reduction}")


class ScalarNormalizer(eqx.Module):
    """Affine (de)normalizer with an online running mean/std. Port of torch
    ScalarNormalizer (a BatchNorm1d(momentum=None) used only as a stat accumulator).

    forward:  (x - mean) / std            (normalize a target)
    inverse:  x * std + mean              (denormalize a prediction)

    `update(x)` advances the running stats by one batch using BatchNorm's
    momentum=None *cumulative* average (every batch weighted equally, count-driven),
    matching torch exactly: running_var tracks the UNBIASED batch variance, and the
    update is a no-op for <2 samples. It returns a NEW normalizer (no in-place state)
    -- thread the result through your train step.

    `inference` is the JAX analogue of torch `.eval()`/`online=False`: when True,
    `update` is a no-op so the stats freeze. Toggle a whole model's flags with
    `eqx.nn.inference_mode(model)` (and `value=False` to go back to training).
    Predictions are always normalized with the *current* stats, so flipping the flag
    only stops the stats from moving -- it never changes the affine itself.
    """

    mean: jax.Array  # (1,)
    std: jax.Array  # (1,)
    count: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(())
    )  # batches tracked
    inference: bool = (
        False  # eval mode: freeze the running stats (no online update)
    )

    def __call__(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / self.std

    def inverse(self, x: jax.Array) -> jax.Array:
        return x * self.std + self.mean

    def update(
        self, x: jax.Array, mask: jax.Array | None = None
    ) -> "ScalarNormalizer":
        """Online cumulative-average update from a TARGET tensor. Returns a new
        normalizer; no-op in inference mode or for <2 flattened samples.

        `mask` (broadcastable to `x`) restricts the batch mean/var to the real rows
        when `x` carries padding, so padded zeros never enter the running stats.
        `None` -> the plain dense statistic (parity with the unpadded path).
        """
        flat = x.reshape(-1)
        if self.inference or flat.shape[0] <= 1:
            return self
        if mask is None:
            batch_mean = flat.mean()
            batch_var = flat.var(ddof=1)
        else:
            m = jnp.broadcast_to(mask, x.shape).reshape(-1).astype(flat.dtype)
            n = jnp.maximum(m.sum(), 2.0)  # guard the ddof=1 / n-1 denominators
            batch_mean = (flat * m).sum() / n
            batch_var = (((flat - batch_mean) ** 2) * m).sum() / (n - 1.0)
        count = self.count + 1.0
        exponential_average_factor = (
            1.0 / count
        )  # momentum=None -> 1/num_batches_tracked (equal weight)
        new_mean = (
            1.0 - exponential_average_factor
        ) * self.mean + exponential_average_factor * batch_mean
        new_var = (
            1.0 - exponential_average_factor
        ) * self.std**2 + exponential_average_factor * batch_var
        return ScalarNormalizer(
            mean=new_mean,
            std=jnp.sqrt(new_var),
            count=count,
            inference=self.inference,
        )


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
        return jax.ops.segment_sum(
            per_atom, per_node_graph_index, n_graphs
        )  # (G,)


class EnergyHead(eqx.Module):
    """Per-graph interaction energy in physical units. Port of torch EnergyHead."""

    mlp: MLP
    normalizer: ScalarNormalizer
    reference: LinearReferenceEnergy
    atom_avg: bool = eqx.field(static=True)
    loss_type: str = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        *,
        key,
        predict_atom_avg: bool = True,
        activation: str = "silu",
        loss_type: str = "huber_0.01",
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
        self.loss_type = loss_type

    def __call__(self, node_features: jax.Array, graph) -> jax.Array:
        """Interaction energy (G,) -- the quantity forces/stress differentiate."""
        reduction = "mean" if self.atom_avg else "sum"
        aggregated = aggregate_nodes(
            node_features,
            graph.per_node_graph_index,
            graph.n_node,
            reduction=reduction,
        )
        mlp_out = self.mlp(aggregated).squeeze(-1)  # (G,)
        energy = self.normalizer.inverse(mlp_out)
        if self.atom_avg:
            energy = energy * graph.n_node
        return energy

    def normalize_for_loss(self, x: jax.Array, graph) -> jax.Array:
        """torch EnergyHead._normalize: per-atom-average (if atom_avg) then normalize.

        Applied to BOTH the interaction-energy prediction and the
        reference-subtracted target before the energy loss.
        """
        if self.atom_avg:
            # clamp >=1: empty padding graphs (n_node=0) are masked out of the loss,
            # but x/0 -> inf would survive the mask as 0*inf=NaN. Identity for real
            # graphs (parity preserved).
            x = x / jnp.maximum(graph.n_node, 1)
        return self.normalizer(x)

    def absolute_energy(
        self, interaction_energy: jax.Array, graph, *, fp64: bool = True
    ) -> jax.Array:
        """interaction energy + fixed reference. Constant w.r.t. positions/params.

        When reference energies are OMol-scale (~1e4-1e5 eV), the fp32 step size at
        that magnitude (~0.01-0.04 eV) destroys kJ/mol resolution, so `fp64=True`
        (the default, matching torch `EnergyHead.absolute_energy`) upcasts the final
        sum. NOTE: this is a no-op unless `jax_enable_x64` is set -- without it JAX
        silently keeps fp32, which is exactly the opt-out. `fp64=False` reproduces the
        old single-precision behaviour regardless of the x64 config.
        """
        n_graphs = graph.n_node.shape[0]
        ref = self.reference(
            graph.node_features["atomic_numbers"],
            graph.per_node_graph_index,
            n_graphs,
        )
        if fp64:
            return interaction_energy.astype(jnp.float64) + ref.astype(jnp.float64)
        return interaction_energy + ref.astype(interaction_energy.dtype)
