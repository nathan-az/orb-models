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

    def absolute_energy(self, interaction_energy: jax.Array, graph) -> jax.Array:
        """interaction energy + fixed reference. Constant w.r.t. positions/params.

        Runs in fp32 (the model's only precision). Reference energies at OMol scale
        (~1e4-1e5 eV) lose kJ/mol resolution to the fp32 step size at that magnitude;
        the energy *loss* avoids this by working on the reference-subtracted target
        (see `interaction_reference` in the regressor), so the absolute sum here is
        used for reporting, not the gradient.
        """
        n_graphs = graph.n_node.shape[0]
        ref = self.reference(
            graph.node_features["atomic_numbers"],
            graph.per_node_graph_index,
            n_graphs,
        )
        return interaction_energy + ref.astype(interaction_energy.dtype)


class ChargeConditionedEnergyHead(EnergyHead):
    """Energy head conditioned on per-atom charges (and optionally spins).

    Port of torch ChargeConditionedEnergyHead. Unlike EnergyHead -- which pools
    node features first then applies the MLP -- this applies the MLP PER ATOM
    (with charge/spin appended), denormalizes each atom's contribution, then
    SUM-pools. Sum-pooling preserves size-consistency: for two non-interacting
    subsystems, E(A u B) = E(A) + E(B). The MLP input is widened by 1 (+1 for
    spins). `normalize_for_loss`/`absolute_energy` are inherited unchanged (they
    only touch the normalizer/reference, which act on the per-atom-average).
    """

    use_spins: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        *,
        key,
        use_spins: bool = False,
        predict_atom_avg: bool = True,
        activation: str = "silu",
        loss_type: str = "huber_0.01",
    ):
        assert predict_atom_avg, "ChargeConditionedEnergyHead always uses per-atom energy"
        super().__init__(
            latent_dim=latent_dim + 1 + int(use_spins),
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            key=key,
            predict_atom_avg=True,
            activation=activation,
            loss_type=loss_type,
        )
        self.use_spins = use_spins

    def __call__(  # type: ignore[override]
        self,
        node_features: jax.Array,
        graph,
        per_atom_charges: jax.Array,
        per_atom_spins: jax.Array | None = None,
    ) -> jax.Array:
        """Interaction energy ``(G,)``, conditioned on per-atom charges/spins."""
        features = jnp.concatenate([node_features, per_atom_charges], axis=-1)
        if self.use_spins:
            assert per_atom_spins is not None, "per_atom_spins required when use_spins=True"
            features = jnp.concatenate([features, per_atom_spins], axis=-1)
        per_atom_mlp = self.mlp(features).squeeze(-1)  # (N,)
        per_atom_energy = self.normalizer.inverse(per_atom_mlp)  # (N,)
        return aggregate_nodes(
            per_atom_energy[:, None],
            graph.per_node_graph_index,
            graph.n_node,
            reduction="sum",
        ).squeeze(-1)


# --- OrbMol-v2: per-atom latent charge / spin heads --------------------------
# These predict per-atom scalars that (1) condition the energy head and (2) feed
# the CoulombModule. Both apply a small MLP per atom then optionally enforce a
# per-system linear constraint. The torch `repeat_interleave(n_node)` that
# broadcasts a per-graph quantity onto its atoms is replaced by a GATHER through
# `per_node_graph_index` -- a fixed-shape op, so the head jits under padding (the
# same trick the ChargeSpinConditioner uses).


def _enforce_per_system_sum(
    values: jax.Array,  # (N, 1)
    graph,
    target_total: jax.Array | None,  # (G,) desired per-system sum, or None
) -> jax.Array:
    """Center `values` to zero per-system mean, then (if given) shift so each
    system sums to `target_total`. Mirrors the torch centering/shift, with the
    per-graph -> per-atom broadcast done by gather instead of repeat_interleave.
    """
    pgi = graph.per_node_graph_index
    mean = aggregate_nodes(values, pgi, graph.n_node, reduction="mean")  # (G, 1)
    values = values - mean[pgi]
    if target_total is not None:
        # clamp n_node >= 1 so empty padding graphs (n_node=0) give 0/1=0, not 0/0.
        shift = target_total / jnp.maximum(graph.n_node, 1)  # (G,)
        values = values + shift[pgi][:, None]
    return values


class LatentChargeHead(eqx.Module):
    """Per-atom latent charges from node features. Port of torch LatentChargeHead.

    Charges are centered to zero per-system mean and (when `total_charge` is in
    the system features) shifted so each system sums to its total charge, then
    scaled by `charge_scale`.
    """

    mlp: MLP
    enforce_total_charge: bool = eqx.field(static=True)
    charge_scale: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int = 1,
        mlp_hidden_dim: int = 128,
        *,
        key,
        enforce_total_charge: bool = True,
        activation: str = "silu",
        charge_scale: float = 1.0,
    ):
        self.mlp = MLP(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * (num_mlp_layers - 1),
            output_size=1,
            activation=activation,
            key=key,
        )
        self.enforce_total_charge = enforce_total_charge
        self.charge_scale = charge_scale

    def __call__(self, node_features: jax.Array, graph) -> jax.Array:
        """Predict per-atom charges ``(N, 1)``."""
        charges = self.mlp(node_features)  # (N, 1)
        if self.enforce_total_charge:
            total_charge = graph.system_features.get("total_charge")
            if total_charge is not None:
                total_charge = total_charge.astype(charges.dtype)
            charges = _enforce_per_system_sum(charges, graph, total_charge)
        return charges * self.charge_scale


class LatentSpinHead(eqx.Module):
    """Per-atom latent spins from node features. Port of torch LatentSpinHead.

    Spins are centered to zero per-system mean and (when `spin_multiplicity` is
    in the system features) shifted so each system sums to 2S = multiplicity - 1.
    """

    mlp: MLP
    enforce_spin_constraint: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int = 1,
        mlp_hidden_dim: int = 128,
        *,
        key,
        enforce_spin_constraint: bool = True,
        activation: str = "silu",
    ):
        self.mlp = MLP(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * (num_mlp_layers - 1),
            output_size=1,
            activation=activation,
            key=key,
        )
        self.enforce_spin_constraint = enforce_spin_constraint

    def __call__(self, node_features: jax.Array, graph) -> jax.Array:
        """Predict per-atom spins ``(N, 1)``."""
        spins = self.mlp(node_features)  # (N, 1)
        if self.enforce_spin_constraint:
            multiplicity = graph.system_features.get("spin_multiplicity")
            total_spin = None
            if multiplicity is not None:
                total_spin = multiplicity.astype(spins.dtype) - 1  # 2S
            spins = _enforce_per_system_sum(spins, graph, total_spin)
        return spins
