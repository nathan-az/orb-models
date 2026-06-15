"""JAX port of the orb loss terms (orb_models.forcefield.models.loss + mean_error).

The actual orb conservative model uses:
  * forces: condhuber_0.01  (MACE conditional Huber)
  * stress: huber_0.01
  * energy: huber_0.01      (on reference-subtracted, normalized interaction energy)
Each is computed on *normalized* quantities: target/pred are passed through an affine
`ScalarNormalizer` ((x - mean)/std) before the elementwise loss. These functions read
the normalizer's CURRENT stats and never mutate them -- the online running-stat update
(torch BatchNorm momentum=None) lives in `conservative_regressor.update_normalizer_buffers`,
called once per train step on the targets before the loss (see that function).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def huber_loss(pred: jax.Array, target: jax.Array, delta: float | jax.Array) -> jax.Array:
    """Elementwise Huber. Matches torch.nn.functional.huber_loss(reduction='none').

    `delta` may be a scalar or broadcastable per-row array (used by condhuber).
    """
    err = jnp.abs(pred - target)
    quad = jnp.minimum(err, delta)
    lin = err - quad
    return 0.5 * quad**2 + delta * lin


def mean_error(pred: jax.Array, target: jax.Array, error_type: str) -> jax.Array:
    """mae / mse / huber_<delta>, averaged over the last axis then the batch.

    Mirrors graph_regressor.mean_error for the no-`batch_n_node` case (energy is
    1-D, stress is (G,6) -> mean over 6 -> mean). The nested per-graph aggregation
    is only used by the non-condhuber force path, which orb does not use.
    """
    if error_type.startswith("huber"):
        delta = float(error_type.split("_")[1])
        errors = huber_loss(pred, target, delta)
    elif error_type == "mae":
        errors = jnp.abs(pred - target)
    elif error_type == "mse":
        errors = (pred - target) ** 2
    else:
        raise ValueError(f"unsupported error_type: {error_type}")

    if errors.ndim > 1:
        errors = errors.mean(axis=-1)
    return errors.mean()


def conditional_huber_force_loss(
    pred_forces: jax.Array, target_forces: jax.Array, huber_delta: float
) -> jax.Array:
    """MACE conditional Huber for forces. Per-row delta by target force magnitude.

    bands on ||target||: [0,100) [100,200) [200,300) [300, inf) -> delta * {1,.7,.4,.1}
    """
    factors = jnp.asarray([huber_delta * x for x in (1.0, 0.7, 0.4, 0.1)])
    norm = jnp.linalg.norm(target_forces, axis=-1)  # (M,)
    band = (
        jnp.where(norm < 100, 0, jnp.where(norm < 200, 1, jnp.where(norm < 300, 2, 3)))
    )
    delta = factors[band][:, None]  # (M, 1), broadcast over the 3 components
    return huber_loss(pred_forces, target_forces, delta).mean()


def forces_loss(
    raw_pred: jax.Array,
    raw_target: jax.Array,
    normalizer,
    loss_type: str = "condhuber_0.01",
) -> jax.Array:
    """Normalize pred/target, then condhuber (or mae/mse/huber). fix_atoms=None only."""
    target = normalizer(raw_target)
    pred = normalizer(raw_pred)
    if loss_type.startswith("condhuber"):
        delta = float(loss_type.split("_")[1])
        return conditional_huber_force_loss(pred, target, delta)
    return mean_error(pred, target, loss_type)


def stress_loss(
    raw_pred: jax.Array,
    raw_target: jax.Array,
    normalizer,
    loss_type: str = "huber_0.01",
) -> jax.Array:
    """Normalize pred/target, then mean_error (huber by default).

    Both args must be in the SAME layout (Voigt-6); use `full_3x3_to_voigt_6` on a
    3x3 stress first, since torch computes the loss on Voigt-6 (6 components, not 9).
    """
    return mean_error(normalizer(raw_pred), normalizer(raw_target), loss_type)


def full_3x3_to_voigt_6(stress: jax.Array) -> jax.Array:
    """(..., 3, 3) -> (..., 6) Voigt [s00,s11,s22,s12,s02,s01], shears averaged.

    Matches forcefield_utils.torch_full_3x3_to_voigt_6_stress.
    """
    s = stress
    return jnp.stack(
        [
            s[..., 0, 0],
            s[..., 1, 1],
            s[..., 2, 2],
            (s[..., 1, 2] + s[..., 2, 1]) / 2,
            (s[..., 0, 2] + s[..., 2, 0]) / 2,
            (s[..., 0, 1] + s[..., 1, 0]) / 2,
        ],
        axis=-1,
    )
