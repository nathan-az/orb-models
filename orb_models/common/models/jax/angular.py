from functools import partial

import equinox as eqx

import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _stable_normalize(x: jax.Array, axis: int = 1) -> jax.Array:
    """Normalize ``x`` to unit length with a stable gradient near zero.

    Module-level (not a bound method) because ``jax.custom_jvp`` is not a
    descriptor: decorating a method leaves ``self`` unbound and the call fails to
    resolve its arguments. ``axis`` is a non-differentiable static arg.

    The "double where" (``safe``/``is_zero``) keeps *both* the value and every
    order of derivative finite at ``x == 0`` -- feeding a non-zero into the sqrt
    avoids the NaN that ``maximum(norm, eps)`` clamping alone leaves in the
    second derivative.
    """
    sq = jnp.sum(x * x, axis=axis, keepdims=True)
    is_zero = sq == 0.0
    safe = jnp.where(is_zero, 1.0, x)
    norm = jnp.linalg.vector_norm(safe, axis=axis, keepdims=True)
    return jnp.where(is_zero, 0.0, x / norm)


@_stable_normalize.defjvp
def _stable_normalize_jvp(axis, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    sq = jnp.sum(x * x, axis=axis, keepdims=True)
    is_zero = sq == 0.0
    safe = jnp.where(is_zero, 1.0, x)
    norm = jnp.linalg.vector_norm(safe, axis=axis, keepdims=True)
    unit = jnp.where(is_zero, 0.0, x / norm)
    # d(x/|x|) = x_dot/|x| - unit * <x_dot, unit> / |x|  (projection off the radial dir)
    x_dot_out = jnp.where(
        is_zero,
        0.0,
        x_dot / norm
        - unit * jnp.sum(x_dot * unit, axis=axis, keepdims=True) / norm,
    )
    return unit, x_dot_out


class StableNormalize(eqx.Module):
    """Custom implementation of UnitVector for a more stable backward pass at
    near-zero division. Differentiable to all orders (forward and reverse)."""

    axis: int = eqx.field(static=True, default=1)
    dim: int = eqx.field(static=True, default=3)

    def __call__(self, x: jax.Array) -> jax.Array:
        return _stable_normalize(x, self.axis)


# as above but without correction for near zero div - baseline impl, useful for
# validating StableNormalize against on non-degenerate inputs.
class UnitVector(eqx.Module):
    dim: int = eqx.field(static=True)

    def __init__(self):
        self.dim = 3

    def __call__(self, x: jax.Array) -> jax.Array:
        norm = jnp.linalg.vector_norm(x, axis=1, keepdims=True)
        # this gives massive grads at near zero div, could be fixed with a custom jvp
        safe_norm = jnp.maximum(norm, 1e-12)  # 1e-12 for pytorch parity
        return x / safe_norm
