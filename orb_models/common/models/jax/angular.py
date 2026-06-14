import equinox as eqx

import jax
import jax.numpy as jnp


class StableNormalize(eqx.Module):
    """Custom implementation of below for more stable backward pass at near zero div"""
    axis: int = 1

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._fwd(x)

    @jax.custom_jvp
    def _fwd(self, x: jax.Array) -> jax.Array:
        eps = 1e-12
        norm = jnp.linalg.vector_norm(x, axis=self.axis, keepdims=True)
        is_small = norm < eps
        unit = x / jnp.maximum(norm, eps)
        return jnp.where(is_small, 0.0, unit)

    @_fwd.defjvp
    def _fwd_jvp(self, primals, tangents):
        x, = primals
        x_dot, = tangents
        eps = 1e-12
        norm = jnp.linalg.vector_norm(x, axis=self.axis, keepdims=True)
        is_small = norm < eps
        unit = x / jnp.maximum(norm, eps)
        y = jnp.where(is_small, 0.0, unit)
        norm_clamped = jnp.maximum(norm, eps)
        x_dot_out = jnp.where(
            is_small,
            0.0,
            x_dot / norm_clamped
            - unit * jnp.sum(x_dot * unit, axis=self.axis, keepdims=True) / norm_clamped,
        )
        return y, x_dot_out


# as above but without correction for near zero div - baseline impl
# not to be used if above works correctly when tested, can delete
class UnitVector(eqx.Module):
    dim: int = eqx.field(static=True)

    def __init__(self):
        self.dim = 3

    def __call__(self, x: jax.Array) -> jax.Array:
        norm = jnp.linalg.vector_norm(x, axis=1, keepdims=True)
        # this gives massive grads at near zero div, could be fixed with a custom jvp
        safe_norm = jnp.maximum(norm, 1e-12)  # 1e-12 for pytorch parity
        return x / safe_norm
