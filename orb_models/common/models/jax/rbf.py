import equinox as eqx

import jax
import jax.numpy as jnp


# "trainable" shouldn't really matter with jax, keeping for API consistency
class BesselBasis(eqx.Module):
    num_bases: int = eqx.field(static=True)
    r_max: float = eqx.field(static=True)
    bessel_weights: jax.Array
    prefactor: jax.Array

    def __init__(
        self, r_max: float, num_bases=8, trainable=False, *, key: jax.Array | None = None
    ):
        self.num_bases = num_bases
        self.r_max = r_max
        self.bessel_weights = jnp.pi / r_max * jnp.linspace(1.0, num_bases, num_bases)
        self.prefactor = jnp.sqrt(2.0 / r_max)

    def __call__(self, x: jax.Array) -> jax.Array:
        numerator = jnp.sin(self.bessel_weights * x[:, None])
        return self.prefactor * (numerator / x[:, None])

