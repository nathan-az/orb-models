import math
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


# Max l implemented below. orb-v3 uses lmax=3; we port 0..4 for headroom. The
# recurrence beyond this is a long auto-generated table (e3nn supports up to 11);
# add more bands here only if a checkpoint needs them.
_LMAX_IMPLEMENTED = 4


def _spherical_harmonics(
    lmax: int, x: jax.Array, y: jax.Array, z: jax.Array
) -> jax.Array:
    """Real spherical harmonics on the unit sphere, e3nn ordering, stacked on the
    last axis. Direct port of torch `angular._spherical_harmonics` (lmax 0..4).

    Returns (..., (lmax+1)**2), each l-block of width 2l+1 contiguous and in e3nn
    order -- so torch's `cat([sh[l*l:(l+1)**2] ...])` reslice is the identity and
    is omitted here.
    """
    sh_0_0 = jnp.ones_like(x)
    if lmax == 0:
        return jnp.stack([sh_0_0], axis=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return jnp.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], axis=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y**2
    x2z2 = x**2 + z**2
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z**2 - x**2)
    if lmax == 2:
        return jnp.stack(
            [sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
            axis=-1,
        )

    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return jnp.stack(
            [
                sh_0_0, sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            ],
            axis=-1,
        )

    sh_4_0 = 0.935414346693485 * sh_3_0 * z + 0.935414346693485 * sh_3_6 * x
    sh_4_1 = (
        0.661437827766148 * sh_3_0 * y
        + 0.810092587300982 * sh_3_1 * z
        + 0.810092587300983 * sh_3_5 * x
    )
    sh_4_2 = (
        -0.176776695296637 * sh_3_0 * z
        + 0.866025403784439 * sh_3_1 * y
        + 0.684653196881458 * sh_3_2 * z
        + 0.684653196881457 * sh_3_4 * x
        + 0.176776695296637 * sh_3_6 * x
    )
    sh_4_3 = (
        -0.306186217847897 * sh_3_1 * z
        + 0.968245836551855 * sh_3_2 * y
        + 0.790569415042095 * sh_3_3 * x
        + 0.306186217847897 * sh_3_5 * x
    )
    sh_4_4 = -0.612372435695795 * sh_3_2 * x + sh_3_3 * y - 0.612372435695795 * sh_3_4 * z
    sh_4_5 = (
        -0.306186217847897 * sh_3_1 * x
        + 0.790569415042096 * sh_3_3 * z
        + 0.968245836551854 * sh_3_4 * y
        - 0.306186217847897 * sh_3_5 * z
    )
    sh_4_6 = (
        -0.176776695296637 * sh_3_0 * x
        - 0.684653196881457 * sh_3_2 * x
        + 0.684653196881457 * sh_3_4 * z
        + 0.866025403784439 * sh_3_5 * y
        - 0.176776695296637 * sh_3_6 * z
    )
    sh_4_7 = (
        -0.810092587300982 * sh_3_1 * x
        + 0.810092587300982 * sh_3_5 * z
        + 0.661437827766148 * sh_3_6 * y
    )
    sh_4_8 = -0.935414346693485 * sh_3_0 * x + 0.935414346693486 * sh_3_6 * z
    if lmax == 4:
        return jnp.stack(
            [
                sh_0_0, sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            ],
            axis=-1,
        )

    raise NotImplementedError(  # pragma: no cover - guarded in __init__
        f"_spherical_harmonics ported up to lmax={_LMAX_IMPLEMENTED}, got {lmax}"
    )


def _normalization_factors(lmax: int, normalization: str) -> jax.Array:
    """Per-component multiplier (shape ((lmax+1)**2,)) applied after the raw SH.

    Matches the per-l-block scaling in torch SphericalHarmonics.forward:
      integral  -> sqrt(2l+1) / sqrt(4*pi)
      component -> sqrt(2l+1)
      norm      -> 1 (no scaling)
    """
    blocks = []
    for l in range(lmax + 1):
        if normalization == "integral":
            scale = math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)
        elif normalization == "component":
            scale = math.sqrt(2 * l + 1)
        else:  # "norm"
            scale = 1.0
        blocks.extend([scale] * (2 * l + 1))
    return jnp.asarray(blocks)


class SphericalHarmonics(eqx.Module):
    """Real spherical harmonics, port of torch `angular.SphericalHarmonics` (e3nn).

    No learnable parameters: a fixed geometric edge featurizer. `normalize=True`
    projects each vector onto the unit sphere first, matching torch's internal
    `F.normalize` (``x / max(||x||, 1e-12)`` -- zero vectors map to zero, no NaN).
    The orb-v3 conservative backbone uses ``lmax=3, normalize=True,
    normalization="component"``. `dim` is exposed because gns reads
    `angular_transform.dim` to size the edge embedding.
    """

    _lmax: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, lmax: int, normalize: bool, normalization: str = "integral"):
        assert normalization in ("integral", "component", "norm")
        if lmax > _LMAX_IMPLEMENTED:
            raise NotImplementedError(
                f"jax SphericalHarmonics ported up to lmax={_LMAX_IMPLEMENTED}, got {lmax}"
            )
        self._lmax = lmax
        self.normalize = normalize
        self.normalization = normalization
        self.dim = (lmax + 1) ** 2

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.normalize:
            # torch F.normalize(x, dim=-1): divide by clamped 2-norm (eps=1e-12).
            norm = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
            x = x / jnp.maximum(norm, 1e-12)
        sh = _spherical_harmonics(self._lmax, x[..., 0], x[..., 1], x[..., 2])
        if self.normalization in ("integral", "component"):
            sh = sh * _normalization_factors(self._lmax, self.normalization)
        return sh
