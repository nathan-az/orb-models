"""jax UnitVector / StableNormalize: parity with torch + gradient stability.

StableNormalize is the custom-jvp normalization on the autodiff hot path (edge
unit vectors -> forces -> force-loss grad), so these tests cover not just the
forward value but first- and second-order differentiation, including the
near-zero regime where a naive normalization blows up.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models.angular import SphericalHarmonics as TorchSphericalHarmonics
from orb_models.common.models.angular import UnitVector as TorchUnitVector
from orb_models.common.models.jax.angular import (
    SphericalHarmonics,
    StableNormalize,
    UnitVector,
)


@pytest.fixture
def vectors():
    # standard-normal rows have norm ~sqrt(3) >> eps: the non-degenerate regime.
    return np.random.default_rng(0).standard_normal((10, 3))


@pytest.mark.equivalence
def test_unit_vector_matches_torch(helpers, vectors):
    jax_out = UnitVector()(jnp.asarray(vectors))
    torch_out = TorchUnitVector()(torch.tensor(vectors))
    helpers.assert_close(jax_out, torch_out)


@pytest.mark.equivalence
def test_stable_normalize_matches_torch(helpers, vectors):
    """Away from zero, stabilisation is a no-op: equals plain normalization."""
    jax_out = StableNormalize()(jnp.asarray(vectors))
    torch_out = TorchUnitVector()(torch.tensor(vectors))
    helpers.assert_close(jax_out, torch_out)


@pytest.mark.equivalence
def test_stable_normalize_grad_matches_torch(helpers, vectors):
    """First derivative agrees with torch autograd on non-degenerate input."""
    jax_grad = jax.grad(lambda x: jnp.sum(jnp.sin(StableNormalize()(x))))(
        jnp.asarray(vectors)
    )
    tx = torch.tensor(vectors, requires_grad=True)
    torch.sin(TorchUnitVector()(tx)).sum().backward()
    helpers.assert_close(jax_grad, tx.grad)


def test_stable_normalize_dim_attr():
    """dim is exposed (gns.py reads angular_transform.dim)."""
    assert StableNormalize().dim == 3


@pytest.mark.parametrize("vec", [[0.0, 0.0, 0.0], [1e-20, 0.0, 0.0]])
def test_stable_normalize_finite_grad_near_zero(vec):
    """The whole point of the custom jvp: finite gradient at/near zero norm."""
    x = jnp.array([vec])
    grad = jax.grad(lambda z: StableNormalize()(z).sum())(x)
    assert np.isfinite(np.asarray(grad)).all()


def test_stable_normalize_finite_hessian_at_zero():
    """The double-where keeps even the second derivative finite at exact zero."""
    x = jnp.zeros((1, 3))
    hess = jax.hessian(lambda z: StableNormalize()(z).sum())(x)
    assert np.isfinite(np.asarray(hess)).all()


@pytest.mark.equivalence
@pytest.mark.parametrize("lmax", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
def test_spherical_harmonics_matches_torch(helpers, vectors, lmax, normalization):
    """jax SH equals torch SH for every ported lmax and normalization scheme.

    normalize=True (the orb-v3 setting) projects onto the unit sphere first.
    """
    jx = SphericalHarmonics(lmax, normalize=True, normalization=normalization)
    tx = TorchSphericalHarmonics(lmax, normalize=True, normalization=normalization)
    assert jx.dim == tx.dim == (lmax + 1) ** 2
    jax_out = jx(jnp.asarray(vectors))
    torch_out = tx(torch.tensor(vectors))
    helpers.assert_close(jax_out, torch_out)


@pytest.mark.equivalence
def test_spherical_harmonics_unnormalized_matches_torch(helpers, vectors):
    """normalize=False: SH of the raw (non-unit) vectors must also match torch."""
    jx = SphericalHarmonics(3, normalize=False, normalization="component")
    tx = TorchSphericalHarmonics(3, normalize=False, normalization="component")
    helpers.assert_close(jx(jnp.asarray(vectors)), tx(torch.tensor(vectors)))


@pytest.mark.equivalence
def test_spherical_harmonics_grad_matches_torch(helpers, vectors):
    """First derivative agrees with torch autograd (SH is on the force path)."""
    jx = SphericalHarmonics(3, normalize=True, normalization="component")
    tx = TorchSphericalHarmonics(3, normalize=True, normalization="component")
    jax_grad = jax.grad(lambda x: jnp.sum(jnp.sin(jx(x))))(jnp.asarray(vectors))
    t = torch.tensor(vectors, requires_grad=True)
    torch.sin(tx(t)).sum().backward()
    helpers.assert_close(jax_grad, t.grad)


def test_spherical_harmonics_lmax_guard():
    """lmax beyond the ported table raises rather than silently misbehaving."""
    with pytest.raises(NotImplementedError):
        SphericalHarmonics(5, normalize=True, normalization="component")


def test_stable_normalize_second_order_consistency(vectors):
    """forward-over-reverse (jvp of grad) must equal reverse-over-reverse (hessian).

    This is exactly the equivalence the force-loss training step relies on when
    swapping the naive double-backward for the jvp-of-grad force path.
    """
    g = lambda x: jnp.sum(jnp.sin(StableNormalize()(x)))
    x = jnp.asarray(vectors)
    v = jnp.ones_like(x)

    hess_v = jnp.tensordot(jax.hessian(g)(x), v, axes=([2, 3], [0, 1]))
    _, jvp_of_grad = jax.jvp(jax.grad(g), (x,), (v,))
    assert np.allclose(np.asarray(hess_v), np.asarray(jvp_of_grad), atol=1e-10)
