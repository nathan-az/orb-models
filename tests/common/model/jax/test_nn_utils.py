"""Compare jax nn building blocks (TensorLinear/LayerNorm, MLP, MLPAndLayerNorm)
to their PyTorch references, with identical weights copied across."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

from orb_models.common.models.jax.nn_utils import (
    MLP,
    MLPAndLayerNorm,
    TensorLinear,
    TensorLayerNorm,
    TensorRMSNorm,
)
from orb_models.common.models.nn_util import build_mlp, mlp_and_layer_norm

pytestmark = pytest.mark.equivalence


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# 2D (graph-style [N, d]) and 3D ([N, K, d]) exercise the vmap-over-leading-dims.
@pytest.mark.parametrize("shape", [(10, 4), (6, 5, 4)])
def test_tensor_linear(helpers, key, rng, shape):
    torch_lin = nn.Linear(4, 3)
    jax_lin = helpers.copy_linear(TensorLinear(4, 3, key=key), torch_lin)

    x = rng.standard_normal(shape)
    helpers.assert_close(jax_lin(jnp.asarray(x)), torch_lin(torch.tensor(x)))


@pytest.mark.parametrize("shape", [(10, 4), (6, 5, 4)])
def test_tensor_layer_norm(helpers, rng, shape):
    torch_ln = nn.LayerNorm(4)
    # randomise affine params so weight/bias are covered, not just normalisation
    with torch.no_grad():
        torch_ln.weight.copy_(torch.tensor(rng.standard_normal(4)))
        torch_ln.bias.copy_(torch.tensor(rng.standard_normal(4)))
    jax_ln = helpers.copy_layer_norm(TensorLayerNorm(4), torch_ln)

    x = rng.standard_normal(shape)
    helpers.assert_close(jax_ln(jnp.asarray(x)), torch_ln(torch.tensor(x)))


@pytest.mark.parametrize("shape", [(10, 4), (6, 5, 4)])
def test_tensor_rms_norm(helpers, rng, shape):
    """TensorRMSNorm matches torch nn.RMSNorm (weight only, no bias, finfo eps)."""
    torch_rn = nn.RMSNorm(4)
    with torch.no_grad():
        torch_rn.weight.copy_(torch.tensor(rng.standard_normal(4)))
    jax_rn = helpers.copy_layer_norm(TensorRMSNorm(4), torch_rn)

    x = rng.standard_normal(shape)
    helpers.assert_close(jax_rn(jnp.asarray(x)), torch_rn(torch.tensor(x)))


def test_mlp_and_layer_norm_rms(helpers, key, rng):
    """MLPAndLayerNorm with norm_type='rms_norm' matches torch mlp_and_layer_norm."""
    torch_mln = mlp_and_layer_norm(4, 3, 8, 2, activation="silu", mlp_norm="rms_norm")
    jax_mln = helpers.copy_mlp_and_layer_norm(
        MLPAndLayerNorm(4, 3, 8, 2, activation="silu", norm_type="rms_norm", key=key),
        torch_mln,
    )

    x = rng.standard_normal((10, 4))
    helpers.assert_close(jax_mln(jnp.asarray(x)), torch_mln(torch.tensor(x)))


def test_mlp(helpers, key, rng):
    torch_mlp = build_mlp(4, [8, 8], 3, activation="silu")
    jax_mlp = helpers.copy_mlp(MLP(4, [8, 8], 3, activation="silu", key=key), torch_mlp)

    x = rng.standard_normal((10, 4))
    helpers.assert_close(jax_mlp(jnp.asarray(x)), torch_mlp(torch.tensor(x)))


def test_mlp_and_layer_norm(helpers, key, rng):
    torch_mln = mlp_and_layer_norm(4, 3, 8, 2, activation="silu", mlp_norm="layer_norm")
    jax_mln = helpers.copy_mlp_and_layer_norm(
        MLPAndLayerNorm(4, 3, 8, 2, activation="silu", norm_type="layer_norm", key=key), torch_mln
    )

    x = rng.standard_normal((10, 4))
    helpers.assert_close(jax_mln(jnp.asarray(x)), torch_mln(torch.tensor(x)))
