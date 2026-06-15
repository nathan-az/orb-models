"""Equivalence of the jax loss terms to torch (condhuber forces, huber stress).

float64, exercised against the real torch loss functions. Condhuber is checked
across all four target-magnitude bands; the normalized-loss paths use non-trivial
ScalarNormalizer stats so the (x-mean)/std affine is actually exercised.
"""

import numpy as np
import torch

from orb_models.common.models.nn_util import ScalarNormalizer as TorchScalarNormalizer
from orb_models.forcefield.models.jax.forcefield_heads import ScalarNormalizer
from orb_models.forcefield.models.jax.loss import (
    conditional_huber_force_loss,
    forces_loss,
    stress_loss,
)
from orb_models.forcefield.models.loss import (
    _conditional_huber_force_loss,
    forces_loss_function,
    stress_loss_function,
)

import jax.numpy as jnp


def _matched_normalizer(mean, std):
    torch_norm = TorchScalarNormalizer(init_mean=mean, init_std=std, online=False).eval()
    jax_norm = ScalarNormalizer(mean=jnp.asarray([mean]), std=jnp.asarray([std]))
    return jax_norm, torch_norm


def test_conditional_huber_spans_all_bands(helpers):
    rng = np.random.default_rng(0)
    norms = np.array([10.0, 50.0, 150.0, 250.0, 350.0, 500.0])  # one per band edge
    dirs = rng.standard_normal((6, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    target = dirs * norms[:, None]
    pred = target + rng.standard_normal((6, 3)) * 0.05

    jax_loss = conditional_huber_force_loss(jnp.asarray(pred), jnp.asarray(target), 0.01)
    torch_loss = _conditional_huber_force_loss(
        torch.tensor(pred), torch.tensor(target), 0.01
    )
    helpers.assert_close(jnp.asarray(jax_loss), torch_loss)


def test_forces_loss_matches_torch(helpers):
    rng = np.random.default_rng(1)
    n_node = np.array([3, 4], dtype=np.int64)
    raw_pred = rng.standard_normal((7, 3))
    raw_target = rng.standard_normal((7, 3))
    jax_norm, torch_norm = _matched_normalizer(0.3, 2.0)

    jax_loss = forces_loss(jnp.asarray(raw_pred), jnp.asarray(raw_target), jax_norm)
    torch_out = forces_loss_function(
        raw_pred=torch.tensor(raw_pred),
        raw_target=torch.tensor(raw_target),
        raw_gold_target=torch.tensor(raw_target),
        name="forces",
        normalizer=torch_norm,
        n_node=torch.tensor(n_node),
        fix_atoms=None,
        loss_type="condhuber_0.01",
        training=False,
    )
    helpers.assert_close(jnp.asarray(jax_loss), torch_out.loss)


def test_stress_loss_matches_torch(helpers):
    rng = np.random.default_rng(2)
    raw_pred = rng.standard_normal((2, 6))
    raw_target = rng.standard_normal((2, 6))
    jax_norm, torch_norm = _matched_normalizer(-0.1, 1.5)

    jax_loss = stress_loss(jnp.asarray(raw_pred), jnp.asarray(raw_target), jax_norm)
    torch_out = stress_loss_function(
        raw_pred=torch.tensor(raw_pred),
        raw_target=torch.tensor(raw_target),
        raw_gold_target=torch.tensor(raw_target),
        name="stress",
        normalizer=torch_norm,
        loss_type="huber_0.01",
    )
    helpers.assert_close(jnp.asarray(jax_loss), torch_out.loss)
