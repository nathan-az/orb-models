"""Compare jax `segment_softmax` to the PyTorch reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models import segment_ops as torch_segment_ops
from orb_models.common.models.jax import segment_ops as jax_segment_ops


@pytest.fixture
def softmax_inputs():
    rng = np.random.default_rng(0)
    num_nodes, num_edges = 6, 14
    seg = rng.integers(0, num_nodes, size=(num_edges,)).astype(np.int64)
    data = rng.standard_normal((num_edges, 1))
    weights = rng.uniform(0.1, 1.0, size=(num_edges, 1))
    return num_nodes, seg, data, weights


@pytest.mark.equivalence
@pytest.mark.parametrize("use_weights", [False, True])
def test_segment_softmax_matches_torch(helpers, softmax_inputs, use_weights):
    num_nodes, seg, data, weights = softmax_inputs
    jw = jnp.asarray(weights) if use_weights else None
    tw = torch.tensor(weights) if use_weights else None

    jax_out = jax_segment_ops.segment_softmax(jnp.asarray(data), jnp.asarray(seg), num_nodes, weights=jw)
    torch_out = torch_segment_ops.segment_softmax(
        torch.tensor(data), torch.tensor(seg), num_nodes, weights=tw
    )
    helpers.assert_close(jax_out, torch_out)


def test_segment_softmax_grad_finite_with_zeroed_segment():
    """A fully zero-weighted segment (all edges beyond the distance cutoff) must
    not leak NaNs back through the `where(denom == 0, ...)` branch under grad."""
    data = jnp.array([[1.0], [2.0], [0.5]])
    seg = jnp.array([0, 0, 1])
    weights = jnp.array([[0.0], [0.0], [1.0]])  # segment 0 fully zeroed

    grad = jax.grad(
        lambda d: jax_segment_ops.segment_softmax(d, seg, 2, weights=weights).sum()
    )(data)
    assert bool(np.isfinite(np.asarray(grad)).all())
