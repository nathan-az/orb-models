"""Compare jax AtomEmbedding to the PyTorch reference, plus init/lookup checks."""

import types

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models.embedding import AtomEmbedding as TorchAtomEmbedding
from orb_models.common.models.jax.embedding import AtomEmbedding


def _batch(atomic_numbers):
    """Minimal stand-in for an AtomGraphs/JaxAtomGraphs: both forwards only read
    node_features["atomic_numbers"]."""
    return types.SimpleNamespace(node_features={"atomic_numbers": atomic_numbers})


@pytest.mark.equivalence
def test_atom_embedding_matches_torch(helpers, key):
    emb_size, num_elements = 16, 118
    torch_emb = TorchAtomEmbedding(emb_size, num_elements)
    jax_emb = AtomEmbedding(emb_size, num_elements, key=key)
    # share the lookup table so the gather is what's being compared
    jax_emb = eqx.tree_at(
        lambda m: m.embeddings.weight, jax_emb, helpers.to_jax(torch_emb.embeddings.weight)
    )

    z = np.array([1, 6, 8, 1, 26, 0], dtype=np.int64)
    jax_out = jax_emb(_batch(jnp.asarray(z)))
    torch_out = torch_emb(_batch(torch.tensor(z)))
    helpers.assert_close(jax_out, torch_out)


def test_atom_embedding_casts_float_indices(helpers, key):
    """atomic_numbers arriving as float must still gather (the .long() analogue)."""
    jax_emb = AtomEmbedding(8, 118, key=key)
    out_int = jax_emb(_batch(jnp.array([1, 6, 8])))
    out_float = jax_emb(_batch(jnp.array([1.0, 6.0, 8.0])))
    assert np.array_equal(np.asarray(out_int), np.asarray(out_float))


def test_atom_embedding_init_distribution(key):
    """Uniform(-sqrt(3), sqrt(3)) init -> bounded and ~unit variance (matches torch's)."""
    w = np.asarray(AtomEmbedding(64, 118, key=key).embeddings.weight)
    assert w.min() >= -(3**0.5) - 1e-6
    assert w.max() <= 3**0.5 + 1e-6
    assert abs(w.var() - 1.0) < 0.1  # var of U(-sqrt3, sqrt3) is exactly 1


def test_atom_embedding_no_key_is_zeros():
    """Without a key the table is zeros (weights are loaded afterwards)."""
    jax_emb = AtomEmbedding(8, 118, key=None)
    assert np.all(np.asarray(jax_emb.embeddings.weight) == 0.0)


def test_atom_embedding_shapes_and_out_dim(key):
    jax_emb = AtomEmbedding(8, 118, key=key)
    assert jax_emb.out_dim == 8
    out = jax_emb(_batch(jnp.array([1, 6, 8, 2])))
    assert out.shape == (4, 8)
