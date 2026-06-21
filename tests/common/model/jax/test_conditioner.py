"""Compare the jax charge/spin conditioner to the PyTorch reference.

Two levels:
  * ChargeSpinEmbedding -- the leaf that maps a 1D array of charge/spin values to
    an embedding. Covers all three embedding types and the spin==0 zeroing rule.
  * ChargeSpinConditioner -- embeds system-level charge/spin and scatters the
    per-graph code onto nodes. The jax version *gathers* (per_node_graph_index)
    where torch *repeat_interleave(n_node)*s; the two must agree, including on a
    zero-node padding graph.
"""

import types

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models.jax.conditioner import (
    ChargeSpinConditioner,
    ChargeSpinEmbedding,
)
from orb_models.common.models.nn_util import (
    ChargeSpinConditioner as TorchChargeSpinConditioner,
)
from orb_models.common.models.nn_util import (
    ChargeSpinEmbedding as TorchChargeSpinEmbedding,
)

LATENT = 16


@pytest.mark.equivalence
@pytest.mark.parametrize(
    "embedding_type",
    [
        "sin_emb",
        # The torch ref hardcasts inputs with `.float()` (fp32) in forward, which
        # collides with the fp64 Linear weights this harness builds (`F.linear`
        # doesn't type-promote). Not the production path (v1 uses sin_emb).
        pytest.param("lin_emb", marks=pytest.mark.xfail(reason="torch ref fp32 hardcast", strict=True)),
        "rand_emb",
    ],
)
@pytest.mark.parametrize("target", ["charge", "spin"])
def test_embedding_matches_torch(helpers, key, embedding_type, target):
    torch_emb = TorchChargeSpinEmbedding(
        num_channels=LATENT, embedding_target=target, embedding_type=embedding_type
    )
    jax_emb = ChargeSpinEmbedding(
        num_channels=LATENT, embedding_target=target, embedding_type=embedding_type, key=key
    )
    jax_emb = helpers.copy_charge_spin_embedding(jax_emb, torch_emb)

    # Include 0 (exercises the spin null-masking rule) and negatives (charges).
    values = np.array([0.0, 1.0, -2.0, 3.0, 2.0], dtype=np.float64)
    with torch.no_grad():
        torch_out = torch_emb(torch.tensor(values))
    jax_out = jax_emb(jnp.asarray(values))
    helpers.assert_close(jax_out, torch_out)


@pytest.mark.equivalence
@pytest.mark.parametrize("target", ["charge", "spin"])
def test_sin_embedding_grad_matches_torch(helpers, key, target):
    """Gradient w.r.t. the frequency param W must match torch -- in particular the
    spin==0 rows (zeroed via `jnp.where` here, in-place in torch) must contribute
    exactly zero grad, and that zeroing must not poison the rest of the batch."""
    torch_emb = TorchChargeSpinEmbedding(num_channels=LATENT, embedding_target=target)
    jax_emb = ChargeSpinEmbedding(num_channels=LATENT, embedding_target=target, key=key)
    jax_emb = helpers.copy_charge_spin_embedding(jax_emb, torch_emb)

    # Includes a 0 (the spin null case) alongside nonzero values.
    values = np.array([0.0, 1.0, -2.0, 3.0], dtype=np.float64)

    tv = torch.tensor(values)
    torch_emb.W.grad = None
    torch_emb(tv).sum().backward()
    torch_grad = torch_emb.W.grad

    jax_grad = eqx.filter_grad(lambda m: jnp.sum(m(jnp.asarray(values))))(jax_emb).W
    helpers.assert_close(jax_grad, torch_grad)

    if target == "spin":
        # The masked row is exactly zero in the forward (sanity on the rule itself).
        out0 = np.asarray(jax_emb(jnp.asarray(values)))[0]
        assert np.all(out0 == 0.0)


def test_neutral_singlet_is_finite(key):
    """The most common real molecule (charge 0, spin multiplicity 1) must embed to
    finite values: charge 0 -> [sin0, cos0] = [0, 1, ...], spin 1 -> sin/cos(W)."""
    cond = ChargeSpinConditioner(LATENT, key=key)
    batch = types.SimpleNamespace(
        system_features={
            "total_charge": jnp.asarray([0.0]),
            "spin_multiplicity": jnp.asarray([1.0]),
        },
        per_node_graph_index=jnp.asarray([0, 0, 0]),
        per_edge_graph_index=jnp.asarray([0, 0]),
    )
    node_embs, _ = cond(batch)
    assert node_embs.shape == (3, LATENT)
    assert np.isfinite(np.asarray(node_embs)).all()


def _conditioner_batch(convert, *, with_padding):
    """A 3-graph batch (+ optional empty padding graph). system_features hold the
    per-graph charge/spin; the index arrays/n_node give both forwards what they read."""
    n_node = [2, 3, 1]
    n_edge = [3, 4, 2]
    total_charge = [1.0, -2.0, 0.0]
    spin_multiplicity = [1.0, 2.0, 0.0]  # a 0 exercises the spin masking rule
    if with_padding:
        n_node.append(0)
        n_edge.append(0)
        total_charge.append(0.0)
        spin_multiplicity.append(0.0)

    per_node = np.concatenate([np.full(n, g) for g, n in enumerate(n_node)]).astype(np.int64)
    per_edge = np.concatenate(
        [np.full(e, g) for g, e in enumerate(n_edge)] + [np.zeros(0)]
    ).astype(np.int64)

    return types.SimpleNamespace(
        system_features={
            "total_charge": convert(np.array(total_charge)),
            "spin_multiplicity": convert(np.array(spin_multiplicity)),
        },
        n_node=convert(np.array(n_node).astype(np.int64)),
        n_edge=convert(np.array(n_edge).astype(np.int64)),
        per_node_graph_index=convert(per_node),
        per_edge_graph_index=convert(per_edge),
    )


@pytest.mark.equivalence
@pytest.mark.parametrize("with_padding", [False, True])
@pytest.mark.parametrize("emits_edge_embs", [False, True])
def test_conditioner_matches_torch(helpers, key, with_padding, emits_edge_embs):
    torch_cond = TorchChargeSpinConditioner(LATENT, emits_edge_embs=emits_edge_embs)
    jax_cond = ChargeSpinConditioner(LATENT, emits_edge_embs=emits_edge_embs, key=key)
    jax_cond = helpers.copy_charge_spin_conditioner(jax_cond, torch_cond)

    with torch.no_grad():
        t_nodes, t_edges = torch_cond(_conditioner_batch(torch.tensor, with_padding=with_padding))
    j_nodes, j_edges = jax_cond(_conditioner_batch(jnp.asarray, with_padding=with_padding))

    helpers.assert_close(j_nodes, t_nodes)
    if emits_edge_embs:
        helpers.assert_close(j_edges, t_edges)
    else:
        assert j_edges is None and t_edges is None
