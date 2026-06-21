"""Numerical equivalence of the edge-axis memory levers in `...jax.optimisation`.

`ChunkedStack` streams a whole `AttentionInteractionNetwork` over its edge axis
via `lax.scan` so only a `chunk`-row slice of any per-edge tensor is ever live.
That rewrite must be a pure memory transform: for the same weights and inputs it
has to reproduce the un-chunked stack *exactly* (fp64), forwards AND through the
gradient -- otherwise it silently corrupts training.

These were originally validated only by the throughput benchmarks (shape, not
maths). Here we pin the maths directly, with a focus on the orbmol_v2 path that
the lever previously refused: a *conditioned* backbone (additive charge/spin
embeddings on the nodes), sigmoid gate, distance cutoff.

Run on CPU/fp64 (see conftest); shared weights => agreement is ~1e-12.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orb_models.common.models.jax.gns import AttentionInteractionNetwork
from orb_models.forcefield.models.jax.optimisation import (
    ChunkedStack,
    chunk_stacks,
    convert_to_chunked,
)

L, N, E = 8, 7, 23  # latent, nodes, edges (E prime => non-divisible chunks)
ATOL = 1e-10


def _inputs(seed=0):
    """Random stack inputs in fp64 (nodes, edges, senders, receivers, cutoff, conds)."""
    rng = np.random.default_rng(seed)
    nodes = jnp.asarray(rng.standard_normal((N, L)))
    edges = jnp.asarray(rng.standard_normal((E, L)))
    senders = jnp.asarray(rng.integers(0, N, size=E))
    receivers = jnp.asarray(rng.integers(0, N, size=E))
    # cutoff carries a trailing feature axis (get_cutoff_p4), values in (0, 1].
    cutoff = jnp.asarray(rng.uniform(0.1, 1.0, size=(E, 1)))
    cond_nodes = jnp.asarray(rng.standard_normal((N, L)))
    cond_edges = jnp.asarray(rng.standard_normal((E, L)))
    return nodes, edges, senders, receivers, cutoff, cond_nodes, cond_edges


def _make_stack(conditioning="none", attention_gate="sigmoid", distance_cutoff=True):
    return AttentionInteractionNetwork(
        latent_dim=L,
        num_mlp_layers=2,
        mlp_hidden_dim=16,
        attention_gate=attention_gate,
        conditioning=conditioning,
        distance_cutoff=distance_cutoff,
        activation="silu",
        mlp_norm="rms_norm",
        key=jax.random.PRNGKey(1),
    )


def _call(stack, args):
    nodes, edges, senders, receivers, cutoff, cn, ce = args
    return stack(nodes, edges, senders, receivers, cutoff, cond_nodes=cn, cond_edges=ce)


def _scalar(stack, args):
    """A weighted scalar of both outputs -- a non-trivial cotangent for the grad check."""
    out_nodes, out_edges = _call(stack, args)
    return (out_nodes**2).sum() + (jnp.sin(out_edges)).sum()


# --- forward equivalence ----------------------------------------------------
CONDITIONINGS = ["none", "additive", ("additive", "none"), ("none", "additive")]


@pytest.mark.parametrize("conditioning", CONDITIONINGS)
@pytest.mark.parametrize("chunk", [5, 8, E])  # non-divisor, divisor-ish, full (no-op)
@pytest.mark.parametrize("remat", [True, False])
def test_chunked_stack_forward_matches(conditioning, chunk, remat):
    stack = _make_stack(conditioning=conditioning)
    chunked = ChunkedStack(stack, chunk, remat)
    args = _inputs()

    ref_nodes, ref_edges = _call(stack, args)
    got_nodes, got_edges = _call(chunked, args)

    np.testing.assert_allclose(np.asarray(got_nodes), np.asarray(ref_nodes), atol=ATOL)
    np.testing.assert_allclose(np.asarray(got_edges), np.asarray(ref_edges), atol=ATOL)


# --- gradient equivalence ---------------------------------------------------
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
@pytest.mark.parametrize("chunk", [5, E])
def test_chunked_stack_grads_match(conditioning, chunk):
    stack = _make_stack(conditioning=conditioning)
    chunked = ChunkedStack(stack, chunk, remat=True)
    args = _inputs()

    # Grad w.r.t. every differentiable input (nodes, edges, cutoff, conds).
    argnums = (0, 1, 4, 5, 6)
    ref = jax.grad(lambda *a: _scalar(stack, a), argnums)(*args)
    got = jax.grad(lambda *a: _scalar(chunked, a), argnums)(*args)

    for r, g in zip(ref, got):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), atol=ATOL)


# --- unsupported paths fail loudly ------------------------------------------
def test_chunked_stack_rejects_softmax():
    stack = _make_stack(attention_gate="softmax")
    with pytest.raises(NotImplementedError, match="softmax|sigmoid"):
        _call(ChunkedStack(stack, 5, True), _inputs())


def test_chunked_stack_rejects_concatenative():
    stack = _make_stack(conditioning="concatenative")
    with pytest.raises(NotImplementedError, match="concatenative|additive"):
        _call(ChunkedStack(stack, 5, True), _inputs())


# --- bubble up: convert_to_chunked on a conditioned backbone ----------------
class _Wrapper(eqx.Module):
    """Minimal carrier so `convert_to_chunked` (which reaches `model.gns.gnn_stacks`)
    can be exercised end-to-end without the full orbmol_v2 head stack."""

    gns: eqx.Module


def _conditioned_gns(key):
    from orb_models.common.models.jax.conditioner import ChargeSpinConditioner
    from orb_models.common.models.jax.gns import MoleculeGNS
    from orb_models.common.models.jax.rbf import BesselBasis

    return MoleculeGNS(
        latent_dim=L,
        num_message_passing_steps=2,
        num_mlp_layers=2,
        mlp_hidden_dim=16,
        rbf_transform=BesselBasis(6.0, num_bases=8),
        use_embedding=True,
        num_node_out_features=3,
        activation="silu",
        mlp_norm="rms_norm",
        interaction_params={"distance_cutoff": True, "attention_gate": "sigmoid"},
        conditioner=ChargeSpinConditioner(L, key=key),
        conditioning_type="additive",
        key=key,
    )


def _gns_batch():
    import types

    rng = np.random.default_rng(3)
    n_node = np.array([2, 3, 1], dtype=np.int64)
    n = int(n_node.sum())
    per_node = np.concatenate([np.full(k, g) for g, k in enumerate(n_node)]).astype(np.int64)
    senders = rng.integers(0, n, size=E).astype(np.int64)
    receivers = rng.integers(0, n, size=E).astype(np.int64)
    z = rng.integers(1, 118, size=(n,)).astype(np.int64)
    return types.SimpleNamespace(
        node_features={
            "atomic_numbers": jnp.asarray(z),
            "atomic_numbers_embedding": jnp.asarray(
                np.eye(118)[z - 1]
            ),
        },
        edge_features={"vectors": jnp.asarray(rng.standard_normal((E, 3)))},
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        n_node=jnp.asarray(n_node),
        n_edge=jnp.asarray(np.array([8, 8, 7], dtype=np.int64)),
        per_node_graph_index=jnp.asarray(per_node),
        per_edge_graph_index=jnp.asarray(per_node[senders]),
        system_features={
            "total_charge": jnp.asarray(np.array([1.0, -1.0, 0.0])),
            "spin_multiplicity": jnp.asarray(np.array([2.0, 1.0, 0.0])),
        },
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_convert_to_chunked_matches_on_conditioned_backbone(key, checkpoint):
    model = _Wrapper(gns=_conditioned_gns(key))
    chunked = convert_to_chunked(
        model, chunk=5, chunk_encoder=True, checkpoint=checkpoint, ckpt_mode="full"
    )
    batch = _gns_batch()

    ref = model.gns(batch)
    got = chunked.gns(batch)
    for field in ("node_features", "edge_features", "pred"):
        np.testing.assert_allclose(
            np.asarray(got[field]), np.asarray(ref[field]), atol=ATOL
        )


@pytest.mark.parametrize("chunk_remat", [False, True])
def test_convert_to_chunked_grad_matches_on_conditioned_backbone(key, chunk_remat):
    model = _Wrapper(gns=_conditioned_gns(key))
    chunked = convert_to_chunked(
        model, chunk=5, chunk_encoder=True, chunk_remat=chunk_remat
    )
    batch = _gns_batch()

    def loss(gns):
        out = gns(batch)
        return (out["pred"] ** 2).sum()

    ref = eqx.filter_grad(loss)(model.gns)
    got = eqx.filter_grad(loss)(chunked.gns)

    ref_leaves = jax.tree_util.tree_leaves(eqx.filter(ref, eqx.is_inexact_array))
    got_leaves = jax.tree_util.tree_leaves(eqx.filter(got, eqx.is_inexact_array))
    assert len(ref_leaves) == len(got_leaves)
    for r, g in zip(ref_leaves, got_leaves):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), atol=ATOL)
