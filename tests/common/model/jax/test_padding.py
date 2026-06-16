"""Packing + padding + masking mechanics (JAX-internal).

Scope here is the JAX-side machinery in isolation:
  * `pack_graphs` groups single-system torch graphs to an edge budget and leans on
    torch `AtomGraphs.batch` for the disjoint concat (offsets / per_*_graph_index).
  * `pad_to_bucket` + masking is a NO-OP on the result: padded loss/grads equal the
    unpadded ones, and the dummy padding produces no NaN.
  * the jitted loss compiles ONCE across two batches that differ in content AND in
    real-graph count but share the bucket shape.

torch-vs-jax equivalence of the PADDED path across optimiser steps lives in
test_train_equivalence.py (the `padded` parametrization). All fp64 (conftest).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.jax.conservative_regressor import (
    compute_grads_reverse,
    total_loss,
)
from tests.common.model.jax.test_conservative_regressor import (
    _arrays,
    _build_real_features,
    _torch_graph,
)

WEIGHTS = {"energy": 1.0, "forces": 1.0, "stress": 1.0}


def _single_torch(n: int, e: int, seed: int) -> AtomGraphs:
    """A one-system torch `AtomGraphs` with random, non-self-loop edges (nonzero
    real edge vectors). The input to `pack_graphs`, which re-batches via AtomGraphs.batch."""
    rng = np.random.default_rng(seed)
    senders = rng.integers(0, n, size=e)
    receivers = (senders + rng.integers(1, n, size=e)) % n
    z = rng.integers(1, 30, size=n)
    return AtomGraphs(
        senders=torch.tensor(senders),
        receivers=torch.tensor(receivers),
        n_node=torch.tensor([n]),
        n_edge=torch.tensor([e]),
        node_features={
            "positions": torch.tensor(rng.standard_normal((n, 3)) * 2.0),
            "atomic_numbers": torch.tensor(z),
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                torch.tensor(z), num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((e, 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(rng.integers(-1, 2, size=(e, 3)).astype(float)),
        },
        system_features={
            "cell": torch.tensor(np.eye(3)[None] * 6.0),
            "pbc": torch.ones((1, 3), dtype=torch.bool),
        },
        node_targets={},
        edge_targets={},
        system_targets={},
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
    )


def _targets(graph: jgb.JaxAtomGraphs, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(graph.per_node_graph_index.shape[0])
    g = int(graph.n_node.shape[0])
    return {
        "energy": jnp.asarray(rng.standard_normal(g)),
        "forces": jnp.asarray(rng.standard_normal((n, 3))),
        "stress": jnp.asarray(rng.standard_normal((g, 6))),
    }


# --- packing: grouping policy, offsets delegated to AtomGraphs.batch ----------


def test_pack_graphs_groups_to_edge_budget():
    s0 = _single_torch(n=3, e=4, seed=0)
    s1 = _single_torch(n=2, e=2, seed=1)
    pool = [s0, s1, s0, s1, s0, s1]  # edge sizes [4,2,4,2,4,2]
    batches = jgb.pack_graphs(pool, n_max=1000, e_max=6)

    # greedy first-fit to the edge budget: (4,2) | (4,2) | (4,2)
    assert len(batches) == 3
    for b in batches:
        assert isinstance(b, AtomGraphs)
        np.testing.assert_array_equal(np.asarray(b.n_node), [3, 2])
        assert int(b.n_edge.sum()) == 6
        # offsets come from torch AtomGraphs.batch: system 1's nodes are graph 1.
        jb = jgb.to_jax(b)
        np.testing.assert_array_equal(
            np.asarray(jb.per_node_graph_index), [0, 0, 0, 1, 1])
        # system 1's senders/receivers were offset by system 0's node count (3).
        assert int(jb.senders.max()) >= 3


def test_pack_graphs_node_budget_guards():
    s0 = _single_torch(n=3, e=4, seed=0)
    s1 = _single_torch(n=2, e=2, seed=1)
    # generous edge budget, tight node budget that fits only one system at a time.
    batches = jgb.pack_graphs([s0, s1, s0], n_max=3, e_max=1000)
    assert [int(b.n_node.sum()) for b in batches] == [3, 2, 3]


# --- padding is a no-op on loss + grads --------------------------------------


def test_padded_loss_and_grads_match_unpadded(key):
    _torch, model = _build_real_features(key)
    a = _arrays()
    real = jgb.to_jax(_torch_graph(a))  # the 2-system disjoint batch
    targets = _targets(real, seed=3)

    n_pad, e_pad, g_pad = a["N"] + 7, a["E"] + 11, a["G"] + 3
    padded = jgb.pad_to_bucket(real, n_pad, e_pad, g_pad)
    padded_targets = jgb.pad_targets(targets, n_pad, g_pad)

    loss_real, bd_real = total_loss(model, real, targets, WEIGHTS)
    loss_pad, bd_pad = total_loss(model, padded, padded_targets, WEIGHTS)

    assert np.isfinite(np.asarray(loss_pad))  # no NaN from the dummy padding graph
    for k in bd_real:
        np.testing.assert_allclose(
            np.asarray(bd_pad[k]), np.asarray(bd_real[k]), atol=1e-10, rtol=1e-10,
            err_msg=f"loss term {k!r} differs after padding")

    g_real, _ = compute_grads_reverse(model, real, targets, WEIGHTS)
    g_pad, _ = compute_grads_reverse(model, padded, padded_targets, WEIGHTS)
    lr = jax.tree.leaves(eqx.filter(g_real, eqx.is_inexact_array))
    lp = jax.tree.leaves(eqx.filter(g_pad, eqx.is_inexact_array))
    assert len(lr) == len(lp)
    for ar, br in zip(lr, lp):
        np.testing.assert_allclose(np.asarray(br), np.asarray(ar), atol=1e-9, rtol=1e-9)


def test_compile_once_across_batches(key):
    """Different content AND different real-graph count, same bucket -> one trace."""
    _torch, model = _build_real_features(key)
    s0 = _single_torch(n=3, e=4, seed=0)
    s1 = _single_torch(n=2, e=2, seed=1)

    one = jgb.to_jax(AtomGraphs.batch([s0]))  # 1 system (3 atoms, 4 edges)
    two = jgb.to_jax(AtomGraphs.batch([s0, s1]))  # 2 systems (5 atoms, 6 edges)
    n_pad, e_pad, g_pad = 12, 14, 5  # dominates both packings
    b1 = jgb.pad_to_bucket(one, n_pad, e_pad, g_pad)
    b2 = jgb.pad_to_bucket(two, n_pad, e_pad, g_pad)
    t1 = jgb.pad_targets(_targets(one, 30), n_pad, g_pad)
    t2 = jgb.pad_targets(_targets(two, 31), n_pad, g_pad)

    traces = {"n": 0}

    @eqx.filter_jit
    def loss_only(m, graph, targets):
        traces["n"] += 1  # runs only while tracing
        return total_loss(m, graph, targets, WEIGHTS)[0]

    l1 = loss_only(model, b1, t1)
    l2 = loss_only(model, b2, t2)
    assert np.isfinite(np.asarray(l1)) and np.isfinite(np.asarray(l2))
    assert traces["n"] == 1, "padding failed: a second compile was triggered"
