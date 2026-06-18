"""Tests for FFD bucket batching (packer + sampler + collator)."""

from __future__ import annotations

import numpy as np
import pytest

from orb_models.common.dataset import packing
from orb_models.common.dataset.bucket_sampler import (
    BucketBatchSampler,
    BucketCollator,
    make_items,
)

BUDGETS = {"edges": 4000, "nodes": 200, "graphs": 8}
DEGREE = 50.0


def _sampler(natoms, **kw):
    return BucketBatchSampler(natoms, BUDGETS, degree_estimate=DEGREE, chunk_size=10, **kw)


def test_every_index_appears_once_per_epoch():
    natoms = [int(x) for x in np.random.default_rng(0).integers(2, 40, size=137)]
    buckets = list(_sampler(natoms, seed=3))
    flat = sorted(i for b in buckets for i in b)
    assert flat == list(range(len(natoms)))


def test_no_bucket_exceeds_budgets():
    natoms = [int(x) for x in np.random.default_rng(1).integers(2, 40, size=200)]
    s = _sampler(natoms, seed=7)
    for bucket in s:
        nodes = sum(natoms[i] for i in bucket)
        edges = sum(round(natoms[i] * DEGREE) for i in bucket)
        assert nodes <= BUDGETS["nodes"]
        assert edges <= BUDGETS["edges"]
        assert len(bucket) <= BUDGETS["graphs"]


def test_deterministic_given_seed():
    natoms = [int(x) for x in np.random.default_rng(2).integers(2, 40, size=90)]
    a = list(_sampler(natoms, seed=5))
    b = list(_sampler(natoms, seed=5))
    assert a == b


def test_epoch_changes_order():
    natoms = [int(x) for x in np.random.default_rng(3).integers(2, 40, size=90)]
    s = _sampler(natoms, seed=5)
    s.set_epoch(0)
    first = list(s)
    s.set_epoch(1)
    second = list(s)
    assert first != second  # a different shuffle -> different packing/order
    # Still a valid partition of all indices.
    assert sorted(i for b in second for i in b) == list(range(len(natoms)))


def test_oversized_examples_dropped_and_counted():
    # One system alone exceeds the node budget -> packer drops it.
    natoms = [10, 10, 10, BUDGETS["nodes"] + 50, 10]
    s = _sampler(natoms, seed=0)
    with pytest.warns(UserWarning, match="exceed a single-bucket budget"):
        buckets = list(s)
    flat = sorted(i for b in buckets for i in b)
    assert 3 not in flat  # the oversized one is gone
    assert flat == [0, 1, 2, 4]


def test_no_shuffle_is_identity_order():
    natoms = [5, 5, 5, 5, 5]
    s = BucketBatchSampler(
        natoms, BUDGETS, degree_estimate=DEGREE, chunk_size=10, shuffle=False
    )
    # All five fit one bucket (5*5=25 nodes, 5*250=1250 edges, 5 graphs).
    assert list(s) == [[0, 1, 2, 3, 4]]


def test_make_items_edge_estimate():
    items = make_items([10, 20], degree_estimate=62.0)
    assert items[0].sizes == {"edges": 620, "nodes": 10, "graphs": 1}
    assert items[1].sizes == {"edges": 1240, "nodes": 20, "graphs": 1}


# --- collator ----------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in exposing .n_node.sum()/.n_edge.sum() like AtomGraphs."""

    def __init__(self, n, e):
        self.n_node = np.array([n])
        self.n_edge = np.array([e])


def test_collator_passthrough_when_within_caps():
    coll = BucketCollator(lambda gs: gs, n_max=100, e_max=1000, g_max=8)
    graphs = [_FakeGraph(10, 100), _FakeGraph(20, 200)]
    out = coll(graphs)
    assert out == graphs
    assert coll.n_dropped_members == 0


def test_collator_trims_overflowing_bucket():
    coll = BucketCollator(lambda gs: gs, n_max=100, e_max=300, g_max=8)
    # Real edges 100+500 = 600 > 300 -> drop the largest-edge member (the 500).
    graphs = [_FakeGraph(10, 100), _FakeGraph(20, 500)]
    with pytest.warns(UserWarning, match="trimmed a bucket"):
        out = coll(graphs)
    assert len(out) == 1 and out[0].n_edge[0] == 100
    assert coll.n_dropped_members == 1
    assert coll.n_trimmed_buckets == 1


def test_collator_returns_none_when_emptied():
    coll = BucketCollator(lambda gs: gs, n_max=5, e_max=10, g_max=8)
    with pytest.warns(UserWarning):
        out = coll([_FakeGraph(50, 5000)])
    assert out is None


def test_pack_dataset_caller_owns_order():
    # pack_dataset is pure: same items + budgets -> same bins, no internal shuffle.
    items = make_items([5, 5, 5, 5], degree_estimate=10.0)
    r1 = packing.pack_dataset(items, BUDGETS, chunk_size=2)
    r2 = packing.pack_dataset(items, BUDGETS, chunk_size=2)
    assert [[i.index for i in b.items] for b in r1.bins] == [
        [i.index for i in b.items] for b in r2.bins
    ]
