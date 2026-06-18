"""Bucketed (FFD-packed) batching for the ASE SQLite dataset.

The plain finetuning loader (``finetune.build_train_loader``) hands the model a
*fixed-count* batch: ``batch_size`` systems regardless of how big they are. That
is fine for torch, but JAX needs every batch padded to one fixed ``(nodes, edges,
graphs)`` bucket so XLA compiles the step once -- and a fixed *count* of randomly
sized systems wastes the bucket (one big system forces the whole batch to pad up).

This module batches by *budget* instead. Per epoch it:

    1. shuffles the dataset index,
    2. streams it in windows of ``chunk_size`` (a streaming shuffle-buffer),
    3. first-fit-decreasing packs each window into buckets that respect the
       node/edge/graph budgets (:mod:`orb_models.common.dataset.packing`),
    4. shuffles the resulting buckets so consecutive steps are size-decorrelated
       (FFD sorts each window descending, so without this, steps would trend
       large -> small within every chunk).

Each bucket is just a list of dataset indices; the ``DataLoader`` fetches them
(parallel workers) and ``adapter.batch`` disjoint-concats them, exactly as the
plain loader collates a fixed-count batch.

**Sizes without building graphs.** FFD needs a size per system *before* fetching.
``n_node`` is the atom count, read straight from the ASE db's ``natoms`` column (a
~2 s scan of 1.58M rows, no graph construction). ``n_edge`` genuinely needs the
neighbour list, so we *estimate* it as ``n_atoms * degree_estimate`` -- with
``knn`` graph construction capped at ``max_num_neighbors`` this is a strict upper
bound when ``degree_estimate == max_num_neighbors``, and a tight estimate when set
to the dataset's mean degree. Because the estimate can undershoot the real edge
count, :class:`BucketCollator` re-checks the *real* totals after fetching and
drops the largest member(s) of any bucket that overflows the hard bucket caps,
warning and counting as it goes. Oversized single systems (bigger than a cap on
their own) are dropped by the packer up front, also counted.
"""

from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from orb_models.common.dataset import packing
from orb_models.common.dataset.packing import Budgets, Item


def read_natoms(db_path: str) -> list[int]:
    """Read every system's atom count from an ASE SQLite db, ordered by id.

    ASE stores ``natoms`` as a column on the ``systems`` table, so this is a
    single fast scan -- no positions/graphs touched. The returned list is aligned
    to the dataset index: ``natoms[i]`` is the system :class:`AseSqliteDataset`
    returns for ``idx == i`` (which fetches db row ``i + 1``).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT natoms FROM systems ORDER BY id").fetchall()
    finally:
        con.close()
    return [int(r[0]) for r in rows]


def make_items(natoms: list[int], degree_estimate: float) -> list[Item]:
    """Build packer :class:`Item`s from atom counts.

    ``edges`` is estimated as ``round(n_atoms * degree_estimate)``; ``nodes`` is
    exact; ``graphs`` is the redundant ``1`` so the graph cap is just another axis.
    """
    return [
        Item(
            index=i,
            sizes={
                "edges": int(round(n * degree_estimate)),
                "nodes": int(n),
                "graphs": 1,
            },
        )
        for i, n in enumerate(natoms)
    ]


class BucketBatchSampler(Sampler[list[int]]):
    """Yields lists of dataset indices, one per FFD-packed bucket.

    Args:
        natoms: per-system atom counts, aligned to dataset index (see
            :func:`read_natoms`).
        budgets: insertion-ordered packing budgets in *estimate* space, e.g.
            ``{"edges": 44000, "nodes": 700, "graphs": 190}``. First key is the
            FFD sort axis. These should sit at or below the hard bucket caps; give
            the edge axis headroom if ``degree_estimate`` may undershoot.
        degree_estimate: edges-per-atom used to size the (uncertain) edge axis.
            Set to ``max_num_neighbors`` for a strict upper bound, or the dataset
            mean degree for tight packing.
        chunk_size: streaming shuffle-buffer window FFD packs at a time.
        shuffle: reshuffle index + bucket order every epoch.
        seed: base RNG seed; epoch ``e`` uses ``seed + e``.
    """

    def __init__(
        self,
        natoms: list[int],
        budgets: Budgets,
        *,
        degree_estimate: float,
        chunk_size: int = 1000,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.items = make_items(natoms, degree_estimate)
        self.budgets = dict(budgets)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._dropped_oversized = 0
        self._len: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so a new shuffle is used (call before each epoch)."""
        self.epoch = epoch

    def _packed(self, seed: int) -> tuple[list[list[int]], int]:
        """Shuffle -> chunk -> FFD pack -> shuffle buckets. Returns (buckets, n_dropped)."""
        items = self.items
        rng = np.random.default_rng(seed)
        if self.shuffle:
            order = rng.permutation(len(items))
            items = [items[i] for i in order]
        res = packing.pack_dataset(items, self.budgets, chunk_size=self.chunk_size)
        buckets = [[it.index for it in b.items] for b in res.bins]
        if self.shuffle:
            perm = rng.permutation(len(buckets))
            buckets = [buckets[i] for i in perm]
        return buckets, len(res.dropped)

    def __iter__(self) -> Iterator[list[int]]:
        seed = self.seed + self.epoch if self.shuffle else self.seed
        buckets, n_dropped = self._packed(seed)
        self._len = len(buckets)
        if n_dropped and n_dropped != self._dropped_oversized:
            self._dropped_oversized = n_dropped
            warnings.warn(
                f"BucketBatchSampler dropped {n_dropped} system(s) that exceed a "
                f"single-bucket budget {self.budgets} on their own (edge axis uses "
                f"degree_estimate). These are skipped every epoch.",
                stacklevel=2,
            )
        yield from buckets

    def __len__(self) -> int:
        """Number of buckets (cached; computed with the base seed if unknown).

        Approximate across epochs -- a different shuffle gives a slightly different
        bucket count -- but exact for the most recent ``__iter__``. The training
        loop is driven by ``max_steps``, so this is only used for progress display.
        """
        if self._len is None:
            self._len = self._packed(self.seed)[0].__len__()
        return self._len


class BucketCollator:
    """Wraps ``adapter.batch`` with a real-size guard against bucket overflow.

    The sampler packs on an edge *estimate*; the real neighbour list may exceed it.
    This trims any bucket whose real totals breach the hard caps by dropping its
    largest-edge members first, warns, and counts. Returns ``None`` for a bucket
    that empties out entirely (a single system too big for the bucket) -- the
    training loop skips ``None`` batches.
    """

    def __init__(self, batch_fn, *, n_max: int, e_max: int, g_max: int):
        self.batch_fn = batch_fn
        self.n_max = n_max
        self.e_max = e_max
        self.g_max = g_max
        self.n_dropped_members = 0
        self.n_trimmed_buckets = 0

    def __call__(self, graphs: list):
        graphs = list(graphs)
        sizes = [(int(g.n_node.sum()), int(g.n_edge.sum())) for g in graphs]
        n_tot = sum(s[0] for s in sizes)
        e_tot = sum(s[1] for s in sizes)
        trimmed = False
        # Drop largest-edge members until real totals fit the hard bucket caps.
        while graphs and (
            n_tot > self.n_max or e_tot > self.e_max or len(graphs) > self.g_max
        ):
            j = max(range(len(graphs)), key=lambda k: sizes[k][1])
            n_j, e_j = sizes.pop(j)
            graphs.pop(j)
            n_tot -= n_j
            e_tot -= e_j
            self.n_dropped_members += 1
            trimmed = True
        if trimmed:
            self.n_trimmed_buckets += 1
            warnings.warn(
                f"BucketCollator trimmed a bucket whose real size exceeded caps "
                f"(nodes<={self.n_max}, edges<={self.e_max}, graphs<={self.g_max}); "
                f"dropped {self.n_dropped_members} member(s) total so far. Consider "
                f"raising degree_estimate or the edge budget.",
                stacklevel=2,
            )
        if not graphs:
            return None
        return self.batch_fn(graphs)


def build_bucket_sampler(
    db_path: str,
    budgets: Budgets,
    *,
    degree_estimate: float,
    chunk_size: int = 1000,
    shuffle: bool = True,
    seed: int = 0,
) -> BucketBatchSampler:
    """Convenience: read atom counts from ``db_path`` and build the sampler."""
    natoms = read_natoms(db_path)
    return BucketBatchSampler(
        natoms,
        budgets,
        degree_estimate=degree_estimate,
        chunk_size=chunk_size,
        shuffle=shuffle,
        seed=seed,
    )
