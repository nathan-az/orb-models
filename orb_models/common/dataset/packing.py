"""Chunked first-fit-decreasing (FFD) packing of graphs into fixed-budget buckets.

JAX training needs every batch padded to a *fixed* shape so XLA compiles the step
once (see ``graph_batch.pad_to_bucket``). A bucket is bounded by several budgets,
each an axis of the padded arrays:

    * edges  -> ``[n_edge, ...]``  message passing (cost-dominant axis)
    * nodes  -> ``[n_node, ...]``  per-atom features, forces
    * graphs -> ``[n_graph, ...]`` per-system energy (segment_sum pooling),
                                   stress as ``[n_graph, 6]``, ...

Budgets are a single **insertion-ordered** ``dict`` mapping an axis name to its
cap, e.g. ``{"edges": 44000, "nodes": 700, "graphs": 190}``. The *first* key is
the FFD sort key (sort by it descending so the big, awkward items seed buckets);
*all* keys are enforced when fitting. Each ``Item`` carries a matching ``sizes``
dict -- including a redundant ``graphs: 1`` -- so graphs is just another axis that
sums over a bucket rather than a special case read off afterward.

Edges is the natural sort key: by far the widest dynamic range (0..~35k vs nodes
1..444 vs graphs always 1), and message passing is O(edges) in FLOPs and memory,
so packing it tightly maximises utilisation. Nodes/graphs are secondary guards
against a pathological bucket of many tiny dense graphs.

Strategy: stream the (optionally shuffled) index in windows of ``chunk_size``;
within each window sort by the primary axis descending and first-fit each item
into the first open bucket it fits on every axis, else open a new one. This
emulates a streaming shuffle-buffer rather than globally sorting the dataset.

This module is the reusable *algorithm* -- it has no I/O and no torch dependency.
``BucketBatchSampler`` (``bucket_sampler.py``) drives it from the live dataset;
the standalone ``training/packing.py`` drives it from an offline sizes CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, NamedTuple

Budgets = Mapping[str, int]


class Item(NamedTuple):
    """One example: its row index plus a size on each budgeted axis."""

    index: int
    sizes: dict[str, int]


@dataclass
class Bin:
    """A bucket accumulating items; ``totals`` sums each axis across items."""

    items: list[Item] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)

    def fits(self, item: Item, budgets: Budgets) -> bool:
        return all(
            self.totals.get(k, 0) + item.sizes.get(k, 0) <= cap
            for k, cap in budgets.items()
        )

    def add(self, item: Item) -> None:
        self.items.append(item)
        for k, v in item.sizes.items():
            self.totals[k] = self.totals.get(k, 0) + v


def pack_chunk(items: Iterable[Item], budgets: Budgets) -> tuple[list[Bin], list[Item]]:
    """First-fit-decreasing pack one chunk. Returns (bins, dropped).

    Sort key is the first budget axis. ``dropped`` holds items that exceed some
    axis budget on their own and can never fit an empty bucket.
    """
    sort_key = next(iter(budgets))
    bins: list[Bin] = []
    dropped: list[Item] = []
    for item in sorted(items, key=lambda it: it.sizes.get(sort_key, 0), reverse=True):
        if any(item.sizes.get(k, 0) > cap for k, cap in budgets.items()):
            dropped.append(item)
            continue
        for b in bins:  # first-fit: scan buckets in creation order
            if b.fits(item, budgets):
                b.add(item)
                break
        else:
            new = Bin()
            new.add(item)
            bins.append(new)
    return bins, dropped


def iter_chunks(items: list[Item], chunk_size: int) -> Iterator[list[Item]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


@dataclass
class PackResult:
    bins: list[Bin]
    dropped: list[Item]
    budgets: dict[str, int]
    chunk_size: int
    n_chunks: int

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    def pad_fraction(self, axis: str) -> float:
        """Fraction of ``axis`` budget left as padding, averaged over buckets."""
        if not self.bins:
            return 0.0
        used = sum(b.totals.get(axis, 0) for b in self.bins)
        return 1.0 - used / (self.budgets[axis] * self.n_bins)

    def bucket_totals(self, axis: str) -> list[int]:
        return [b.totals.get(axis, 0) for b in self.bins]


def pack_dataset(
    items: list[Item],
    budgets: Budgets,
    *,
    chunk_size: int,
    num_chunks: int | None = None,
) -> PackResult:
    """Window the (pre-ordered) index into chunks and FFD-pack each independently.

    The caller owns shuffling -- pass ``items`` already in the order you want so
    this stays pure/deterministic.
    """
    items = list(items)
    all_bins: list[Bin] = []
    all_dropped: list[Item] = []
    chunks_done = 0
    for chunk in iter_chunks(items, chunk_size):
        if num_chunks is not None and chunks_done >= num_chunks:
            break
        bins, dropped = pack_chunk(chunk, budgets)
        all_bins.extend(bins)
        all_dropped.extend(dropped)
        chunks_done += 1

    return PackResult(
        bins=all_bins,
        dropped=all_dropped,
        budgets=dict(budgets),
        chunk_size=chunk_size,
        n_chunks=chunks_done,
    )
