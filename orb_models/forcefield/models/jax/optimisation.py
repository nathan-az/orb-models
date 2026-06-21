from __future__ import annotations

import equinox as eqx

import jax
import jax.numpy as jnp


class RematStack(eqx.Module):
    """Gradient-checkpoint one `AttentionInteractionNetwork`.

    Wraps a stack so its activations are dropped in the forward and recomputed in the
    backward (`eqx.filter_checkpoint`); swapped in via `eqx.tree_at`. Same call
    signature as the wrapped stack. `inner` is passed as an argument to the
    checkpointed function (not closed over) so `filter_checkpoint` can split its
    array params from its static config.
    """

    inner: eqx.Module

    @staticmethod
    def _apply(inner, nodes, edges, senders, receivers, cutoff, cond_nodes, cond_edges):
        return inner(nodes, edges, senders, receivers, cutoff,
                     cond_nodes=cond_nodes, cond_edges=cond_edges)

    def __call__(self, nodes, edges, senders, receivers, cutoff,
                 *, cond_nodes=None, cond_edges=None):
        return eqx.filter_checkpoint(self._apply)(
            self.inner, nodes, edges, senders, receivers, cutoff, cond_nodes, cond_edges
        )


def _checkpoint_stacks_manual(model):
    """Replace every gnn stack with a rematerialising wrapper (external swap)."""
    return eqx.tree_at(
        lambda m: m.gns.gnn_stacks,
        model,
        [RematStack(s) for s in model.gns.gnn_stacks],
    )


class ChunkedMLP(eqx.Module):
    """Run an `MLPAndLayerNorm` over its leading (edge/node) axis in `chunk`-row tiles.

    Equivalent to `inner(x)` but with the MLP's working set reduced from
    `O(rows·hidden)` to `O(chunk·hidden)`: `jax.lax.map` streams the tiles and the
    weight gradients accumulate across them automatically. `remat=True` recomputes
    each tile's activations in the backward instead of storing them. `inner` keeps its
    original weights, so this is a pure call-time transform swapped in via `tree_at`.

    Args:
        x: `[rows, K]` input. Returns `[rows, out]`.
    """

    inner: eqx.Module
    chunk: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __call__(self, x: jax.Array, *, key=None) -> jax.Array:
        n_rows = x.shape[0]
        if self.chunk <= 0 or self.chunk >= n_rows:  # no tiling needed
            return self.inner(x, key=key)
        n_pad = (-n_rows) % self.chunk
        padded = jnp.pad(x, [(0, n_pad)] + [(0, 0)] * (x.ndim - 1))
        tiles = padded.reshape(-1, self.chunk, *x.shape[1:])  # [n_tiles, chunk, K]
        body = eqx.filter_checkpoint(self.inner) if self.remat else self.inner
        out = jax.lax.map(body, tiles)  # [n_tiles, chunk, out]
        return out.reshape(-1, out.shape[-1])[:n_rows]


class ChunkedStack(eqx.Module):
    """Run a whole `AttentionInteractionNetwork` over its edge axis in `chunk`-row tiles.

    Equivalent to the wrapped stack but with a smaller memory footprint: only a
    `chunk`-row slice of any per-edge tensor is live at once. A `lax.scan` streams the
    edges; the node aggregations (edge-axis `segment_sum`s) accumulate in the scan
    carry, and the node MLP runs once afterwards on the aggregated nodes. `remat=True`
    recomputes each tile's activations in the backward pass instead of storing them.

    `inner` keeps its original weights, so this is a pure call-time transform swapped
    in via `eqx.tree_at`. Same call signature and outputs as `inner`.

    Supports the sigmoid gate with optional *additive* charge/spin conditioning on
    nodes and/or edges (the orbmol_v2 config). Raises `NotImplementedError` for the
    `softmax` gate (its per-node denominator can't be formed in a single streaming
    pass) and for `concatenative` conditioning (it widens the tiled edge/attn inputs).
    """

    inner: eqx.Module
    chunk: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __call__(self, nodes, edges, senders, receivers, cutoff,
                 *, cond_nodes=None, cond_edges=None):
        stack = self.inner
        if stack._node_cond not in ("none", "additive") or stack._edge_cond not in ("none", "additive"):
            raise NotImplementedError(
                "ChunkedStack supports conditioning 'none'/'additive' only; "
                f"got node={stack._node_cond!r} edge={stack._edge_cond!r} "
                "(concatenative would widen the tiled edge/attn inputs)."
            )
        if stack._attention_gate != "sigmoid":
            raise NotImplementedError("ChunkedStack tiles the sigmoid gate only.")

        n_edges, n_nodes = edges.shape[0], nodes.shape[0]
        tile_size, latent_dim = self.chunk, stack.latent_dim
        if tile_size <= 0 or tile_size >= n_edges:  # no tiling needed -> exact original call
            return stack(nodes, edges, senders, receivers, cutoff,
                         cond_nodes=cond_nodes, cond_edges=cond_edges)

        # Additive node conditioning folds onto the small [N, L] nodes once, up front:
        # this conditioned `nodes` feeds both the gathers and the node residual, exactly
        # as the un-chunked stack does. (Edge conditioning is applied per tile, below.)
        if stack._node_cond == "additive" and cond_nodes is not None:
            nodes = nodes + stack._cond_node_proj(cond_nodes)
        condition_edges = stack._edge_cond == "additive" and cond_edges is not None

        n_pad = (-n_edges) % tile_size

        def pad_rows(x, fill=0):
            return jnp.pad(x, [(0, n_pad)] + [(0, 0)] * (x.ndim - 1), constant_values=fill)

        def to_tiles(x):
            return pad_rows(x).reshape(-1, tile_size, *x.shape[1:])

        # Pad sender/receiver ids with N: segment_sum drops ids >= num_segments, so the
        # padding rows vanish from the aggregation; the out-of-bounds gather they trigger
        # clamps to a valid row whose (unused) result is sliced off the edge output below.
        edge_tiles = to_tiles(edges)
        sender_tiles = pad_rows(senders, n_nodes).reshape(-1, tile_size)
        receiver_tiles = pad_rows(receivers, n_nodes).reshape(-1, tile_size)
        cutoff_tiles = to_tiles(cutoff)
        # cond_edges rides along as a tiled input so its projection stays O(chunk); a
        # zero placeholder keeps the scan signature uniform when there's no edge cond.
        cond_edge_tiles = to_tiles(cond_edges if condition_edges else jnp.zeros_like(edges))

        def body(carry, tile):
            sent_agg, received_agg = carry
            edge, sender, receiver, cutoff_tile, cond_edge = tile
            if condition_edges:
                edge = edge + stack._cond_edge_proj(cond_edge)  # feeds attn + edge mlp
            send_attn = jax.nn.sigmoid(stack._send_attn(edge))
            recv_attn = jax.nn.sigmoid(stack._receive_attn(edge))
            if stack._distance_cutoff:
                send_attn, recv_attn = send_attn * cutoff_tile, recv_attn * cutoff_tile
            edge_mlp_input = jnp.concatenate([edge, nodes[sender], nodes[receiver]], axis=-1)
            edge_update = stack._edge_mlp(edge_mlp_input)
            sent_agg = sent_agg + jax.ops.segment_sum(edge_update * send_attn, sender, n_nodes)
            received_agg = received_agg + jax.ops.segment_sum(edge_update * recv_attn, receiver, n_nodes)
            # Carry out the new edge value (conditioned edge + update) so the scan's
            # stacked output is the residual edges directly -- matches `edges + updated_edges`.
            return (sent_agg, received_agg), edge + edge_update

        scan_body = jax.checkpoint(body) if self.remat else body
        zeros = jnp.zeros((n_nodes, latent_dim), dtype=nodes.dtype)
        (sent_agg, received_agg), new_edge_tiles = jax.lax.scan(
            scan_body, (zeros, zeros),
            (edge_tiles, sender_tiles, receiver_tiles, cutoff_tiles, cond_edge_tiles),
        )

        new_edges = new_edge_tiles.reshape(-1, latent_dim)[:n_edges]
        # Original order: node_features = [nodes, received_attributes, sent_attributes].
        node_features = jnp.concatenate([nodes, received_agg, sent_agg], axis=-1)
        updated_nodes = stack._node_mlp(node_features)
        return nodes + updated_nodes, new_edges


def chunk_stacks(model, chunk: int, *, remat: bool = True):
    """Swap every gnn stack for a `ChunkedStack` (external swap, plain stacks only).

    Apply before any `_checkpoint_stacks` (which would wrap the stacks and hide
    their fields), so we chunk first and checkpoint second.
    """
    return eqx.tree_at(
        lambda m: m.gns.gnn_stacks,
        model,
        [ChunkedStack(s, chunk, remat) for s in model.gns.gnn_stacks],
    )


def chunk_encoder_edge(model, chunk: int, *, remat: bool = True):
    """Wrap the encoder's `edge_fn` in a `ChunkedMLP` so it too streams over the edge
    axis (the leading full-edge term once the stacks are chunked).
    """
    return eqx.tree_at(
        lambda m: m.gns._encoder.edge_fn,
        model,
        ChunkedMLP(model.gns._encoder.edge_fn, chunk, remat),
    )


def _checkpoint_stacks(model):
    """Checkpoint every gnn stack: recompute each stack's `__call__` body in the
    backward, keeping only the per-stack inputs (`nodes`, `edges`) as the saved
    boundary. Removes the depth (num_message_passing) multiplier on activation memory.
    """
    return eqx.tree_at(
        lambda m: m.gns.gnn_stacks,
        model,
        [eqx.filter_checkpoint(s) for s in model.gns.gnn_stacks],
    )


def _checkpoint_encoder(model):
    """Checkpoint the encoder: recompute its node/edge MLP activations in the backward.

    The encoder runs once before the message-passing loop, so its activations would
    otherwise stay live across every stack and both grad traversals -- a memory floor
    the per-stack checkpoint never reaches.
    """
    return eqx.tree_at(
        lambda m: m.gns._encoder,
        model,
        eqx.filter_checkpoint(model.gns._encoder),
    )


def _checkpoint_full(model):
    """Checkpoint the encoder and every stack: removes both the depth multiplier and
    the encoder floor. The two swaps are independent, so order is irrelevant.
    """
    return _checkpoint_stacks(_checkpoint_encoder(model))


# Public registry of checkpointing policies (key -> model transform).
CHECKPOINTERS = {
    "stack_manual": _checkpoint_stacks_manual,
    "stack": _checkpoint_stacks,
    "encoder": _checkpoint_encoder,
    "full": _checkpoint_full,
}


def convert_to_chunked(
    model,
    *,
    chunk: int = 0,
    chunk_encoder: bool = False,
    chunk_remat: bool = True,
    checkpoint: bool = False,
    ckpt_mode: str = "stack",
):
    """Apply the chunking + checkpointing memory levers to a built model.

    Chunks first (while the stacks are still plain `AttentionInteractionNetwork`s whose
    `_edge_mlp` etc. resolve), then checkpoints (which may wrap the stacks in opaque
    modules) -- this order is load-bearing.

    Args:
        model: a built `ConservativeRegressor` (random or ported weights).
        chunk: edge-axis tile width for `ChunkedStack`. 0 disables chunking.
        chunk_encoder: also stream the encoder's `edge_fn` (no effect if `chunk==0`).
        chunk_remat: recompute each tile in the backward pass instead of storing it
            (no effect if `chunk==0`). Effectively mandatory whenever the energy is
            differentiated -- which is *every* path in this conservative model, since
            forces are `-dE/dpos`. A `lax.scan`'s reverse pass otherwise materialises
            every tile's residuals as dense stacked arrays (`O(n_edges)`, and unlike
            the fused unchunked stack XLA can't free them), so `chunk` + `False`
            OOMs *worse* than no chunking at all. Only set `False` for a genuinely
            forward-only call (none exists here today); it trades the ~50% recompute
            cost for that blow-up otherwise.
        checkpoint: enable activation rematerialisation.
        ckpt_mode: which policy from `CHECKPOINTERS` (only used if `checkpoint`).

    Returns:
        The transformed model (a new pytree; the input is untouched).

    Raises:
        ValueError: on an unknown `ckpt_mode`, or the `chunk` + `edge`/`both`
            incompatibility (those modes reach into `s._edge_mlp`, which a
            `ChunkedStack` no longer exposes).
    """
    if checkpoint and ckpt_mode not in CHECKPOINTERS:
        raise ValueError(
            f"ckpt_mode must be one of {sorted(CHECKPOINTERS)}; got {ckpt_mode!r}."
        )
    if chunk:
        if checkpoint and ckpt_mode in ("edge", "both"):
            raise ValueError(
                f"chunk replaces each stack with a ChunkedStack, so ckpt_mode "
                f"{ckpt_mode!r} (which reaches into s._edge_mlp) is incompatible; "
                "use none/stack/stack_manual/encoder/full."
            )
        model = chunk_stacks(model, chunk, remat=chunk_remat)
        if chunk_encoder:
            model = chunk_encoder_edge(model, chunk, remat=chunk_remat)
    if checkpoint:
        model = CHECKPOINTERS[ckpt_mode](model)
    return model
