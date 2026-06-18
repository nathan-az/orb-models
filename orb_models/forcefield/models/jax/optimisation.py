from __future__ import annotations

import equinox as eqx

import jax
import jax.numpy as jnp


class RematStack(eqx.Module):
    """Per-gnn-stack gradient checkpointing, applied purely from the outside.

    Wraps one `AttentionInteractionNetwork` so its activations are dropped and
    recomputed in the backward pass (`eqx.filter_checkpoint`). We swap each stack
    for one of these via `eqx.tree_at` on the dynamic `gnn_stacks` list, so the
    library `MoleculeGNS.__call__` (`for gnn in self.gnn_stacks: gnn(...)`) is
    untouched -- the port stays pristine; checkpointing is an external concern.

    `inner` is passed as an ARGUMENT to the checkpointed function (not closed
    over) so filter_checkpoint can split its array params from its static config.
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
    """Stream an `MLPAndLayerNorm` over its leading (edge/node) axis in tiles.

    The lever for the *width* problem that checkpointing cannot touch: a Linear's
    backward `W̄ = Xᵀ·Ȳ` is a reduction over the leading axis E, so its working set
    is intrinsically `O(E·hidden)` no matter what is or isn't saved. Tiling turns
    that reduction into a running sum over chunks of `chunk` rows, so only
    `O(chunk·hidden)` is ever live.

    We express the loop as `jax.lax.map` (a `scan` underneath): reverse-mode
    transposes it into a loop whose cotangent w.r.t. the closed-over weights is
    *accumulated* across iterations -- the streaming backward is automatic. But a
    bare scan stashes every tile's residuals (re-materialising `[E, hidden]`), so
    `remat=True` wraps the body in `eqx.filter_checkpoint`: each tile's activations
    are recomputed in the backward, one tile at a time. The wrapped `inner` already
    handles a batched `[chunk, K]` input (its Linears/RMSNorm vmap internally), so
    a tile is just `inner(tile)`.

    `inner` is the ORIGINAL module, so the params/treedef are unchanged -- this is a
    pure call-time reshape, injected by `tree_at` exactly like the checkpointers,
    and reversible by swapping `inner` back in.
    """

    inner: eqx.Module
    chunk: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __call__(self, x: jax.Array, *, key=None) -> jax.Array:
        n = x.shape[0]
        # No-op fast path: one tile covers everything (or chunking disabled).
        if self.chunk <= 0 or self.chunk >= n:
            return self.inner(x, key=key)
        pad = (-n) % self.chunk
        xp = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
        tiles = xp.reshape(-1, self.chunk, *x.shape[1:])  # [n_tiles, chunk, K]
        body = eqx.filter_checkpoint(self.inner) if self.remat else self.inner
        out = jax.lax.map(body, tiles)  # [n_tiles, chunk, out]
        return out.reshape(-1, out.shape[-1])[:n]


class ChunkedStack(eqx.Module):
    """Tile a whole `AttentionInteractionNetwork` over its edge axis via `lax.scan`.

    `ChunkedMLP` only streams the edge MLP; measurements show that bottoms out at
    the *other* full-E tensors of one stack's forward-over-reverse backward (the
    `[E, 3L]` concat, the gathers, the attention) -- a floor neither edge-MLP
    chunking nor stack-remat breaks. This streams ALL of them: only a `chunk`-row
    slice of any per-edge tensor is ever live.

    The two node aggregations are `segment_sum`s -- reductions over the edge axis --
    so they accumulate across tiles in the scan CARRY (sum is associative; the easy
    case). `updated_edges` is the scan's stacked output. The node MLP runs ONCE
    afterwards on the aggregated `[N, 3L]` (N<<E, cheap). `remat=True` wraps the scan
    body so each tile's `[chunk, hidden]` activations are recomputed in the backward
    rather than stashed per iteration.

    Injected by `tree_at`, reusing `inner`'s ported weights -- no new params, nothing
    copied from torch. Implements only the released orb-v3 path (no conditioning,
    sigmoid gate); asserts anything else, since softmax would need a two-pass
    `segment_softmax` (a global per-node denominator before the weighted sum).
    """

    inner: eqx.Module
    chunk: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __call__(self, nodes, edges, senders, receivers, cutoff,
                 *, cond_nodes=None, cond_edges=None):
        m = self.inner
        if m._node_cond != "none" or m._edge_cond != "none":
            raise NotImplementedError("ChunkedStack supports conditioning='none' only.")
        if m._attention_gate != "sigmoid":
            raise NotImplementedError("ChunkedStack tiles the sigmoid gate only.")
        E, N, T, L = edges.shape[0], nodes.shape[0], self.chunk, m.latent_dim
        if T <= 0 or T >= E:  # no-op fast path -> exact original call
            return m(nodes, edges, senders, receivers, cutoff,
                     cond_nodes=cond_nodes, cond_edges=cond_edges)

        pad = (-E) % T

        def padrows(x, fill=0):
            return jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1), constant_values=fill)

        # Pad senders/receivers with N: segment_sum drops ids >= num_segments, so the
        # padding rows vanish from the aggregation; the OOB gather they trigger clamps
        # to a valid row whose (unused) result is sliced off the edge output below.
        e_t = padrows(edges).reshape(-1, T, edges.shape[-1])
        s_t = padrows(senders, N).reshape(-1, T)
        r_t = padrows(receivers, N).reshape(-1, T)
        c_t = padrows(cutoff).reshape(-1, T, cutoff.shape[-1])

        def body(carry, tile):
            agg_send, agg_recv = carry
            e, s, r, c = tile
            send_attn = jax.nn.sigmoid(m._send_attn(e))
            recv_attn = jax.nn.sigmoid(m._receive_attn(e))
            if m._distance_cutoff:
                send_attn, recv_attn = send_attn * c, recv_attn * c
            ef = jnp.concatenate([e, nodes[s], nodes[r]], axis=-1)  # [T, 3L]
            ue = m._edge_mlp(ef)  # [T, L]; the [T, hidden] activations are streamed
            agg_send = agg_send + jax.ops.segment_sum(ue * send_attn, s, N)
            agg_recv = agg_recv + jax.ops.segment_sum(ue * recv_attn, r, N)
            return (agg_send, agg_recv), ue

        scan_body = jax.checkpoint(body) if self.remat else body
        zeros = jnp.zeros((N, L), dtype=nodes.dtype)
        (agg_send, agg_recv), ue_t = jax.lax.scan(scan_body, (zeros, zeros), (e_t, s_t, r_t, c_t))

        updated_edges = ue_t.reshape(-1, L)[:E]
        # Original order: node_features = [nodes, received_attributes, sent_attributes].
        node_features = jnp.concatenate([nodes, agg_recv, agg_send], axis=-1)
        updated_nodes = m._node_mlp(node_features)
        return nodes + updated_nodes, edges + updated_edges


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
    """Stream the encoder's `edge_fn` (same `[E, hidden]` shape as a stack edge MLP,
    the leading full-E term once the stacks are chunked). Pure map -> reuse `ChunkedMLP`.
    """
    return eqx.tree_at(
        lambda m: m.gns._encoder.edge_fn,
        model,
        ChunkedMLP(model.gns._encoder.edge_fn, chunk, remat),
    )


# Gradient checkpointing is applied purely from the outside, by swapping
# `eqx.filter_checkpoint`-wrapped callables into the dynamic `gnn_stacks` list
# via `eqx.tree_at`. The library `MoleculeGNS.__call__`
# (`for gnn in self.gnn_stacks: gnn(...)`) is never touched -- the port stays
# pristine. `filter_checkpoint` returns an `eqx.Module` that partitions array
# params from static config itself, so wrapping a module instance directly (closed
# over as its `_fun`) is equivalent to passing it as an argument -- same residual
# set, same gradients (verified by `.scripts/debug/confirm_stack_residuals.py`).


def _checkpoint_stacks(model):
    """Outer remat: drop & recompute each whole stack's activations in backward.

    Reclaims the entire `__call__` body (gathers, concat, edge/node MLPs, attn),
    leaving only the per-stack input (`nodes`, `edges`) as the saved boundary.
    This is the dominant lever -- it removes the depth (num_message_passing)
    multiplier on activation memory.
    """
    return eqx.tree_at(
        lambda m: m.gns.gnn_stacks,
        model,
        [eqx.filter_checkpoint(s) for s in model.gns.gnn_stacks],
    )


def _checkpoint_encoder(model):
    """Remat the (un-looped) encoder: drop & recompute its node/edge MLP
    activations in the backward pass.

    Unlike the stacks, the encoder runs ONCE *before* the message-passing loop,
    so its `[E, mlp_hidden_dim]` edge-MLP (and `[N, mlp_hidden_dim]` node-MLP)
    activations are otherwise held live across every stack AND both grad
    traversals (the reverse pass in `_energy_and_grads` and the jvp in
    `surrogate`). That makes it a fixed memory floor the per-stack checkpoint
    never reaches -- this is the lever for that floor.
    """
    return eqx.tree_at(
        lambda m: m.gns._encoder,
        model,
        eqx.filter_checkpoint(model.gns._encoder),
    )


def _checkpoint_full(model):
    """Encoder remat composed with per-stack remat: removes both the depth
    multiplier (the stacks) and the encoder floor in one model. The two swaps are
    independent (the encoder is not inside `gnn_stacks`), so order is irrelevant;
    we checkpoint the stacks last for symmetry with `_checkpoint_nested`.
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
    checkpoint: bool = False,
    ckpt_mode: str = "stack",
):
    """Apply the chunking + checkpointing memory levers to a built model.

    The single entry point shared by the benchmark and the finetuning script. The
    order is fixed and load-bearing: chunk FIRST (while the stacks are still plain
    `AttentionInteractionNetwork`s whose `_edge_mlp` etc. resolve), then checkpoint
    (which may wrap the stacks in opaque modules).

    Args:
        model: a built `ConservativeRegressor` (random or ported weights).
        chunk: edge-axis tile width for `ChunkedStack`. 0 disables chunking.
        chunk_encoder: also stream the encoder's `edge_fn` (no effect if `chunk==0`).
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
        model = chunk_stacks(model, chunk)
        if chunk_encoder:
            model = chunk_encoder_edge(model, chunk)
    if checkpoint:
        model = CHECKPOINTERS[ckpt_mode](model)
    return model
