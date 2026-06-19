"""Charge/spin conditioning for the JAX MoleculeGNS.

Port of `orb_models.common.models.nn_util.{ChargeSpinEmbedding,
ChargeSpinConditioner}`. Used by the OrbMol-v1 models (orb-v3 finetuned on
OMol25), which condition the backbone on system-level total charge and spin
multiplicity. OrbMol-v1 uses `embedding_type="sin_emb"` with node-only,
*additive* conditioning; the other embedding types are ported for fidelity.

jit note: the torch conditioner scatters per-graph embeddings onto nodes with
`repeat_interleave(n_node)`, whose output length is data-dependent. Here we
instead *gather* with `per_node_graph_index` (a fixed-length index array the
batch already carries), so the conditioned forward jits with static shapes.
"""

from __future__ import annotations

import math

import equinox as eqx

import jax
import jax.numpy as jnp
from orb_models.common.atoms.jax.graph_batch import JaxAtomGraphs


class ChargeSpinEmbedding(eqx.Module):
    """Embed a 1D array of charge or spin values. Port of the torch ref.

    - ``sin_emb`` (== ``pos_emb``): random-frequency Fourier features. The single
      parameter ``W`` has shape ``(num_channels // 4,)``; the output dim is
      ``num_channels // 2`` (sin and cos halves concatenated).
    - ``lin_emb``: a learned ``Linear(1, num_channels // 2)``.
    - ``rand_emb``: an integer lookup table over a clamped value range.
    """

    embedding_target: str = eqx.field(static=True)
    embedding_type: str = eqx.field(static=True)
    # sin_emb / pos_emb
    W: jax.Array | None
    # lin_emb
    lin_emb: eqx.nn.Linear | None
    # rand_emb
    rand_emb: eqx.nn.Embedding | None
    min_val: int = eqx.field(static=True)
    max_val: int = eqx.field(static=True)
    offset: int = eqx.field(static=True)

    def __init__(
        self,
        num_channels: int,
        embedding_target: str = "charge",
        embedding_type: str = "sin_emb",
        scale: float = 1.0,
        *,
        key: jax.Array | None = None,
    ):
        assert embedding_target in ("charge", "spin")
        assert embedding_type in ("sin_emb", "pos_emb", "lin_emb", "rand_emb")
        self.embedding_target = embedding_target
        self.embedding_type = embedding_type

        dim = num_channels // 2  # half the latent for charge, half for spin
        self.W = None
        self.lin_emb = None
        self.rand_emb = None
        self.min_val = 0
        self.max_val = 0
        self.offset = 0

        if embedding_type in ("sin_emb", "pos_emb"):
            if key is None:
                self.W = jnp.zeros(dim // 2)
            else:
                self.W = jax.random.normal(key, (dim // 2,)) * scale
        elif embedding_type == "lin_emb":
            lin_key = key if key is not None else jax.random.PRNGKey(0)
            self.lin_emb = eqx.nn.Linear(1, dim, key=lin_key)
        elif embedding_type == "rand_emb":
            if embedding_target == "charge":
                self.min_val, self.max_val, self.offset = -100, 100, 100
                table_size = 201
            else:
                self.min_val, self.max_val, self.offset = 0, 100, 0
                table_size = 101
            emb_key = key if key is not None else jax.random.PRNGKey(0)
            self.rand_emb = eqx.nn.Embedding(table_size, dim, key=emb_key)

    def __call__(self, values: jax.Array) -> jax.Array:
        assert values.ndim == 1, "Expected 1D tensor"
        values = values.astype(jnp.result_type(values, jnp.float32))

        if self.embedding_type in ("sin_emb", "pos_emb"):
            x_proj = values[:, None] * self.W[None, :] * 2 * math.pi
            emb = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
            if self.embedding_target == "spin":
                # Null spin (multiplicity 0) contributes nothing.
                emb = jnp.where((values == 0)[:, None], 0.0, emb)
            return emb

        if self.embedding_type == "lin_emb":
            values_ = values
            if self.embedding_target == "spin":
                values_ = jnp.where(values_ == 0, -100.0, values_)
            return jax.vmap(self.lin_emb)(values_[:, None])

        # rand_emb
        values_rounded = jnp.round(values).astype(jnp.int32)
        values_clamped = jnp.clip(values_rounded, self.min_val, self.max_val)
        indices = values_clamped + self.offset
        return jax.vmap(self.rand_emb)(indices)


class ChargeSpinConditioner(eqx.Module):
    """Per-graph charge/spin embeddings scattered onto nodes (and/or edges).

    Port of the torch `ChargeSpinConditioner`. Reads ``total_charge`` and
    ``spin_multiplicity`` from ``batch.system_features`` (both shape ``(G,)``),
    embeds each, concatenates to a ``(G, latent_dim)`` per-graph code, then
    gathers it onto nodes/edges via the batch's precomputed graph-index arrays.
    """

    charge_embedding: ChargeSpinEmbedding
    spin_embedding: ChargeSpinEmbedding
    emits_node_embs: bool = eqx.field(static=True)
    emits_edge_embs: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        embedding_type: str = "sin_emb",
        emits_node_embs: bool = True,
        emits_edge_embs: bool = False,
        *,
        key: jax.Array | None = None,
    ):
        if key is None:
            charge_key, spin_key = None, None
        else:
            charge_key, spin_key = jax.random.split(key, 2)
        self.charge_embedding = ChargeSpinEmbedding(
            num_channels=latent_dim,
            embedding_target="charge",
            embedding_type=embedding_type,
            key=charge_key,
        )
        self.spin_embedding = ChargeSpinEmbedding(
            num_channels=latent_dim,
            embedding_target="spin",
            embedding_type=embedding_type,
            key=spin_key,
        )
        self.emits_node_embs = emits_node_embs
        self.emits_edge_embs = emits_edge_embs

    def __call__(
        self, batch: JaxAtomGraphs
    ) -> tuple[jax.Array | None, jax.Array | None]:
        charges = batch.system_features["total_charge"]
        spins = batch.system_features["spin_multiplicity"]

        charge_emb = self.charge_embedding(charges)
        spin_emb = self.spin_embedding(spins)
        combined_emb = jnp.concatenate([charge_emb, spin_emb], axis=-1)

        node_embs = (
            combined_emb[batch.per_node_graph_index]
            if self.emits_node_embs
            else None
        )
        edge_embs = (
            combined_emb[batch.per_edge_graph_index]
            if self.emits_edge_embs
            else None
        )
        return node_embs, edge_embs
