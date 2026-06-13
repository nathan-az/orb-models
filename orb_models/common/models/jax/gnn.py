import typing
from typing import Literal

import equinox as eqx

import jax
import jax.numpy as jnp
from orb_models.common.models.gns import ConditioningType
from orb_models.common.models.jax import segment_ops
from orb_models.common.models.jax.nn_utils import MLPAndLayerNorm


class Encoder(eqx.Module):
    node_fn: MLPAndLayerNorm
    edge_fn: MLPAndLayerNorm

    def __init__(
        self,
        num_node_in_features: int,
        num_edge_in_features: int,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        key,
        checkpoint: str | None = None,
        activation: str = "silu",
        mlp_norm: str = "layer_norm",
    ):
        key_nodes, key_edges = jax.random.split(key, 2)
        self.node_fn = MLPAndLayerNorm(
            num_node_in_features,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            key_nodes,
            activation=activation,
            mlp_norm=mlp_norm,
        )
        self.edge_fn = MLPAndLayerNorm(
            num_edge_in_features,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            key_edges,
            activation=activation,
            mlp_norm=mlp_norm,
        )

    def __call__(self, node_features, edge_features):
        nodes = self.node_fn(node_features)
        edges = self.edge_fn(edge_features)
        return nodes, edges


class AttentionInteractionNetwork(eqx.Module):
    _node_mlp: MLPAndLayerNorm
    _edge_mlp: MLPAndLayerNorm
    _receive_attn: eqx.nn.Linear
    _send_attn: eqx.nn.Linear
    _cond_node_proj: eqx.nn.Linear | None
    _cond_edge_proj: eqx.nn.Linear | None
    _attention_gate: str = eqx.field(static=True)
    _distance_cutoff: bool = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    _node_cond: str = eqx.field(static=True)
    _edge_cond: str = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        key,
        attention_gate: Literal["sigmoid", "softmax"] = "sigmoid",
        conditioning: ConditioningType | tuple[ConditioningType, ConditioningType] = "none",
        distance_cutoff: bool = False,
        activation: str = "ssp",
        mlp_norm: str = "layer_norm",
        dropout: float | None = None,
    ):
        if isinstance(conditioning, tuple):
            self._node_cond, self._edge_cond = conditioning
        else:
            self._node_cond, self._edge_cond = conditioning, conditioning

        assert self._node_cond in typing.get_args(ConditioningType)
        assert self._edge_cond in typing.get_args(ConditioningType)

        num_keys = 4
        if self._node_cond != "none":
            num_keys += 1
        if self._edge_cond != "none":
            num_keys += 1

        keys = jax.random.split(key, num_keys)

        node_mlp_cond_dim = latent_dim if self._node_cond == "concatenative" else 0
        edge_mlp_cond_dim = latent_dim if self._edge_cond == "concatenative" else 0

        self._node_mlp = MLPAndLayerNorm(
            latent_dim * 3 + node_mlp_cond_dim,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            keys[0],
            activation=activation,
            norm_type=mlp_norm,
            dropout=dropout,
        )
        self._edge_mlp = MLPAndLayerNorm(
            latent_dim * 3 + edge_mlp_cond_dim + 2 * node_mlp_cond_dim,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            keys[1],
            activation=activation,
            norm_type=mlp_norm,
            dropout=dropout,
        )
        self._receive_attn = eqx.nn.Linear(latent_dim + edge_mlp_cond_dim, 1, key=keys[2])
        self._send_attn = eqx.nn.Linear(latent_dim + edge_mlp_cond_dim, 1, key=keys[3])

        if self._node_cond != "none":
            self._cond_node_proj = eqx.nn.Linear(latent_dim, latent_dim, key=keys[4])
        if self._edge_cond != "none":
            self._cond_edge_proj = eqx.nn.Linear(latent_dim, latent_dim, key=keys[5])

        self._distance_cutoff = distance_cutoff
        self._attention_gate = attention_gate
        self.latent_dim = latent_dim

    def forward(
        self,
        nodes: jax.Array,
        edges: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        cutoff: jax.Array,
        cond_nodes: jax.Array | None = None,
        cond_edges: jax.Array | None = None,
    ):
        if self._node_cond == "additive":
            if cond_nodes is not None:
                nodes = nodes + self._cond_node_proj(cond_nodes)
        elif self._node_cond == "concatenative" and cond_nodes is not None:
            nodes = jnp.concatenate([nodes, self._cond_node_proj(cond_nodes)], axis=-1)

        if self._edge_cond == "additive":
            if cond_edges is not None:
                edges = edges + self._cond_edge_proj(cond_edges)
        elif self._edge_cond == "concatenative" and cond_edges is not None:
            edges = jnp.concatenate([edges, self._cond_edge_proj(cond_edges)], axis=-1)

        if self._edge_cond == "softmax":
            num_segments = nodes.shape[0]
            receive_attn = segment_ops.segment_softmax(
                self._receive_attn(edges),
                receivers,
                num_segments,
                weights=cutoff if self._distance_cutoff else None,
            )
            send_attn = segment_ops.segment_softmax(
                self._send_attn(edges),
                senders,
                num_segments,
                weights=cutoff if self._distance_cutoff else None,
            )
        else:
            receive_attn = jax.nn.sigmoid(self._receive_attn(edges))
            send_attn = jax.nn.sigmoid(self._send_attn(edges))

        if self._distance_cutoff:
            receive_attn = receive_attn * cutoff
            send_attn = send_attn * cutoff

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]
        edge_features = jnp.concatenate([edges, sent_attributes, received_attributes], axis=-1)
        updated_edges = self._edge_mlp(edge_features)

        sent_attributes = jax.ops.segment_sum(
            updated_edges * send_attn, senders, nodes.shape[0]
        )
        received_attributes = jax.ops.segment_sum(
            updated_edges * receive_attn, receivers, nodes.shape[0]
        )

        node_features = jnp.concatenate([nodes, received_attributes, sent_attributes], axis=-1)
        updated_nodes = self._node_mlp(node_features)

        if self._node_cond == "concatenative":
            nodes = nodes[:, : self.latent_dim]
        if self._edge_cond == "concatenative":
            edges = edges[:, : self.latent_dim]

        nodes = nodes + updated_nodes
        edges = edges + updated_edges

        return nodes, edges
        