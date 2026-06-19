import typing
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx

import jax
import jax.numpy as jnp
from orb_models.common.atoms.jax.graph_batch import JaxAtomGraphs
from orb_models.common.models.gns import ConditioningType
from orb_models.common.models.jax import segment_ops
from orb_models.common.models.jax.angular import StableNormalize
from orb_models.common.models.jax.embedding import AtomEmbedding
from orb_models.common.models.jax.nn_utils import (
    MLP,
    MLPAndLayerNorm,
    TensorLinear,
    get_cutoff_p4,
)


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
        checkpoint: str | None = None,
        activation: str = "silu",
        mlp_norm: str = "layer_norm",
        *,
        key,
    ):
        key_nodes, key_edges = jax.random.split(key, 2)
        self.node_fn = MLPAndLayerNorm(
            num_node_in_features,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            activation=activation,
            norm_type=mlp_norm,
            key=key_nodes,
        )
        self.edge_fn = MLPAndLayerNorm(
            num_edge_in_features,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            activation=activation,
            norm_type=mlp_norm,
            key=key_edges,
        )

    def __call__(self, node_features, edge_features):
        nodes = self.node_fn(node_features)
        edges = self.edge_fn(edge_features)
        return nodes, edges


class AttentionInteractionNetwork(eqx.Module):
    _node_mlp: MLPAndLayerNorm
    _edge_mlp: MLPAndLayerNorm
    _receive_attn: TensorLinear
    _send_attn: TensorLinear
    _cond_node_proj: TensorLinear | None
    _cond_edge_proj: TensorLinear | None
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
        attention_gate: Literal["sigmoid", "softmax"] = "sigmoid",
        conditioning: ConditioningType
        | tuple[ConditioningType, ConditioningType] = "none",
        distance_cutoff: bool = False,
        activation: str = "silu",
        mlp_norm: str = "layer_norm",
        dropout: float | None = None,
        *,
        key,
    ):
        if isinstance(conditioning, tuple):
            self._node_cond, self._edge_cond = conditioning
        else:
            self._node_cond, self._edge_cond = conditioning, conditioning

        assert self._node_cond in typing.get_args(ConditioningType)
        assert self._edge_cond in typing.get_args(ConditioningType)

        (
            node_key,
            edge_key,
            receive_attn_key,
            send_attn_key,
            cond_node_key,
            cond_edge_key,
        ) = jax.random.split(key, 6)

        node_mlp_cond_dim = (
            latent_dim if self._node_cond == "concatenative" else 0
        )
        edge_mlp_cond_dim = (
            latent_dim if self._edge_cond == "concatenative" else 0
        )

        self._node_mlp = MLPAndLayerNorm(
            latent_dim * 3 + node_mlp_cond_dim,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            activation=activation,
            norm_type=mlp_norm,
            dropout=dropout,
            key=node_key,
        )
        self._edge_mlp = MLPAndLayerNorm(
            latent_dim * 3 + edge_mlp_cond_dim + 2 * node_mlp_cond_dim,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            activation=activation,
            norm_type=mlp_norm,
            dropout=dropout,
            key=edge_key,
        )
        self._receive_attn = TensorLinear(
            latent_dim + edge_mlp_cond_dim, 1, key=receive_attn_key
        )
        self._send_attn = TensorLinear(
            latent_dim + edge_mlp_cond_dim, 1, key=send_attn_key
        )

        if self._node_cond != "none":
            self._cond_node_proj = TensorLinear(
                latent_dim, latent_dim, key=cond_node_key
            )
        else:
            self._cond_node_proj = None
        if self._edge_cond != "none":
            self._cond_edge_proj = TensorLinear(
                latent_dim, latent_dim, key=cond_edge_key
            )
        else:
            self._cond_edge_proj = None

        self._distance_cutoff = distance_cutoff
        self._attention_gate = attention_gate
        self.latent_dim = latent_dim

    def __call__(
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
            nodes = jnp.concatenate(
                [nodes, self._cond_node_proj(cond_nodes)], axis=-1
            )

        if self._edge_cond == "additive":
            if cond_edges is not None:
                edges = edges + self._cond_edge_proj(cond_edges)
        elif self._edge_cond == "concatenative" and cond_edges is not None:
            edges = jnp.concatenate(
                [edges, self._cond_edge_proj(cond_edges)], axis=-1
            )

        if self._attention_gate == "softmax":
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
        edge_features = jnp.concatenate(
            [edges, sent_attributes, received_attributes], axis=-1
        )
        updated_edges = self._edge_mlp(edge_features)

        sent_attributes = jax.ops.segment_sum(
            updated_edges * send_attn, senders, nodes.shape[0]
        )
        received_attributes = jax.ops.segment_sum(
            updated_edges * receive_attn, receivers, nodes.shape[0]
        )

        node_features = jnp.concatenate(
            [nodes, received_attributes, sent_attributes], axis=-1
        )
        updated_nodes = self._node_mlp(node_features)

        if self._node_cond == "concatenative":
            nodes = nodes[:, : self.latent_dim]
        if self._edge_cond == "concatenative":
            edges = edges[:, : self.latent_dim]

        nodes = nodes + updated_nodes
        edges = edges + updated_edges

        return nodes, edges


class Decoder(eqx.Module):
    mlp: MLP

    def __init__(
        self,
        num_node_in: int,
        num_node_out: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        activation: str = "silu",
        *,
        key,
    ):
        self.mlp = MLP(
            num_node_in,
            [mlp_hidden_dim] * num_mlp_layers,
            num_node_out,
            activation=activation,
            key=key,
        )

    def __call__(
        self, x: jax.Array, *, key: jax.Array | None = None
    ) -> jax.Array:
        return self.mlp(x, key=key)


class MoleculeGNS(eqx.Module):
    node_feature_names: list[str] = eqx.field(static=True)
    edge_feature_names: list[str] = eqx.field(static=True)
    num_message_passing_steps: int = eqx.field(static=True)
    outer_product_with_cutoff: bool = eqx.field(static=True)
    rbf_transform: eqx.Module | Callable
    angular_transform: eqx.Module | Callable
    edge_embed_size: int = eqx.field(static=True)
    use_embedding: bool = eqx.field(static=True)
    node_embed_size: int = eqx.field(static=True)
    atom_emb: AtomEmbedding | None
    conditioner: Callable | None
    _encoder: Encoder
    gnn_stacks: list[AttentionInteractionNetwork]
    _decoder: Decoder

    def __init__(
        self,
        latent_dim: int,
        num_message_passing_steps: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        rbf_transform: Callable,
        angular_transform: Callable | None = None,
        outer_product_with_cutoff: bool = False,
        use_embedding: bool = False,  # atom type embedding
        interaction_params: dict[str, Any] | None = None,
        num_node_out_features: int = 3,
        extra_embed_dims: int | tuple[int, int] = 0,
        node_feature_names: list[str] | None = None,
        edge_feature_names: list[str] | None = None,
        conditioner: Callable | None = None,
        conditioning_type: ConditioningType = "additive",
        activation="ssp",
        mlp_norm: str = "layer_norm",
        *,
        key,
    ):
        key_embeddings, key_encoder, key_gnn_stacks, key_decoder = (
            jax.random.split(key, 4)
        )

        self.node_feature_names = node_feature_names or []
        self.edge_feature_names = edge_feature_names or []
        self.num_message_passing_steps = num_message_passing_steps

        # Edge embedding
        self.outer_product_with_cutoff = outer_product_with_cutoff
        self.rbf_transform = rbf_transform
        if angular_transform is None:
            angular_transform = StableNormalize()
        self.angular_transform = angular_transform

        if self.outer_product_with_cutoff:
            self.edge_embed_size = (
                rbf_transform.num_bases * angular_transform.dim
            )
        else:
            if hasattr(rbf_transform, "num_bases"):
                num_bases = rbf_transform.num_bases
            else:
                num_bases = rbf_transform.keywords["num_bases"]
            self.edge_embed_size = num_bases + angular_transform.dim

        self.use_embedding = use_embedding
        if self.use_embedding:
            self.node_embed_size = latent_dim
            self.atom_emb = AtomEmbedding(
                self.node_embed_size, 118, key=key_embeddings
            )
        else:
            self.node_embed_size = 118
            self.atom_emb = None

        if isinstance(extra_embed_dims, int):
            extra_embed_dims = (extra_embed_dims, extra_embed_dims)

        # Conditioning
        if conditioner is not None:
            node_conditioning = (
                conditioning_type if conditioner.emits_node_embs else "none"
            )  # type: ignore
            edge_conditioning = (
                conditioning_type if conditioner.emits_edge_embs else "none"
            )  # type: ignore
            self.conditioner: Callable | None = conditioner
        else:
            node_conditioning, edge_conditioning = "none", "none"
            self.conditioner = None

        self._encoder = Encoder(
            num_node_in_features=self.node_embed_size + extra_embed_dims[0],
            num_edge_in_features=self.edge_embed_size + extra_embed_dims[1],
            latent_dim=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            activation=activation,
            mlp_norm=mlp_norm,
            key=key_encoder,
        )
        gnn_keys = jax.random.split(
            key_gnn_stacks, self.num_message_passing_steps
        )
        self.gnn_stacks = [
            AttentionInteractionNetwork(
                latent_dim=latent_dim,
                num_mlp_layers=num_mlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                conditioning=(node_conditioning, edge_conditioning),
                **(interaction_params or {}),
                activation=activation,
                mlp_norm=mlp_norm,
                key=gnn_keys[i],
            )
            for i in range(self.num_message_passing_steps)
        ]
        self._decoder = Decoder(
            num_node_in=latent_dim,
            num_node_out=num_node_out_features,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            activation=activation,
            key=key_decoder,
        )

    def __call__(self, batch: JaxAtomGraphs):
        edge_features = self.featurize_edges(batch)
        node_features = self.featurize_nodes(batch)
        if self.conditioner is not None:
            cond_nodes, cond_edges = self.conditioner(batch)
        else:
            cond_nodes, cond_edges = None, None

        nodes, edges = self._encoder(node_features, edge_features)

        cutoff = get_cutoff_p4(jnp.linalg.norm(batch.edge_features["vectors"], axis=-1))
        for gnn in self.gnn_stacks:
            nodes, edges = gnn(
                nodes,
                edges,
                batch.senders,
                batch.receivers,
                cutoff,
                cond_nodes=cond_nodes,
                cond_edges=cond_edges,
            )
        pred = self._decoder(nodes)
        return {"node_features": nodes, "edge_features": edges, "pred": pred}

    def featurize_nodes(self, batch: JaxAtomGraphs):
        if self.use_embedding:
            atomic_embedding = self.atom_emb(batch)
        else:
            atomic_embedding = batch.node_features["atomic_numbers_embedding"]

        feature_names = [k for k in self.node_feature_names if k != "feat"]
        return jnp.concatenate(
            [
                atomic_embedding,
                *[batch.node_features[k] for k in feature_names],
            ],
            axis=-1,
        )

    def featurize_edges(self, batch: JaxAtomGraphs):
        vectors = batch.edge_features["vectors"]
        lengths = jnp.linalg.norm(vectors, axis=-1)

        angular_embedding = self.angular_transform(vectors)
        rbfs = self.rbf_transform(lengths)

        if self.outer_product_with_cutoff:
            cutoff = get_cutoff_p4(lengths)
            outer_product = rbfs[:, :, None] * angular_embedding[:, None, :]
            edge_features = cutoff * outer_product.reshape(
                vectors.shape[0], self.edge_embed_size
            )
        else:
            edge_features = jnp.concatenate([rbfs, angular_embedding], axis=-1)

        feature_names = [k for k in self.edge_feature_names if k != "feat"]
        return jnp.concatenate(
            [edge_features, *[batch.edge_features[k] for k in feature_names]],
            axis=-1,
        )
