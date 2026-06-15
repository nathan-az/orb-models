from collections.abc import Callable

import equinox as eqx

import jax
import jax.numpy as jnp


def get_activation(activation: str):
    act_fns = {
        "silu": jax.nn.silu,
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "identity": jax.nn.identity,
    }
    act_fn = act_fns.get(activation)
    if act_fn is None:
        raise ValueError(f"Unknown activation: {activation}")
    return act_fn


def tensor_apply(
    fn: Callable, x: jax.Array, dims_exclude=1, *call_args, **call_kwargs
) -> jax.Array:
    """Apply a function to a tensor-shaped input."""
    for _ in range(x.ndim - dims_exclude):
        fn = jax.vmap(fn)
    return fn(x, *call_args, **call_kwargs)


class TensorLinear(eqx.nn.Linear):
    def __call__(self, x: jax.Array) -> jax.Array:
        return tensor_apply(super().__call__, x)


class TensorLayerNorm(eqx.nn.LayerNorm):
    def __call__(self, x: jax.Array) -> jax.Array:
        return tensor_apply(super().__call__, x, len(self.shape))


def get_layer_norm(norm_type: str):
    norms = {"layer_norm": TensorLayerNorm}
    norm = norms.get(norm_type)
    if norm is None:
        raise ValueError(f"Unknown layer norm: {norm_type}")
    return norm


class MLP(eqx.Module):
    """In hindsight this is very similar to eqx.nn.MLP. Key difference is this
    incorporates dropout. Might be able to deprecate in future"""

    layers: list[eqx.Module]

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: list[int],
        output_size: int,
        activation: str = "silu",
        output_activation: str = "identity",
        dropout: float = 0.0,
        *,
        key,
    ):
        layer_sizes = [input_size] + hidden_layer_sizes
        if output_size:
            layer_sizes.append(output_size)
        linear_keys = jax.random.split(key, len(layer_sizes) - 1)

        activations = [
            get_activation(activation) for _ in range(len(layer_sizes) - 1)
        ]
        activations[-1] = get_activation(output_activation)

        layers = []
        for i in range(len(layer_sizes) - 1):
            if dropout is not None and dropout > 0.0:
                layers.append(eqx.nn.Dropout(dropout))
            layers.append(
                TensorLinear(
                    layer_sizes[i], layer_sizes[i + 1], key=linear_keys[i]
                )
            )
            layers.append(activations[i])
        self.layers = layers

    def __call__(
        self, x: jax.Array, *, key: jax.Array | None = None
    ) -> jax.Array:
        n_dropout = sum(
            isinstance(layer, eqx.nn.Dropout) for layer in self.layers
        )
        if n_dropout > 0:
            if key is None:
                raise ValueError(
                    "A key is required when the MLP contains dropout."
                )
            keys = jax.random.split(key, n_dropout)
            dropout_idx = 0
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x, key=keys[dropout_idx])
                dropout_idx += 1
            else:
                x = layer(x)
        return x


class MLPAndLayerNorm(eqx.Module):
    mlp: MLP
    layer_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "silu",
        output_activation: str = "identity",
        norm_type: str = "layer_norm",
        dropout: float = 0.0,
        *,
        key,
    ):
        self.mlp = MLP(
            in_dim,
            [hidden_dim] * n_layers,
            out_dim,
            activation,
            output_activation,
            dropout,
            key=key,
        )
        norm_fn = get_layer_norm(norm_type)
        self.layer_norm = norm_fn(out_dim)

    def __call__(
        self, x: jax.Array, *, key: jax.Array | None = None
    ) -> jax.Array:
        x = self.mlp(x, key=key)
        x = self.layer_norm(x)
        return x


def polynomial_cutoff(
    r: jax.Array, r_max: jax.Array | float, p: int = 4
) -> jax.Array:
    """Polynomial envelope going smoothly to 0 at r_max (and exactly 0 beyond).

    Shared by the attention cutoff (p=4, scalar r_max) and ZBL repulsion (p=6,
    per-edge r_max). Returns the same shape as `r`.
    """
    ratio = r / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * jnp.pow(ratio, p)
        + p * (p + 2.0) * jnp.pow(ratio, p + 1)
        - (p * (p + 1.0) / 2) * jnp.pow(ratio, p + 2)
    )
    return envelope * (r < r_max)


def get_cutoff_p4(r: jax.Array, r_max: float = 6.0) -> jax.Array:
    """Attention cutoff (p=4) with a trailing feature axis for broadcasting."""
    return jnp.expand_dims(polynomial_cutoff(r, r_max, p=4), axis=-1)
