import equinox as eqx
import jax


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


def get_layer_norm(norm_type: str):
    norms = {"layer_norm": eqx.nn.LayerNorm}
    norm = norms.get(norm_type)
    if norm is None:
        raise ValueError(f"Unknown layer norm: {norm_type}")
    return norm


def tensor_linear(linear: eqx.nn.Linear, x: jax.Array) -> jax.Array:
    """Apply a linear layer to a tensor-shaped input."""
    fn = linear
    for _ in range(x.ndim-1):
        fn = jax.vmap(fn)
    return fn(x)


class TensorLinear(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_features: int, out_features: int, key):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return tensor_linear(self.linear, x)


class MLP(eqx.Module):
    layers: list[eqx.Module]

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: list[int],
        output_size: int,
        key,
        activation: str = "silu",
        output_activation: str = "identity",
        dropout: float = 0.0,
    ):
        layer_sizes = [input_size] + hidden_layer_sizes
        if output_size:
            layer_sizes.append(output_size)
        linear_keys = jax.random.split(key, len(layer_sizes) - 1)

        activations = [get_activation(activation) for _ in range(len(layer_sizes) - 1)]
        activations[-1] = get_activation(output_activation)

        layers = []
        for i in range(len(layer_sizes) - 1):
            if dropout is not None and dropout > 0.0:
                layers.append(eqx.nn.Dropout(dropout))
            layers.append(TensorLinear(layer_sizes[i], layer_sizes[i + 1], key=linear_keys[i]))
            layers.append(activations[i])
        self.layers = layers

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        n_dropout = sum(isinstance(layer, eqx.nn.Dropout) for layer in self.layers)
        if n_dropout > 0:
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
        key,
        activation: str = "silu",
        output_activation: str = "identity",
        norm_type: str = "layer_norm",
        dropout: float = 0.0,
    ):
        self.mlp = MLP(
            in_dim, [hidden_dim] * n_layers, out_dim, key, activation, output_activation, dropout
        )
        norm_fn = get_layer_norm(norm_type)
        self.layer_norm = norm_fn(out_dim)

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        x = self.mlp(x, key=key)
        x = self.layer_norm(x)
        return x
