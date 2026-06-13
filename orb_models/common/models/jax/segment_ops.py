import jax
import jax.numpy as jnp


def segment_softmax(
    data: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    weights: jax.Array | None = None,
):
    data_max = jax.ops.segment_max(data, segment_ids, num_segments)
    data = data - data_max[segment_ids]
    exp = jnp.exp(data)
    if weights is not None:
        exp = exp * weights
    expsums = jax.ops.segment_sum(exp, segment_ids, num_segments)
    denominators = expsums[segment_ids]
    probs = jnp.where(denominators == 0, 0, exp / denominators)
    return probs
