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

    # separate variable required for jax grad calculation due to branch evaluation
    safe_denom = jnp.where(denominators == 0, 1.0, denominators)
    probs = jnp.where(denominators == 0, 0.0, exp / safe_denom)
    return probs
