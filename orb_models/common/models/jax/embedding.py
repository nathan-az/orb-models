import equinox as eqx

import jax
import jax.numpy as jnp
from orb_models.common.atoms.jax.graph_batch import JaxAtomGraphs
from orb_models.common.models.jax.nn_utils import tensor_apply


class AtomEmbedding(eqx.Module):
    """Initial atom embeddings based on atom type.

    The embedding *table* is float (``dtype``); the *indices* (atomic numbers)
    must be integers, mirroring the torch ref's ``.long()`` on the lookup.
    """

    embed_size: int = eqx.field(static=True)
    embeddings: eqx.nn.Embedding

    def __init__(
        self,
        emb_size: int,
        num_elements: int,
        *,
        key: jax.Array | None,
    ):
        self.embed_size = emb_size
        shape = (num_elements + 1, emb_size)
        if key is not None:
            # init by uniform distribution, matching torch nn.init.uniform_(-sqrt(3), sqrt(3))
            weight = jax.random.uniform(
                key, shape, minval=-(3**0.5), maxval=3**0.5
            )
        else:
            # no key -> assume weights are loaded afterwards (e.g. pretrained); a
            # valid eqx.nn.Embedding still needs a concrete table, so use zeros.
            weight = jnp.zeros(shape)
        self.embeddings = eqx.nn.Embedding(weight=weight)

    @property
    def out_dim(self):
        """Size of the embedding."""
        return self.embed_size

    def __call__(self, batch: JaxAtomGraphs) -> jax.Array:
        """Atom embeddings, shape (nAtoms, emb_size)."""
        # eqx.nn.Embedding takes a scalar index, so map over the atom axis.
        # Cast to int for the gather (the torch ref does this with .long()).
        atomic_number_rep = batch.node_features["atomic_numbers"].astype(jnp.int32)
        return tensor_apply(self.embeddings, atomic_number_rep, dims_exclude=0)
