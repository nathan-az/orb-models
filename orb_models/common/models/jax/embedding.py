import equinox as eqx

import jax
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models.jax.nn_utils import tensor_apply


class AtomEmbedding(eqx.Module):
    embed_size: int = eqx.field(static=True)
    embeddings: eqx.nn.Embedding

    def __init__(self, emb_size, num_elements, *, key: jax.Array | None):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = eqx.nn.Embedding(num_elements + 1, emb_size)
        # init by uniform distribution
        if key:
            # if no key is passed, assume we have weights already (pretrained)
            jax.nn.initializers.uniform(scale=2*(3**0.5))(self.embeddings.weight, key=key)
            self.embeddings.weight -= 3**0.5

    @property
    def out_dim(self):
        """Size of the embedding."""
        return self.emb_size

    def __call__(self, batch: AtomGraphs):
        """
        Forward pass of the atom embedding layer.

        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        # NOTE: We can't use getters or setters here because torch.compile can't handle them.
        atomic_number_rep = batch.node_features["atomic_numbers"]
        h = tensor_apply(self.embeddings, atomic_number_rep, dims_exclude=0)
        return h
