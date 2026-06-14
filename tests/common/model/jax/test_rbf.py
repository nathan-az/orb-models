"""Compare jax BesselBasis to the PyTorch reference."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.models.rbf import BesselBasis as TorchBesselBasis
from orb_models.common.models.jax.rbf import BesselBasis


@pytest.mark.parametrize("num_bases", [4, 8])
def test_bessel_basis_matches_torch(helpers, num_bases):
    r_max = 6.0
    torch_b = TorchBesselBasis(r_max, num_bases=num_bases)
    jax_b = BesselBasis(r_max, num_bases=num_bases)

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, r_max, size=(12,))  # >0 to avoid the 1/x singularity
    helpers.assert_close(jax_b(jnp.asarray(x)), torch_b(torch.tensor(x)))


def test_bessel_basis_num_bases_attr():
    """num_bases is exposed (gns.py reads rbf_transform.num_bases)."""
    assert BesselBasis(6.0, num_bases=8).num_bases == 8
