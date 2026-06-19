"""Shared fixtures/helpers for comparing the JAX modules to their PyTorch refs.

These tests run in float64 (both frameworks) so that any disagreement reflects
the maths rather than float32 GEMM-accumulation order. With identical weights
copied across, correct modules agree to ~1e-12; a real bug is far larger.
"""

import types

import jax

# Must be enabled before any array is created. Conftest is imported first, so
# this is in effect for every test module in this package.
jax.config.update("jax_enable_x64", True)
# Tiny fp64 equivalence tests: the GPU gives nothing here and JAX's default 75%
# VRAM preallocation collides with torch's CUDA context in-process (OOM). Pin CPU.
# (Test-only — does NOT affect real jobs, which import jax without this conftest.)
jax.config.update("jax_platform_name", "cpu")

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

# The torch->jax weight-copy helpers now live in the library (so real checkpoint
# loading can reuse them); re-exported through `helpers` for the equivalence tests.
from orb_models.forcefield.models.jax.port_weights import (
    copy_attention_network,
    copy_charge_spin_conditioner,
    copy_charge_spin_embedding,
    copy_conservative_regressor,
    copy_decoder,
    copy_encoder,
    copy_energy_head,
    copy_layer_norm,
    copy_linear,
    copy_mlp,
    copy_mlp_and_layer_norm,
    copy_molecule_gns,
    copy_scalar_normalizer,
)

# Tight tolerance is meaningful in float64 with shared weights.
ATOL = 1e-10


@pytest.fixture(autouse=True)
def _torch_float64():
    """Build torch refs in float64 for the duration of these tests."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# --- numeric comparison -----------------------------------------------------
def to_jax(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().numpy())


def assert_close(jax_out, torch_out, atol: float = ATOL):
    j = np.asarray(jax_out)
    t = torch_out.detach().numpy()
    assert j.shape == t.shape, f"shape mismatch: {j.shape} vs {t.shape}"
    max_err = np.abs(j - t).max()
    assert max_err <= atol, f"max_err={max_err:.2e} > atol={atol:.0e}"


@pytest.fixture
def helpers():
    """Namespace of comparison + weight-copy helpers used across test modules."""
    return types.SimpleNamespace(
        ATOL=ATOL,
        to_jax=to_jax,
        assert_close=assert_close,
        copy_linear=copy_linear,
        copy_layer_norm=copy_layer_norm,
        copy_mlp=copy_mlp,
        copy_mlp_and_layer_norm=copy_mlp_and_layer_norm,
        copy_encoder=copy_encoder,
        copy_attention_network=copy_attention_network,
        copy_decoder=copy_decoder,
        copy_molecule_gns=copy_molecule_gns,
        copy_charge_spin_embedding=copy_charge_spin_embedding,
        copy_charge_spin_conditioner=copy_charge_spin_conditioner,
        copy_energy_head=copy_energy_head,
        copy_scalar_normalizer=copy_scalar_normalizer,
        copy_conservative_regressor=copy_conservative_regressor,
    )
