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


# --- torch -> equinox weight copying ----------------------------------------
# TensorLinear / TensorLayerNorm subclass eqx.nn.Linear / LayerNorm, so weight
# and bias live at the same leaves; ordering of Linear layers matches torch.
def copy_linear(jax_linear, torch_linear):
    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, to_jax(torch_linear.weight))
    jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, to_jax(torch_linear.bias))
    return jax_linear


def copy_layer_norm(jax_ln, torch_ln):
    jax_ln = eqx.tree_at(lambda m: m.weight, jax_ln, to_jax(torch_ln.weight))
    jax_ln = eqx.tree_at(lambda m: m.bias, jax_ln, to_jax(torch_ln.bias))
    return jax_ln


def copy_mlp(jax_mlp, torch_seq):
    torch_linears = [m for m in torch_seq if isinstance(m, nn.Linear)]
    idxs = [i for i, layer in enumerate(jax_mlp.layers) if isinstance(layer, eqx.nn.Linear)]
    assert len(torch_linears) == len(idxs), "Linear count mismatch between jax and torch MLP"
    for i, tl in zip(idxs, torch_linears):
        jax_mlp = eqx.tree_at(lambda m, i=i: m.layers[i].weight, jax_mlp, to_jax(tl.weight))
        jax_mlp = eqx.tree_at(lambda m, i=i: m.layers[i].bias, jax_mlp, to_jax(tl.bias))
    return jax_mlp


def copy_mlp_and_layer_norm(jax_mln, torch_seq):
    """torch_seq is a Sequential with `.mlp` and `.layer_norm` (mlp_and_layer_norm)."""
    jax_mln = eqx.tree_at(lambda m: m.mlp, jax_mln, copy_mlp(jax_mln.mlp, torch_seq.mlp))
    jax_mln = eqx.tree_at(
        lambda m: m.layer_norm, jax_mln, copy_layer_norm(jax_mln.layer_norm, torch_seq.layer_norm)
    )
    return jax_mln


def copy_encoder(jax_enc, torch_enc):
    jax_enc = eqx.tree_at(
        lambda m: m.node_fn, jax_enc, copy_mlp_and_layer_norm(jax_enc.node_fn, torch_enc._node_fn)
    )
    jax_enc = eqx.tree_at(
        lambda m: m.edge_fn, jax_enc, copy_mlp_and_layer_norm(jax_enc.edge_fn, torch_enc._edge_fn)
    )
    return jax_enc


def copy_attention_network(jax_ain, torch_ain):
    jax_ain = eqx.tree_at(
        lambda m: m._node_mlp, jax_ain, copy_mlp_and_layer_norm(jax_ain._node_mlp, torch_ain._node_mlp)
    )
    jax_ain = eqx.tree_at(
        lambda m: m._edge_mlp, jax_ain, copy_mlp_and_layer_norm(jax_ain._edge_mlp, torch_ain._edge_mlp)
    )
    jax_ain = eqx.tree_at(
        lambda m: m._receive_attn, jax_ain, copy_linear(jax_ain._receive_attn, torch_ain._receive_attn)
    )
    jax_ain = eqx.tree_at(
        lambda m: m._send_attn, jax_ain, copy_linear(jax_ain._send_attn, torch_ain._send_attn)
    )
    if jax_ain._cond_node_proj is not None:
        jax_ain = eqx.tree_at(
            lambda m: m._cond_node_proj,
            jax_ain,
            copy_linear(jax_ain._cond_node_proj, torch_ain._cond_node_proj),
        )
    if jax_ain._cond_edge_proj is not None:
        jax_ain = eqx.tree_at(
            lambda m: m._cond_edge_proj,
            jax_ain,
            copy_linear(jax_ain._cond_edge_proj, torch_ain._cond_edge_proj),
        )
    return jax_ain


def copy_decoder(jax_dec, torch_dec):
    """torch Decoder wraps the mlp in `node_fn` (a Sequential[OrderedDict(mlp=...)])."""
    return eqx.tree_at(
        lambda m: m.mlp, jax_dec, copy_mlp(jax_dec.mlp, torch_dec.node_fn.mlp)
    )


def copy_molecule_gns(jax_model, torch_model):
    """Share every weight of a torch MoleculeGNS into its jax counterpart."""
    if jax_model.use_embedding:
        jax_model = eqx.tree_at(
            lambda m: m.atom_emb.embeddings.weight,
            jax_model,
            to_jax(torch_model.atom_emb.embeddings.weight),
        )
    jax_model = eqx.tree_at(
        lambda m: m._encoder, jax_model, copy_encoder(jax_model._encoder, torch_model._encoder)
    )
    for i in range(len(jax_model.gnn_stacks)):
        jax_model = eqx.tree_at(
            lambda m, i=i: m.gnn_stacks[i],
            jax_model,
            copy_attention_network(jax_model.gnn_stacks[i], torch_model.gnn_stacks[i]),
        )
    jax_model = eqx.tree_at(
        lambda m: m._decoder, jax_model, copy_decoder(jax_model._decoder, torch_model._decoder)
    )
    return jax_model


def copy_energy_head(jax_head, torch_head):
    """Share an EnergyHead's MLP + the fixed normalizer/reference buffers."""
    jax_head = eqx.tree_at(lambda m: m.mlp, jax_head, copy_mlp(jax_head.mlp, torch_head.mlp))
    jax_head = eqx.tree_at(
        lambda m: m.normalizer.mean, jax_head, to_jax(torch_head.normalizer.bn.running_mean)
    )
    jax_head = eqx.tree_at(
        lambda m: m.normalizer.std,
        jax_head,
        to_jax(torch.sqrt(torch_head.normalizer.bn.running_var)),
    )
    jax_head = eqx.tree_at(
        lambda m: m.reference.coefficients,
        jax_head,
        to_jax(torch_head.reference.linear.weight).reshape(-1),
    )
    return jax_head


def copy_scalar_normalizer(jax_norm, torch_norm):
    """Share a ScalarNormalizer's fixed mean/std buffers."""
    jax_norm = eqx.tree_at(lambda m: m.mean, jax_norm, to_jax(torch_norm.bn.running_mean))
    jax_norm = eqx.tree_at(
        lambda m: m.std, jax_norm, to_jax(torch.sqrt(torch_norm.bn.running_var))
    )
    return jax_norm


def copy_conservative_regressor(jax_model, torch_model):
    """Share a whole ConservativeRegressor: backbone + energy head + normalizers.

    ZBL pair repulsion has no trainable weights (fixed physical constants), so it
    needs no copying.
    """
    jax_model = eqx.tree_at(
        lambda m: m.gns, jax_model, copy_molecule_gns(jax_model.gns, torch_model.model)
    )
    jax_model = eqx.tree_at(
        lambda m: m.energy_head,
        jax_model,
        copy_energy_head(jax_model.energy_head, torch_model.heads["energy"]),
    )
    jax_model = eqx.tree_at(
        lambda m: m.grad_forces_normalizer,
        jax_model,
        copy_scalar_normalizer(jax_model.grad_forces_normalizer, torch_model.grad_forces_normalizer),
    )
    jax_model = eqx.tree_at(
        lambda m: m.grad_stress_normalizer,
        jax_model,
        copy_scalar_normalizer(jax_model.grad_stress_normalizer, torch_model.grad_stress_normalizer),
    )
    return jax_model


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
        copy_energy_head=copy_energy_head,
        copy_scalar_normalizer=copy_scalar_normalizer,
        copy_conservative_regressor=copy_conservative_regressor,
    )
