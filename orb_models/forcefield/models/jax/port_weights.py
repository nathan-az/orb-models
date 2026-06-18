"""Port a trained torch orb model into the JAX `ConservativeRegressor`.

Two pieces:
  * `copy_*` -- share weights from a live torch module into the matching jax
    pytree (the same helpers the equivalence tests use; promoted here so real
    checkpoint loading is library code, not test-only).
  * `build_orb_v3_conservative_jax` -- construct a jax model whose architecture
    mirrors `pretrained.orb_v3_conservative_architecture`, ready to receive the
    copied weights.

What is and isn't copied: the Bessel `rbf_transform` (fixed, matches by
construction) and `SphericalHarmonics` (no params) need no copying; the energy
`reference`, ZBL constants and normalizer running-stats are copied as fixed
buffers. The torch `ConfidenceHead` (and any electrostatics/conditioner) has no
jax counterpart here and is simply ignored -- it does not affect energy/forces/
stress, which is all the conservative model differentiates.
"""

from __future__ import annotations

import equinox as eqx
import torch
from torch import nn

import jax.numpy as jnp

from orb_models.common.models.jax.angular import SphericalHarmonics
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.common.models.jax.rbf import BesselBasis
from orb_models.forcefield.models.jax.conservative_regressor import ConservativeRegressor
from orb_models.forcefield.models.jax.forcefield_heads import EnergyHead
from orb_models.forcefield.models.jax.pair_repulsion import ZBLBasis


def to_jax(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().cpu().numpy())


# --- torch -> equinox weight copying ----------------------------------------
def copy_linear(jax_linear, torch_linear):
    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, to_jax(torch_linear.weight))
    jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, to_jax(torch_linear.bias))
    return jax_linear


def copy_layer_norm(jax_ln, torch_ln):
    jax_ln = eqx.tree_at(lambda m: m.weight, jax_ln, to_jax(torch_ln.weight))
    # torch RMSNorm has weight only; LayerNorm has weight + bias.
    if getattr(torch_ln, "bias", None) is not None:
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
    return eqx.tree_at(lambda m: m.mlp, jax_dec, copy_mlp(jax_dec.mlp, torch_dec.node_fn.mlp))


def copy_molecule_gns(jax_model, torch_model):
    """Share every weight of a torch MoleculeGNS into its jax counterpart.

    rbf_transform / angular_transform are not copied: the Bessel weights match by
    construction and spherical harmonics have no parameters.
    """
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
    """Share a ScalarNormalizer's running mean/std AND batch count.

    The count matters for finetuning, not just inference: the online `update` uses
    BatchNorm's momentum=None cumulative average, factor = 1 / count. torch's
    `num_batches_tracked` is huge for a pretrained checkpoint (~2e6), so its factor
    is ~5e-7 and one finetuning mini-batch barely perturbs the calibrated stats.
    If we left the jax count at 0, the FIRST update would use factor = 1/1 = 1.0 and
    overwrite the pretrained mean/std wholesale with a single mini-batch -- which
    destabilises training (NaNs within a step or two). Copy it so the jax online
    update tracks torch's.
    """
    jax_norm = eqx.tree_at(lambda m: m.mean, jax_norm, to_jax(torch_norm.bn.running_mean))
    jax_norm = eqx.tree_at(
        lambda m: m.std, jax_norm, to_jax(torch.sqrt(torch_norm.bn.running_var))
    )
    jax_norm = eqx.tree_at(
        lambda m: m.count,
        jax_norm,
        to_jax(torch_norm.bn.num_batches_tracked).astype(jax_norm.count.dtype),
    )
    return jax_norm


def copy_conservative_regressor(jax_model, torch_model):
    """Share a whole ConservativeRegressor: backbone + energy head + normalizers.

    ZBL pair repulsion has no trainable weights (fixed physical constants), and any
    ConfidenceHead / electrostatics in the torch model are ignored.
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


# --- jax architecture mirroring pretrained.orb_v3_conservative_architecture ---
def build_orb_v3_conservative_jax(
    *,
    key,
    latent_dim: int = 256,
    base_mlp_hidden_dim: int = 1024,
    base_mlp_depth: int = 2,
    head_mlp_hidden_dim: int = 256,
    head_mlp_depth: int = 1,
    num_message_passing_steps: int = 5,
    activation: str = "silu",
    zbl_node_aggregation: str = "sum",
) -> ConservativeRegressor:
    """A jax `ConservativeRegressor` with the orb-v3 conservative (omat/mpa) arch.

    Matches `pretrained.orb_v3_conservative_architecture` for the released
    conservative checkpoints: no charge/spin conditioner, no electrostatics, no
    confidence head (none of which affect energy/forces/stress).

    `zbl_node_aggregation` must match the torch model: the released omat/mpa
    checkpoints are loaded with "mean" (a backward-compat quirk in pretrained.py),
    not the "sum" used by new code.
    """
    gns = MoleculeGNS(
        latent_dim=latent_dim,
        num_message_passing_steps=num_message_passing_steps,
        num_mlp_layers=base_mlp_depth,
        mlp_hidden_dim=base_mlp_hidden_dim,
        rbf_transform=BesselBasis(r_max=6.0, num_bases=8),
        angular_transform=SphericalHarmonics(lmax=3, normalize=True, normalization="component"),
        outer_product_with_cutoff=True,
        use_embedding=True,
        interaction_params={"distance_cutoff": True, "attention_gate": "sigmoid"},
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
        activation=activation,
        mlp_norm="rms_norm",
        key=key,
    )
    energy_head = EnergyHead(
        latent_dim=latent_dim,
        num_mlp_layers=head_mlp_depth,
        mlp_hidden_dim=head_mlp_hidden_dim,
        predict_atom_avg=True,
        activation=activation,
        key=key,
    )
    return ConservativeRegressor(
        gns=gns,
        energy_head=energy_head,
        pair_repulsion=ZBLBasis(p=6, node_aggregation=zbl_node_aggregation),
    )


def load_orb_v3_conservative_into_jax(torch_model, *, key) -> ConservativeRegressor:
    """Build the jax arch matching a torch `orb_v3_conservative_architecture` model
    and copy its weights across. `torch_model` should already have checkpoint
    weights loaded (e.g. via `pretrained.orb_v3_conservative_inf_omat`).

    ZBL aggregation is read off the torch model so the omat/mpa "mean" quirk is
    honoured automatically.
    """
    # torch MoleculeGNS doesn't expose latent_dim; read it off the energy head's
    # first Linear (in_features == latent_dim).
    head_linears = [m for m in torch_model.heads["energy"].mlp if isinstance(m, nn.Linear)]
    latent_dim = head_linears[0].in_features
    jax_model = build_orb_v3_conservative_jax(
        key=key,
        latent_dim=latent_dim,
        num_message_passing_steps=len(torch_model.model.gnn_stacks),
        zbl_node_aggregation=torch_model.pair_repulsion_fn.node_aggregation,
    )
    return copy_conservative_regressor(jax_model, torch_model)
