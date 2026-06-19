"""End-to-end equivalence for the OrbMol-v1 configuration: orb-v3 conservative
backbone + system-level charge/spin conditioner (the omol inclusion set), with
real orb-v3 features (SH angular, outer-product+cutoff edge init, distance-cutoff
sigmoid attention, rms_norm). Shares every weight and checks energy/forces/stress
on a charged periodic 2-system batch.

This is the highest rung of the conditioner test pyramid: it confirms the
conditioning feeds correctly through the full energy -> forces/stress autograd.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.common.models import gns as torch_gns
from orb_models.common.models.angular import SphericalHarmonics as TorchSphericalHarmonics
from orb_models.common.models.jax.angular import SphericalHarmonics
from orb_models.common.models.jax.conditioner import ChargeSpinConditioner
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.common.models.jax.rbf import BesselBasis
from orb_models.common.models.nn_util import (
    ChargeSpinConditioner as TorchChargeSpinConditioner,
)
from orb_models.common.models.rbf import BesselBasis as TorchBesselBasis
from orb_models.forcefield.models.conservative_regressor import (
    ConservativeForcefieldRegressor,
)
from orb_models.forcefield.models.forcefield_heads import EnergyHead as TorchEnergyHead
from orb_models.forcefield.models.forcefield_utils import (
    torch_full_3x3_to_voigt_6_stress,
)
from orb_models.forcefield.models.jax.conservative_regressor import (
    ConservativeRegressor,
    compute_grads_reverse,
    predict,
    total_loss,
    trainable_filter,
)
from orb_models.forcefield.models.jax.forcefield_heads import EnergyHead
from orb_models.forcefield.models.jax.pair_repulsion import ZBLBasis

LATENT, STEPS, N_LAYERS, HIDDEN, NUM_BASES = 8, 2, 2, 16, 8
N_NODE = [3, 2]
SENDERS = [0, 1, 2, 0, 3, 4]
RECEIVERS = [1, 2, 0, 2, 4, 3]
N_EDGE = [4, 2]
TOTAL_CHARGE = [1.0, -1.0]  # a non-neutral system + an anion
SPIN_MULTIPLICITY = [2.0, 1.0]


def _arrays():
    rng = np.random.default_rng(0)
    n_node = np.array(N_NODE, dtype=np.int64)
    n_edge = np.array(N_EDGE, dtype=np.int64)
    N, G, E = int(n_node.sum()), len(n_node), int(n_edge.sum())
    return dict(
        n_node=n_node, n_edge=n_edge, N=N, G=G, E=E,
        positions=rng.standard_normal((N, 3)) * 1.5,
        cell=np.stack([np.eye(3) * 6.0 + 0.3 * rng.standard_normal((3, 3)) for _ in range(G)]),
        unit_shifts=rng.integers(-1, 2, size=(E, 3)).astype(np.float64),
        atomic_numbers=rng.integers(1, 30, size=(N,)).astype(np.int64),
    )


def _torch_graph(a):
    z = torch.tensor(a["atomic_numbers"])
    return AtomGraphs(
        senders=torch.tensor(SENDERS),
        receivers=torch.tensor(RECEIVERS),
        n_node=torch.tensor(a["n_node"]),
        n_edge=torch.tensor(a["n_edge"]),
        node_features={
            "positions": torch.tensor(a["positions"]),
            "atomic_numbers": z,
            "atomic_numbers_embedding": torch.nn.functional.one_hot(z - 1, num_classes=118).double(),
        },
        edge_features={
            "vectors": torch.zeros((a["E"], 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(a["unit_shifts"]),
        },
        system_features={
            "cell": torch.tensor(a["cell"]),
            "pbc": torch.ones((a["G"], 3), dtype=torch.bool),
            "total_charge": torch.tensor(TOTAL_CHARGE),
            "spin_multiplicity": torch.tensor(SPIN_MULTIPLICITY),
        },
        node_targets={}, edge_targets={}, system_targets={},
        system_id=None, fix_atoms=None, tags=None,
        radius=6.0, max_num_neighbors=torch.tensor([20]),
    )


def _build(key):
    common = dict(
        latent_dim=LATENT,
        num_message_passing_steps=STEPS,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        use_embedding=True,
        num_node_out_features=3,
        activation="silu",
        outer_product_with_cutoff=True,
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
        interaction_params={"distance_cutoff": True, "attention_gate": "sigmoid"},
        mlp_norm="rms_norm",
        conditioning_type="additive",
    )
    head_kw = dict(
        latent_dim=LATENT, num_mlp_layers=N_LAYERS, mlp_hidden_dim=HIDDEN,
        predict_atom_avg=True, activation="silu",
    )
    torch_model = ConservativeForcefieldRegressor(
        heads={"energy": TorchEnergyHead(**head_kw)},
        model=torch_gns.MoleculeGNS(
            rbf_transform=TorchBesselBasis(6.0, num_bases=NUM_BASES),
            angular_transform=TorchSphericalHarmonics(lmax=3, normalize=True, normalization="component"),
            conditioner=TorchChargeSpinConditioner(LATENT),
            **common,
        ),
        loss_weights={"energy": 1.0, "forces": 1.0, "stress": 1.0},
        pair_repulsion=True,
        has_stress=True,
    ).eval()
    jax_model = ConservativeRegressor(
        gns=MoleculeGNS(
            rbf_transform=BesselBasis(6.0, num_bases=NUM_BASES),
            angular_transform=SphericalHarmonics(lmax=3, normalize=True, normalization="component"),
            conditioner=ChargeSpinConditioner(LATENT, key=key),
            key=key,
            **common,
        ),
        energy_head=EnergyHead(key=key, **head_kw),
        pair_repulsion=ZBLBasis(p=6, node_aggregation="sum"),
    )
    return torch_model, jax_model


def test_orbmol_v1_regressor_matches_torch(helpers, key):
    torch_model, jax_model = _build(key)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)

    a = _arrays()
    torch_graph = _torch_graph(a)
    jax_graph = jgb.to_jax(torch_graph)  # before torch forward mutates the batch

    out = torch_model(torch_graph, fp64_energy=True)
    preds = predict(jax_graph, jax_model, has_stress=True)

    helpers.assert_close(preds.energy, out["interaction_energy"])
    jax_absolute = jax_model.energy_head.absolute_energy(preds.energy, jax_graph)
    helpers.assert_close(jax_absolute, out["energy"])
    helpers.assert_close(preds.forces, out["forces"])
    jax_stress_voigt = torch_full_3x3_to_voigt_6_stress(torch.tensor(np.asarray(preds.stress)))
    helpers.assert_close(jnp.asarray(jax_stress_voigt.numpy()), out["stress"])


def test_conditioner_params_are_trainable(key):
    """The conditioner's W params must be in the trainable partition (torch keeps
    them requires_grad=True) -- otherwise the optimiser would silently freeze the
    charge/spin conditioning and diverge from torch."""
    _, jax_model = _build(key)
    spec = trainable_filter(jax_model)
    assert spec.gns.conditioner.charge_embedding.W is True
    assert spec.gns.conditioner.spin_embedding.W is True


@pytest.mark.parametrize("padded", [False, True])
def test_orbmol_v1_training_grads(helpers, key, padded):
    """Second-order training path (grad of loss-through-forces) reaches the
    conditioner params with finite, nonzero grads -- on both an exact batch and a
    padded fixed-bucket batch (whose padding graph has charge=spin=0)."""
    torch_model, jax_model = _build(key)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)

    a = _arrays()
    rng = np.random.default_rng(7)
    if padded:
        n_pad, e_pad, g_pad = a["N"] + 6, a["E"] + 9, a["G"] + 2
        tg = _torch_graph(a)
        tg.system_targets["energy"] = torch.tensor(rng.standard_normal(a["G"]))
        tg.node_targets["forces"] = torch.tensor(rng.standard_normal((a["N"], 3)))
        tg.system_targets["stress"] = torch.tensor(rng.standard_normal((a["G"], 6)))
        graph_np, targets_np = jgb.to_padded_numpy(tg, n_pad, e_pad, g_pad, has_stress=True)
        jax_graph = jax.device_put(graph_np)
        targets = jax.device_put(targets_np)
    else:
        jax_graph = jgb.to_jax(_torch_graph(a))
        targets = {
            "energy": jnp.asarray(rng.standard_normal(a["G"])),
            "forces": jnp.asarray(rng.standard_normal((a["N"], 3))),
            "stress": jnp.asarray(rng.standard_normal((a["G"], 6))),
        }
    weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}

    loss, _ = total_loss(jax_model, jax_graph, targets, weights)
    assert np.isfinite(np.asarray(loss))

    grads, _ = compute_grads_reverse(jax_model, jax_graph, targets, weights)
    for emb in (grads.gns.conditioner.charge_embedding, grads.gns.conditioner.spin_embedding):
        g = np.asarray(emb.W)
        assert np.isfinite(g).all()
        assert np.abs(g).max() > 0  # real graphs drive a nonzero update


def test_orbmol_v1_conditioned_forward_jits(key):
    """The conditioned energy->forces/stress path jits (no data-dependent shapes):
    the conditioner gathers via per_node_graph_index instead of repeat_interleave,
    so a padded fixed-bucket batch compiles once. jit must match eager."""
    # Any weights suffice: jit==eager is a property of the graph, not the values.
    _, jax_model = _build(key)

    a = _arrays()
    n_pad, e_pad, g_pad = a["N"] + 6, a["E"] + 9, a["G"] + 2
    graph_np, _ = jgb.to_padded_numpy(_torch_graph(a), n_pad, e_pad, g_pad, has_stress=True)
    jax_graph = jax.device_put(graph_np)

    eager = predict(jax_graph, jax_model, has_stress=True)
    jitted = eqx.filter_jit(lambda g, m: predict(g, m, has_stress=True))(jax_graph, jax_model)

    for field in ("energy", "forces", "stress"):
        e_arr, j_arr = np.asarray(getattr(eager, field)), np.asarray(getattr(jitted, field))
        assert np.isfinite(e_arr).all()
        np.testing.assert_allclose(j_arr, e_arr, atol=1e-10)
