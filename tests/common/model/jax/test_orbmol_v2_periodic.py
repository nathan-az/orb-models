"""Phase 4: end-to-end OrbMol-v2 with the PERIODIC Coulomb branch wired into the
conservative regressor (jax-pme engine).

There is no fp64 torch-parity here: jax-pme is not bit-identical to the torch
nvalchemiops PME (different smearing/k-grid conventions), so we validate
INTEGRATION + internal consistency instead -- finite energy/forces/stress through
the full stack, the periodic Coulomb term actually contributing, jit==eager, and
the 2nd-order training grads reaching the latent charge head. Physical correctness
of the engine itself is anchored by test_periodic_coulomb / test_periodic_energy.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.common.models.jax.angular import SphericalHarmonics
from orb_models.common.models.jax.conditioner import ChargeSpinConditioner
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.common.models.jax.rbf import BesselBasis

pytest.importorskip("jaxpme")
import jaxpme.batched_mixed  # noqa: E402,F401

from orb_models.forcefield.models.jax.conservative_regressor import (  # noqa: E402
    ConservativeRegressor,
    compute_grads_reverse,
    predict,
    total_loss,
)
from orb_models.forcefield.models.jax.coulomb_module import CoulombModule  # noqa: E402
from orb_models.forcefield.models.jax.forcefield_heads import (  # noqa: E402
    ChargeConditionedEnergyHead,
    LatentChargeHead,
)
from orb_models.forcefield.models.jax.pair_repulsion import ZBLBasis  # noqa: E402
from orb_models.forcefield.models.jax.pme import build_pme_batch, build_pme_structure  # noqa: E402

LATENT, STEPS, N_LAYERS, HIDDEN, NUM_BASES = 8, 2, 2, 16, 8
N_NODE = [3, 2]
SENDERS = [0, 1, 2, 0, 3, 4]
RECEIVERS = [1, 2, 0, 2, 4, 3]
N_EDGE = [4, 2]
TOTAL_CHARGE = [0.0, 0.0]
SPIN_MULTIPLICITY = [1.0, 1.0]
LR = 1.0


def _arrays():
    rng = np.random.default_rng(0)
    n_node = np.array(N_NODE, dtype=np.int64)
    n_edge = np.array(N_EDGE, dtype=np.int64)
    N, G, E = int(n_node.sum()), len(n_node), int(n_edge.sum())
    # two periodic boxes, atoms near the centre
    positions = np.concatenate([
        rng.standard_normal((3, 3)) * 0.7 + 6.0,
        rng.standard_normal((2, 3)) * 0.7 + 7.0,
    ])
    cell = np.stack([np.eye(3) * 12.0, np.eye(3) * 14.0])
    return dict(
        n_node=n_node, n_edge=n_edge, N=N, G=G, E=E,
        positions=positions, cell=cell,
        unit_shifts=np.zeros((E, 3), dtype=np.float64),
        atomic_numbers=rng.integers(1, 30, size=(N,)).astype(np.int64),
    )


def _torch_graph(a):
    z = torch.tensor(a["atomic_numbers"])
    return AtomGraphs(
        senders=torch.tensor(SENDERS), receivers=torch.tensor(RECEIVERS),
        n_node=torch.tensor(a["n_node"]), n_edge=torch.tensor(a["n_edge"]),
        node_features={
            "positions": torch.tensor(a["positions"]), "atomic_numbers": z,
            "atomic_numbers_embedding": torch.nn.functional.one_hot(z - 1, num_classes=118).double(),
        },
        edge_features={
            "vectors": torch.zeros((a["E"], 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(a["unit_shifts"]),
        },
        system_features={
            "cell": torch.tensor(a["cell"]),
            "pbc": torch.ones((a["G"], 3), dtype=torch.bool),  # periodic
            "total_charge": torch.tensor(TOTAL_CHARGE),
            "spin_multiplicity": torch.tensor(SPIN_MULTIPLICITY),
        },
        node_targets={}, edge_targets={}, system_targets={},
        system_id=None, fix_atoms=None, tags=None,
        radius=6.0, max_num_neighbors=torch.tensor([20]),
    )


def _build_model(key):
    common = dict(
        latent_dim=LATENT, num_message_passing_steps=STEPS, num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN, use_embedding=True, num_node_out_features=3,
        activation="silu", outer_product_with_cutoff=True,
        node_feature_names=["feat"], edge_feature_names=["feat"],
        interaction_params={"distance_cutoff": True, "attention_gate": "sigmoid"},
        mlp_norm="rms_norm", conditioning_type="additive",
    )
    head_kw = dict(latent_dim=LATENT, num_mlp_layers=N_LAYERS, mlp_hidden_dim=HIDDEN, activation="silu")
    return ConservativeRegressor(
        gns=MoleculeGNS(
            rbf_transform=BesselBasis(6.0, num_bases=NUM_BASES),
            angular_transform=SphericalHarmonics(lmax=3, normalize=True, normalization="component"),
            conditioner=ChargeSpinConditioner(LATENT, key=key), key=key, **common,
        ),
        energy_head=ChargeConditionedEnergyHead(key=key, use_spins=False, **head_kw),
        pair_repulsion=ZBLBasis(p=6, node_aggregation="sum"),
        latent_charge_head=LatentChargeHead(key=key, enforce_total_charge=True,
                                            latent_dim=LATENT, num_mlp_layers=2,
                                            mlp_hidden_dim=HIDDEN, activation="silu"),
        coulomb_module=CoulombModule(),
    )


def _padded_periodic_graph(a):
    """Padded jax graph with PME prep attached, sized to the padded bucket."""
    n_pad, e_pad, g_pad = a["N"] + 4, a["E"] + 6, a["G"] + 1
    tg = _torch_graph(a)
    rng = np.random.default_rng(3)
    tg.system_targets["energy"] = torch.tensor(rng.standard_normal(a["G"]))
    tg.node_targets["forces"] = torch.tensor(rng.standard_normal((a["N"], 3)))
    tg.system_targets["stress"] = torch.tensor(rng.standard_normal((a["G"], 6)))
    graph_np, targets_np = jgb.to_padded_numpy(tg, n_pad, e_pad, g_pad, has_stress=True)
    graph = jax.device_put(graph_np)

    # Host-prep the PME batch sized to (n_pad atoms, g_pad structures).
    structures = [
        build_pme_structure(a["positions"][s:e], a["cell"][i], np.array([True, True, True]), LR)
        for i, (s, e) in enumerate([(0, 3), (3, 5)])
    ]
    pme_prep = build_pme_batch(structures, num_atoms=n_pad, num_structures=g_pad)
    graph = eqx.tree_at(lambda g: g.pme_prep, graph, pme_prep,
                        is_leaf=lambda x: x is None)
    return graph, jax.device_put(targets_np)


def test_periodic_predict_finite_and_coulomb_contributes(key):
    a = _arrays()
    model = _build_model(key)
    graph, _ = _padded_periodic_graph(a)

    preds = predict(graph, model, has_stress=True)
    for name in ("energy", "forces", "stress"):
        assert np.isfinite(np.asarray(getattr(preds, name))).all(), name

    # Removing the PME prep (no periodic Coulomb) must change the energy.
    graph_no_pme = eqx.tree_at(lambda g: g.pme_prep, graph, None)
    preds_no = predict(graph_no_pme, model, has_stress=True)
    assert not np.allclose(np.asarray(preds.energy), np.asarray(preds_no.energy)), (
        "periodic Coulomb term is not contributing to the energy"
    )


def test_periodic_jit_matches_eager(key):
    a = _arrays()
    model = _build_model(key)
    graph, _ = _padded_periodic_graph(a)

    eager = predict(graph, model, has_stress=True)
    jitted = eqx.filter_jit(lambda g, m: predict(g, m, has_stress=True))(graph, model)
    for name in ("energy", "forces", "stress"):
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, name)), np.asarray(getattr(eager, name)), atol=1e-8
        )


def test_periodic_training_grads_reach_charge_head(key):
    a = _arrays()
    model = _build_model(key)
    graph, targets = _padded_periodic_graph(a)
    weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}

    loss, _ = total_loss(model, graph, targets, weights)
    assert np.isfinite(np.asarray(loss))

    grads, _ = compute_grads_reverse(model, graph, targets, weights)
    last_linear = [
        layer for layer in grads.latent_charge_head.mlp.layers if hasattr(layer, "weight")
    ][-1]
    g = np.asarray(last_linear.weight)
    assert np.isfinite(g).all()
    assert np.abs(g).max() > 0
