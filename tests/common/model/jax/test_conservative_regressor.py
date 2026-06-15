"""End-to-end equivalence: jax ConservativeRegressor vs torch.

Builds a small matched conservative model (backbone + EnergyHead + ZBL, stress
on, no confidence/electrostatics -- the omat-style inclusion set), shares every
weight, and checks energy / forces / stress on a dummy periodic 2-system batch.

forces/stress are themselves gradients (dE/dpos, dE/ddisp), so this exercises the
full energy -> forces/stress autograd through the assembled model.
"""

import types

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.common.models import gns as torch_gns
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.common.models.jax.rbf import BesselBasis
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
    predict,
)
from orb_models.forcefield.models.jax.forcefield_heads import EnergyHead
from orb_models.forcefield.models.jax.pair_repulsion import ZBLBasis

LATENT, STEPS, N_LAYERS, HIDDEN, NUM_BASES = 8, 2, 2, 16, 8
N_NODE = [3, 2]
SENDERS = [0, 1, 2, 0, 3, 4]
RECEIVERS = [1, 2, 0, 2, 4, 3]
N_EDGE = [4, 2]


def _arrays():
    rng = np.random.default_rng(0)
    n_node = np.array(N_NODE, dtype=np.int64)
    n_edge = np.array(N_EDGE, dtype=np.int64)
    N, G, E = int(n_node.sum()), len(n_node), int(n_edge.sum())
    return dict(
        n_node=n_node,
        n_edge=n_edge,
        N=N,
        G=G,
        E=E,
        positions=rng.standard_normal((N, 3)) * 1.5,
        cell=np.stack([np.eye(3) * 6.0 + 0.3 * rng.standard_normal((3, 3)) for _ in range(G)]),
        unit_shifts=rng.integers(-1, 2, size=(E, 3)).astype(np.float64),
        # physical atomic numbers; one-hot at index Z-1 so torch's argmax+1 == Z
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
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                z - 1, num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((a["E"], 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(a["unit_shifts"]),
        },
        system_features={
            "cell": torch.tensor(a["cell"]),
            "pbc": torch.ones((a["G"], 3), dtype=torch.bool),
        },
        node_targets={},
        edge_targets={},
        system_targets={},
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
    )


def _build(key):
    gns_kw = dict(
        latent_dim=LATENT,
        num_message_passing_steps=STEPS,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        use_embedding=True,
        num_node_out_features=3,
        activation="silu",
    )
    head_kw = dict(
        latent_dim=LATENT,
        num_mlp_layers=N_LAYERS,
        mlp_hidden_dim=HIDDEN,
        predict_atom_avg=True,
        activation="silu",
    )
    torch_model = ConservativeForcefieldRegressor(
        heads={"energy": TorchEnergyHead(**head_kw)},
        model=torch_gns.MoleculeGNS(
            rbf_transform=TorchBesselBasis(6.0, num_bases=NUM_BASES), **gns_kw
        ),
        loss_weights={"energy": 1.0, "forces": 1.0, "stress": 1.0},
        pair_repulsion=True,
        has_stress=True,
    ).eval()
    jax_model = ConservativeRegressor(
        gns=MoleculeGNS(rbf_transform=BesselBasis(6.0, num_bases=NUM_BASES), key=key, **gns_kw),
        energy_head=EnergyHead(key=key, **head_kw),
        pair_repulsion=ZBLBasis(p=6, node_aggregation="sum"),
    )
    return torch_model, jax_model


def test_conservative_regressor_matches_torch(helpers, key):
    torch_model, jax_model = _build(key)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)

    a = _arrays()
    torch_graph = _torch_graph(a)
    jax_graph = jgb.to_jax(torch_graph)  # before torch forward mutates the batch

    out = torch_model(torch_graph, fp64_energy=True)
    preds = predict(jax_graph, jax_model, has_stress=True)

    # interaction energy (network + ZBL, no reference)
    helpers.assert_close(preds.energy, out["interaction_energy"])
    # absolute energy (+ reference)
    jax_absolute = jax_model.energy_head.absolute_energy(preds.energy, jax_graph)
    helpers.assert_close(jax_absolute, out["energy"])
    # conservative forces = -dE/dpos
    helpers.assert_close(preds.forces, out["forces"])
    # conservative stress: jax (G,3,3) -> Voigt-6 to match torch
    jax_stress_voigt = torch_full_3x3_to_voigt_6_stress(torch.tensor(np.asarray(preds.stress)))
    helpers.assert_close(jnp.asarray(jax_stress_voigt.numpy()), out["stress"])
