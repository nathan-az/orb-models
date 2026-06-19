"""Integration coverage for `load_orb_v3_conservative_into_jax`.

Three rungs:
  * `test_loader_infers_dims_matches_torch` (fast, default) -- a *non-default* small
    architecture, to prove the loader infers every shape-bearing hyperparameter
    (latent/base/head dims + depths, activation, conditioner presence) rather than
    assuming the released defaults. Parametrized over has_charge_spin_cond.
  * `test_loader_real_scale_matches_torch` (integration) -- the full released
    orb-v3 conservative dims (latent 256, hidden 1024, 5 steps), random weights.
  * `test_loader_real_omol_checkpoint` (integration) -- downloads the real
    OrbMol-v1 (orb-v3-conservative-omol) checkpoint and checks energy/forces parity
    on a real charged molecule.

The two integration tests only run with `--run-integration` (real-scale models /
network download); the dim-inference test runs in the normal unit suite.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.forcefield_utils import (
    torch_full_3x3_to_voigt_6_stress,
)
from orb_models.forcefield.models.jax.conservative_regressor import predict
from orb_models.forcefield.models.jax.port_weights import (
    load_orb_v3_conservative_into_jax,
)
from orb_models.forcefield.pretrained import orb_v3_conservative_architecture

N_NODE = [3, 2]
SENDERS = [0, 1, 2, 0, 3, 4]
RECEIVERS = [1, 2, 0, 2, 4, 3]
N_EDGE = [4, 2]


def _torch_graph():
    rng = np.random.default_rng(0)
    n_node = np.array(N_NODE, dtype=np.int64)
    n_edge = np.array(N_EDGE, dtype=np.int64)
    N, G, E = int(n_node.sum()), len(n_node), int(n_edge.sum())
    z = torch.tensor(rng.integers(1, 30, size=(N,)).astype(np.int64))
    return AtomGraphs(
        senders=torch.tensor(SENDERS),
        receivers=torch.tensor(RECEIVERS),
        n_node=torch.tensor(n_node),
        n_edge=torch.tensor(n_edge),
        node_features={
            "positions": torch.tensor(rng.standard_normal((N, 3)) * 1.5),
            "atomic_numbers": z,
            "atomic_numbers_embedding": torch.nn.functional.one_hot(z - 1, num_classes=118).double(),
        },
        edge_features={
            "vectors": torch.zeros((E, 3), dtype=torch.float64),
            "unit_shifts": torch.tensor(rng.integers(-1, 2, size=(E, 3)).astype(np.float64)),
        },
        system_features={
            "cell": torch.tensor(np.stack([np.eye(3) * 6.0 for _ in range(G)])),
            "pbc": torch.ones((G, 3), dtype=torch.bool),
            "total_charge": torch.tensor([1.0, -1.0]),
            "spin_multiplicity": torch.tensor([2.0, 1.0]),
        },
        node_targets={}, edge_targets={}, system_targets={},
        system_id=None, fix_atoms=None, tags=None,
        radius=6.0, max_num_neighbors=torch.tensor([20]),
    )


def _assert_loader_parity(helpers, torch_model, jax_model, torch_graph):
    jax_graph = jgb.to_jax(torch_graph)  # before torch forward mutates the batch
    out = torch_model(torch_graph, fp64_energy=True)
    preds = predict(jax_graph, jax_model, has_stress=True)

    jax_absolute = jax_model.energy_head.absolute_energy(preds.energy, jax_graph)
    helpers.assert_close(jax_absolute, out["energy"])
    helpers.assert_close(preds.forces, out["forces"])
    jax_stress_voigt = torch_full_3x3_to_voigt_6_stress(torch.tensor(np.asarray(preds.stress)))
    helpers.assert_close(jnp.asarray(jax_stress_voigt.numpy()), out["stress"])


@pytest.mark.parametrize("has_charge_spin_cond", [False, True])
def test_loader_infers_dims_matches_torch(helpers, key, has_charge_spin_cond):
    """Non-default dims: the loader must read latent/hidden/depth/activation off the
    torch model. (A loader that assumed the released defaults would trip the
    copy_mlp Linear-count assertion here.)"""
    torch_model = orb_v3_conservative_architecture(
        latent_dim=8,
        base_mlp_hidden_dim=16,
        base_mlp_depth=2,
        head_mlp_hidden_dim=12,
        head_mlp_depth=3,  # != the released default of 1
        num_message_passing_steps=2,
        has_charge_spin_cond=has_charge_spin_cond,
        has_stress=True,
        device="cpu",
    ).eval()
    assert (torch_model.model.conditioner is not None) == has_charge_spin_cond

    jax_model = load_orb_v3_conservative_into_jax(torch_model, key=key)
    assert (jax_model.gns.conditioner is not None) == has_charge_spin_cond
    # The inferred head depth (4 linears -> depth 3) round-trips into the jax head.
    assert len([m for m in torch_model.heads["energy"].mlp if isinstance(m, torch.nn.Linear)]) == 4

    _assert_loader_parity(helpers, torch_model, jax_model, _torch_graph())


@pytest.mark.integration
@pytest.mark.parametrize("has_charge_spin_cond", [False, True])
def test_loader_real_scale_matches_torch(helpers, key, has_charge_spin_cond):
    """Full released orb-v3 conservative dims (latent 256, hidden 1024, 5 steps)."""
    torch_model = orb_v3_conservative_architecture(
        has_charge_spin_cond=has_charge_spin_cond, has_stress=True, device="cpu"
    ).eval()
    jax_model = load_orb_v3_conservative_into_jax(torch_model, key=key)
    _assert_loader_parity(helpers, torch_model, jax_model, _torch_graph())


@pytest.mark.integration
def test_loader_real_omol_checkpoint(key):
    """Download the real OrbMol-v1 (orb-v3-conservative-omol) checkpoint, load it
    into jax, and check energy/forces parity on a charged molecule. Weights are
    upcast to fp64 on both sides so the comparison reflects the maths, not fp32
    GEMM order."""
    from ase.build import molecule

    from orb_models.forcefield import pretrained

    torch_model, adapter = pretrained.orb_v3_conservative_omol(device="cpu", compile=False)
    torch_model = torch_model.double().eval()
    assert torch_model.model.conditioner is not None  # OrbMol uses charge/spin cond

    atoms = molecule("H2O")
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc(True)
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0  # singlet multiplicity
    graph = adapter.from_ase_atoms(atoms, device="cpu")
    jax_graph = jgb.to_jax(graph)

    jax_model = load_orb_v3_conservative_into_jax(torch_model, key=key)
    out = torch_model(graph, fp64_energy=True)
    preds = predict(jax_graph, jax_model, has_stress=True)

    jax_absolute = np.asarray(jax_model.energy_head.absolute_energy(preds.energy, jax_graph))
    np.testing.assert_allclose(jax_absolute, out["energy"].detach().numpy(), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(preds.forces), out["forces"].detach().numpy(), rtol=1e-5, atol=1e-4
    )
