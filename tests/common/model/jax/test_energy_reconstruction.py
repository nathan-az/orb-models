"""Energy-per-atom torch<->jax equivalence for the fp64 reference reconstruction.

The network predicts only the small (~eV) interaction energy; the absolute energy is
`interaction + per-element reference`, and at OMol scale the reference is ~1e3-1e5 eV.
Adding it back in fp32 rounds to the reference's ~meV grid (the ~1.4 meV/atom fp32
floor); `reconstruct_absolute_energy` does the add in host fp64, recovering the torch
fp64 absolute energy/atom. Gated `integration` (uses the real OMol checkpoint).

Because the conftest force-enables x64, an fp32 forward gets promoted back to fp64, so
we instead reconstruct from a single fp64 interaction in explicit numpy fp32 vs fp64.
"""

import numpy as np
import pytest

import jax

from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.jax.conservative_regressor import (
    predict,
    reconstruct_absolute_energy,
)
from orb_models.forcefield.models.jax.port_weights import (
    load_orb_v3_conservative_into_jax,
)


def _water_cluster(nx=5, ny=5, nz=4, spacing=3.1):
    """`nx*ny*nz` water molecules in a padded periodic box (~300 atoms)."""
    from ase import Atoms
    from ase.build import molecule

    w = molecule("H2O")
    atoms = Atoms()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shifted = w.copy()
                shifted.translate(np.array([i, j, k]) * spacing)
                atoms += shifted
    atoms.set_cell(np.array([nx, ny, nz]) * spacing + 6.0)
    atoms.set_pbc(True)
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0
    return atoms


@pytest.fixture(scope="module")
def omol_models():
    """Real OMol conservative checkpoint (download+convert): torch fp64 + jax port."""
    import torch

    from orb_models.forcefield import pretrained

    torch_model, adapter = pretrained.orb_v3_conservative_omol(device="cpu", compile=False)
    torch_model = torch_model.double().eval()
    jax_model = load_orb_v3_conservative_into_jax(torch_model, key=jax.random.PRNGKey(0))
    coeffs_f64 = np.asarray(jax_model.energy_head.reference.coefficients, dtype=np.float64)
    return torch_model, adapter, jax_model, coeffs_f64


@pytest.mark.integration
def test_fp64_reconstruction_matches_torch_at_omol_scale(omol_models):
    """jax fp64 reconstruction recovers torch's fp64 abs energy/atom; the fp32
    reference-add carries a floor that the host fp64 add removes."""
    torch_model, adapter, jax_model, coeffs_f64 = omol_models

    # A few differently-sized clusters: with several graphs the MAX per-atom error
    # reliably sits near the fp32 floor (a single graph can round near zero by luck).
    atoms_list = [_water_cluster(nx, 5, 4) for nx in (5, 6, 7)]
    graphs = [adapter.from_ase_atoms(a, device="cpu") for a in atoms_list]
    graph = type(graphs[0]).batch(graphs)
    n_atoms = np.asarray(graph.n_node)  # (G,)
    jax_graph = jgb.to_jax(graph)  # snapshot before torch mutates the batch

    # torch fp64 reference 'truth' (per graph).
    out = torch_model(graph, fp64_energy=True)
    e_ref = np.asarray(out["energy"].detach().reshape(-1))
    assert np.all(np.abs(e_ref / n_atoms) > 1e2), "OMol references should dominate"

    # jax interaction prediction (fp64 under x64; matches torch's fp64 interaction by
    # parity). Reconstruct the absolute energy from the SAME interaction two ways:
    #   OLD = the add in fp32 (pre-fix production behaviour);
    #   NEW = `reconstruct_absolute_energy` (the host fp64 add).
    interaction = np.asarray(predict(jax_graph, jax_model, has_stress=False).energy)
    Z = np.asarray(jax_graph.node_features["atomic_numbers"]).astype(np.int64)
    pgi = np.asarray(jax_graph.per_node_graph_index).astype(np.int64)
    ref_pg_f64 = np.bincount(pgi, weights=coeffs_f64[Z], minlength=jax_graph.n_node.shape[0])

    e_new = np.asarray(reconstruct_absolute_energy(interaction, jax_graph, coeffs_f64))
    e_old = (interaction.astype(np.float32) + ref_pg_f64.astype(np.float32)).astype(np.float32)

    err_new = np.abs(e_new - e_ref) / n_atoms
    err_old = np.abs(e_old - e_ref) / n_atoms

    print(f"\n[{n_atoms.tolist()} atoms]  |ref| ~ {abs(e_ref[0]/n_atoms[0]):.0f} eV/atom")
    print(f"  jax NEW (fp64 add) max |dE|/atom = {err_new.max():.2e} eV   <- matches torch")
    print(f"  jax OLD (fp32 add) max |dE|/atom = {err_old.max():.2e} eV   <- fp32 floor")
    print(f"  floor removed: OLD/NEW = {err_old.max() / max(err_new.max(), 1e-30):.0f}x")

    assert err_new.max() < 1e-6, f"jax fp64 reconstruction != torch: {err_new.max():.2e}"
    assert err_old.max() > 1e-5, f"expected an fp32 floor, got {err_old.max():.2e}"
    assert err_old.max() / err_new.max() > 100, (
        f"fp64 reconstruction did not remove the floor: "
        f"OLD={err_old.max():.2e} NEW={err_new.max():.2e}"
    )


@pytest.mark.integration
def test_training_interaction_target_is_fp64_precise(omol_models):
    """The collator's host fp64 `raw - reference` recovers the small interaction
    TARGET precisely at OMol scale (real references), whereas the naive fp32
    subtraction (an in-graph `raw_f32 - reference_f32` after device_put) loses ~meV."""
    import torch

    _, adapter, _, coeffs_f64 = omol_models
    graph = adapter.from_ase_atoms(_water_cluster(5, 5, 4), device="cpu")
    n = graph.node_features["positions"].shape[0]
    e = graph.senders.shape[0]
    G = graph.n_node.shape[0]

    # Known small (~eV) interaction + the REAL OMol per-element reference baseline,
    # so the absolute energy label is `reference + interaction` in fp64.
    Z = graph.node_features["atomic_numbers"].numpy().astype(np.int64)
    pgi = graph.node_batch_index.numpy().astype(np.int64)
    ref_pg = np.bincount(pgi, weights=coeffs_f64[Z], minlength=G)  # fp64, OMol scale
    rng = np.random.default_rng(0)
    true_interaction = rng.standard_normal(G) * 5.0  # ~eV
    abs_energy = ref_pg + true_interaction  # fp64 absolute label

    graph.system_targets["energy"] = torch.tensor(abs_energy, dtype=torch.float64)
    graph.node_targets["forces"] = torch.zeros((n, 3), dtype=torch.float64)

    _, targets = jgb.to_padded_numpy(
        graph, n + 8, e + 16, G + 2, has_stress=False, reference_coefficients=coeffs_f64
    )
    host = targets["interaction_energy"][:G].astype(np.float64)  # collator fp64 path
    naive = (abs_energy.astype(np.float32) - ref_pg.astype(np.float32)).astype(np.float32)

    err_host = np.abs(host - true_interaction).max()
    err_naive = np.abs(naive - true_interaction).max()
    print(f"\n  |ref| ~ {abs(ref_pg[0]/graph.n_node[0].item()):.0f} eV/atom")
    print(f"  host fp64 target  max |dE| = {err_host:.2e} eV")
    print(f"  naive fp32 target max |dE| = {err_naive:.2e} eV  ({err_naive / err_host:.0f}x)")

    assert err_host < 1e-3, f"host fp64 target lost the interaction: {err_host:.2e}"
    assert err_naive > 100 * err_host, f"naive fp32 unexpectedly precise: {err_naive:.2e}"
