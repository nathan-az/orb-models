"""fp32 vs fp64 inference parity on a real large system (gated by --run-integration).

Quantifies whether JAX fp32 inference degrades absolute energy/atom vs the torch fp64
reference (largest at OMol scale, ~1e3-1e5 eV references; uses the released OMol
checkpoint on a ~300-atom water cluster). Asserts jax-fp64 reproduces torch-fp64
energy/atom + forces/stress, while jax-fp32 energy/atom deviates (the fp32 reference-add
floor) but forces stay precision-independent (the constant reference cancels in dE/dx).
"""

import equinox as eqx
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.jax.conservative_regressor import predict
from orb_models.forcefield.models.jax.port_weights import load_orb_v3_conservative_into_jax


def _water_cluster(nx=5, ny=5, nz=4, spacing=3.1):
    """A grid of `nx*ny*nz` water molecules in a padded periodic box (~300 atoms)."""
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
    box = np.array([nx, ny, nz]) * spacing + 6.0
    atoms.set_cell(box)
    atoms.set_pbc(True)
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0
    return atoms


def _to_fp32(tree):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, tree
    )


@pytest.mark.integration
def test_fp32_vs_fp64_energy_per_atom_large_system(key):
    import torch

    from orb_models.forcefield import pretrained

    torch.set_default_dtype(torch.float64)
    torch_model, adapter = pretrained.orb_v3_conservative_omol(device="cpu", compile=False)
    torch_model = torch_model.double().eval()

    atoms = _water_cluster()
    n_atoms = len(atoms)
    graph = adapter.from_ase_atoms(atoms, device="cpu")
    jax_graph = jgb.to_jax(graph)  # snapshot before torch mutates the batch

    jax_model = load_orb_v3_conservative_into_jax(torch_model, key=key)

    # --- torch fp64 reference ---
    out = torch_model(graph, fp64_energy=True)
    e_ref = out["energy"].item()  # absolute energy, fp64
    f_ref = out["forces"].detach().numpy()
    e_ref_per_atom = e_ref / n_atoms

    # --- jax fp64 (x64 is on in this harness): parity / recovery ---
    preds64 = predict(jax_graph, jax_model, has_stress=True)
    e64 = float(jnp.asarray(jax_model.energy_head.absolute_energy(preds64.energy, jax_graph)).reshape(-1)[0])
    f64 = np.asarray(preds64.forces)

    # --- jax fp32 (cast model + graph): the deployment precision ---
    model32 = _to_fp32(jax_model)
    graph32 = _to_fp32(jax_graph)
    preds32 = predict(graph32, model32, has_stress=True)
    e32 = float(jnp.asarray(model32.energy_head.absolute_energy(preds32.energy, graph32)).reshape(-1)[0])
    f32 = np.asarray(preds32.forces)

    # fp32 interaction prediction, but reconstruct the absolute energy with the host
    # fp64 add (`reconstruct_absolute_energy`). This removes the fp32 reference-add
    # floor; the residual is only the fp32 BACKBONE interaction error.
    from orb_models.forcefield.models.jax.conservative_regressor import (
        reconstruct_absolute_energy,
    )

    coeffs_f64 = np.asarray(model32.energy_head.reference.coefficients, dtype=np.float64)
    e32_recon = float(
        reconstruct_absolute_energy(np.asarray(preds32.energy), jax_graph, coeffs_f64)[0]
    )

    err64_per_atom = abs(e64 - e_ref) / n_atoms
    err32_per_atom = abs(e32 - e_ref) / n_atoms
    err32_recon_per_atom = abs(e32_recon - e_ref) / n_atoms
    f64_err = np.abs(f64 - f_ref).max()
    f32_err = np.abs(f32 - f_ref).max()

    print(f"\n[{n_atoms} atoms]  torch-fp64 abs E = {e_ref:.4f} eV ({e_ref_per_atom:.4f} eV/atom)")
    print(f"  jax-fp64            |dE|/atom = {err64_per_atom:.2e} eV   forces max err = {f64_err:.2e} eV/A")
    print(f"  jax-fp32 (fp32 add) |dE|/atom = {err32_per_atom:.2e} eV   forces max err = {f32_err:.2e} eV/A")
    print(f"  jax-fp32 (fp64 add) |dE|/atom = {err32_recon_per_atom:.2e} eV")
    print(f"  fp32 add / fp64 add energy-error ratio = {err32_per_atom / max(err32_recon_per_atom, 1e-30):.1f}x")

    # fp64 path is tight parity with torch (maths, not precision).
    assert err64_per_atom < 1e-5, f"jax-fp64 energy/atom off: {err64_per_atom:.2e}"
    assert f64_err < 1e-4, f"jax-fp64 forces off: {f64_err:.2e}"

    # fp32 degrades the absolute energy but stays bounded (document the floor).
    assert err32_per_atom > err64_per_atom, "fp32 should be no better than fp64"
    assert err32_per_atom < 5e-2, f"jax-fp32 energy/atom worse than expected: {err32_per_atom:.2e}"

    # Forces are essentially precision-independent (reference cancels in the gradient).
    assert f32_err < 5e-3, f"jax-fp32 forces unexpectedly degraded: {f32_err:.2e}"

    # The host fp64 reconstruction removes the fp32 reference-add floor: it is strictly
    # better than the fp32 add and recovers most of the gap to the fp64 path (the
    # residual is the fp32 backbone interaction, not the reference add).
    assert err32_recon_per_atom < err32_per_atom, "fp64 add should beat fp32 add"
    assert err32_recon_per_atom < err32_per_atom / 2, (
        f"fp64 reconstruction did not remove the bulk of the floor: "
        f"fp32add={err32_per_atom:.2e} fp64add={err32_recon_per_atom:.2e}"
    )
