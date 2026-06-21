"""Validate the host-side PME prep (orb neighbour list -> jax-pme batch container)
against jax-pme's own vesin-driven `prepare`: matching energies confirm the
orb-neighbour reuse assembles the container correctly.
"""

import numpy as np
import pytest
from ase import Atoms

jaxpme = pytest.importorskip("jaxpme")
import jaxpme.batched_mixed  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from orb_models.forcefield.models.jax.coulomb_module import COULOMB_CONSTANT  # noqa: E402
from orb_models.forcefield.models.jax.pme import build_pme_batch, build_pme_structure  # noqa: E402

LR_WAVELENGTH = 1.0


def _systems():
    rng = np.random.default_rng(0)
    # two periodic water-ish clusters in different cubic boxes
    s1 = dict(
        positions=rng.standard_normal((3, 3)) * 0.8 + 6.0,
        cell=np.eye(3) * 12.0,
        charges=np.array([-0.8, 0.4, 0.4]),
    )
    s2 = dict(
        positions=rng.standard_normal((4, 3)) * 0.8 + 7.0,
        cell=np.eye(3) * 14.0,
        charges=np.array([0.5, 0.5, -0.5, -0.5]),
    )
    return [s1, s2]


def _energy_via_jaxpme_prepare(systems):
    """Reference: jax-pme's own vesin-backed batched prepare."""
    calc = jaxpme.batched_mixed.Ewald(prefactor=COULOMB_CONSTANT)
    atomss = [Atoms(positions=s["positions"], cell=s["cell"], pbc=True) for s in systems]
    cutoff = LR_WAVELENGTH * 8.0
    batches = calc.prepare(atomss, cutoff=cutoff, lr_wavelength=LR_WAVELENGTH)
    charges_padded, sr, nopbc, pbc = batches
    charges = np.zeros_like(np.asarray(charges_padded))
    offset = 0
    for s in systems:
        n = len(s["charges"])
        charges[offset : offset + n] = s["charges"]
        offset += n
    e = calc.energy(jnp.asarray(charges), sr, nopbc, pbc)
    return np.asarray(e), sr, nopbc, pbc


def _energy_via_orb_prep(systems, sizes):
    """Our path: orb neighbour list -> structures -> get_batch."""
    structures = [
        build_pme_structure(s["positions"], s["cell"], np.array([True, True, True]), LR_WAVELENGTH)
        for s in systems
    ]
    sr, nopbc, pbc = build_pme_batch(structures, **sizes)
    calc = jaxpme.batched_mixed.Ewald(prefactor=COULOMB_CONSTANT)
    # charges in the same (system-concatenated) order, padded to sr.atom_mask length
    charges = np.zeros(sr.atom_mask.shape[0])
    offset = 0
    for s in systems:
        n = len(s["charges"])
        charges[offset : offset + n] = s["charges"]
        offset += n
    e = calc.energy(jnp.asarray(charges), sr, nopbc, pbc)
    return np.asarray(e)


def test_orb_prep_matches_jaxpme_prepare():
    systems = _systems()
    ref_e, sr, nopbc, pbc = _energy_via_jaxpme_prepare(systems)

    # Match jax-pme's auto-padded sizes so the two containers are directly comparable.
    sizes = dict(
        num_structures=sr.cell.shape[0],
        num_atoms=sr.atom_mask.shape[0],
        num_pairs=sr.pair_mask.shape[0],
        num_pairs_nonpbc=nopbc.pair_mask.shape[0],
        num_structures_pbc=pbc.structure_mask.shape[0],
        num_atoms_pbc=pbc.atom_mask.shape[1],
        num_k=pbc.k_grid.shape[1],
    )
    ours_e = _energy_via_orb_prep(systems, sizes)

    # Per-structure energies (real structures only) must agree.
    np.testing.assert_allclose(ours_e[:2], ref_e[:2], rtol=1e-6, atol=1e-8)


def test_pme_energy_is_differentiable_wrt_charges():
    """dE/dq flows through the assembled batch (the charge-head training path)."""
    systems = _systems()
    structures = [
        build_pme_structure(s["positions"], s["cell"], np.array([True, True, True]), LR_WAVELENGTH)
        for s in systems
    ]
    sr, nopbc, pbc = build_pme_batch(structures)
    calc = jaxpme.batched_mixed.Ewald(prefactor=COULOMB_CONSTANT)
    n_pad = sr.atom_mask.shape[0]
    q = jnp.asarray(np.random.default_rng(2).standard_normal(n_pad))

    g = jax.grad(lambda c: calc.energy(c, sr, nopbc, pbc).sum())(q)
    assert np.isfinite(np.asarray(g)).all()
    assert np.abs(np.asarray(g)).sum() > 0
