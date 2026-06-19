"""Phase 2: the differentiable periodic-energy wrapper.

`periodic_coulomb_energy` substitutes LIVE positions/cell into the host-prepped
jax-pme batch and returns per-structure energy. We check that (a) it reproduces
jax-pme's own energy, (b) `jax.grad` forces match jax-pme's analytic forces, (c)
the strain-derivative matches jax-pme's analytic stress, and (d) it jits. This is
the engine our CoulombModule periodic branch differentiates through.
"""

import numpy as np
import pytest
from ase import Atoms

jaxpme = pytest.importorskip("jaxpme")
import jaxpme.batched_mixed  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from orb_models.forcefield.models.jax.pme import (  # noqa: E402
    COULOMB_CONSTANT,
    build_pme_batch,
    build_pme_structure,
    periodic_coulomb_energy,
)

LR = 1.0


def _prep():
    rng = np.random.default_rng(0)
    systems = [
        dict(positions=rng.standard_normal((3, 3)) * 0.8 + 6.0, cell=np.eye(3) * 12.0,
             charges=np.array([-0.8, 0.4, 0.4])),
        dict(positions=rng.standard_normal((4, 3)) * 0.8 + 7.0, cell=np.eye(3) * 14.0,
             charges=np.array([0.5, 0.5, -0.5, -0.5])),
    ]
    structures = [
        build_pme_structure(s["positions"], s["cell"], np.array([True, True, True]), LR)
        for s in systems
    ]
    sr, nopbc, pbc = build_pme_batch(structures)
    n_pad = sr.atom_mask.shape[0]
    charges = np.zeros(n_pad)
    offset = 0
    for s in systems:
        n = len(s["charges"])
        charges[offset : offset + n] = s["charges"]
        offset += n
    return systems, sr, nopbc, pbc, jnp.asarray(charges)


def test_energy_identity_with_live_positions():
    """Feeding back the prepped positions/cell reproduces jax-pme's own energy."""
    _, sr, nopbc, pbc, charges = _prep()
    calc = jaxpme.batched_mixed.Ewald(prefactor=COULOMB_CONSTANT)
    ref = calc.energy(charges, sr, nopbc, pbc)
    ours = periodic_coulomb_energy(charges, sr.positions, sr.cell, sr, nopbc, pbc)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), atol=1e-12)


def test_autodiff_forces_match_jaxpme_analytic():
    _, sr, nopbc, pbc, charges = _prep()
    calc = jaxpme.batched_mixed.Ewald(prefactor=COULOMB_CONSTANT)
    _, ref_forces, _ = calc.energy_forces_stress(charges, sr, nopbc, pbc)

    forces = -jax.grad(
        lambda p: periodic_coulomb_energy(charges, p, sr.cell, sr, nopbc, pbc).sum()
    )(sr.positions)
    np.testing.assert_allclose(np.asarray(forces), np.asarray(ref_forces), atol=1e-8)


def test_autodiff_stress_matches_jaxpme_analytic():
    """Global symmetric-strain derivative equals the summed per-structure virial."""
    _, sr, nopbc, pbc, charges = _prep()
    calc = jaxpme.batched_mixed.Ewald(prefactor=COULOMB_CONSTANT)
    _, _, ref_stress = calc.energy_forces_stress(charges, sr, nopbc, pbc)
    ref_total = np.asarray(ref_stress).sum(axis=0)  # (3,3)

    def energy_of_strain(strain):
        deform = jnp.eye(3) + strain
        pos = sr.positions @ deform.T
        cell = sr.cell @ deform.T
        return periodic_coulomb_energy(charges, pos, cell, sr, nopbc, pbc).sum()

    virial = jax.grad(energy_of_strain)(jnp.zeros((3, 3)))
    np.testing.assert_allclose(np.asarray(virial), ref_total, atol=1e-7)


def test_jit_and_dEdq():
    _, sr, nopbc, pbc, charges = _prep()
    fn = jax.jit(lambda q, p, c: periodic_coulomb_energy(q, p, c, sr, nopbc, pbc))
    eager = periodic_coulomb_energy(charges, sr.positions, sr.cell, sr, nopbc, pbc)
    jitted = fn(charges, sr.positions, sr.cell)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-10)

    g = jax.grad(lambda q: periodic_coulomb_energy(q, sr.positions, sr.cell, sr, nopbc, pbc).sum())(charges)
    assert np.isfinite(np.asarray(g)).all()
    assert np.abs(np.asarray(g)).sum() > 0
