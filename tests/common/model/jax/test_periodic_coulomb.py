"""Pin down `jax-pme` as a trustworthy periodic Coulomb engine.

These do NOT require bit-parity with the torch nvalchemiops PME (a deferred concern);
they validate the engine's physics:
  (a) it reproduces the analytic NaCl Madelung constant;
  (b) its non-periodic Ewald == our non-periodic `CoulombModule` (both the bare 1/r
      all-pairs sum x COULOMB_CONSTANT) -- the oracle the rest of the port reuses;
  (c) its periodic energy converges to the isolated-molecule energy as the box grows.
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.jax.coulomb_module import COULOMB_CONSTANT, CoulombModule

jaxpme = pytest.importorskip("jaxpme")
import jax.numpy as jnp  # noqa: E402


def _nonperiodic_jax_graph(positions, n_node):
    positions = np.asarray(positions, dtype=np.float64)
    n_node = np.asarray(n_node, dtype=np.int64)
    N, G = positions.shape[0], n_node.shape[0]
    z = np.ones(N, dtype=np.int64)
    tg = AtomGraphs(
        senders=torch.zeros(0, dtype=torch.long),
        receivers=torch.zeros(0, dtype=torch.long),
        n_node=torch.tensor(n_node),
        n_edge=torch.zeros(G, dtype=torch.long),
        node_features={
            "positions": torch.tensor(positions),
            "atomic_numbers": torch.tensor(z),
            "atomic_numbers_embedding": torch.nn.functional.one_hot(
                torch.tensor(z - 1), num_classes=118
            ).double(),
        },
        edge_features={
            "vectors": torch.zeros((0, 3), dtype=torch.float64),
            "unit_shifts": torch.zeros((0, 3), dtype=torch.float64),
        },
        system_features={
            "cell": torch.zeros((G, 3, 3), dtype=torch.float64),
            "pbc": torch.zeros((G, 3), dtype=torch.bool),
        },
        node_targets={}, edge_targets={}, system_targets={},
        system_id=None, fix_atoms=None, tags=None,
        radius=6.0, max_num_neighbors=torch.tensor([20]),
    )
    return jgb.to_jax(tg)


def test_jaxpme_reproduces_nacl_madelung():
    """Sanity: the engine gets the textbook Madelung constant right."""
    pos = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float
    )
    charges = jnp.array([1.0, -1, -1, -1, 1, 1, 1, -1])
    atoms = Atoms(positions=pos, cell=2 * np.eye(3), pbc=True)
    calc = jaxpme.Ewald()  # unit prefactor -> raw Madelung
    sr = 2.0
    smearing = sr / 5.0
    inputs = calc.prepare(atoms, charges, sr, 0.5 * smearing, smearing)
    energy = float(calc.energy(*inputs))
    madelung = -energy / 4  # 4 formula units
    np.testing.assert_allclose(madelung, 1.747565, rtol=1e-5)


def _jaxpme_nonperiodic_energy(positions, charges):
    """jax-pme non-periodic Ewald (bare 1/r all-pairs) in orb's eV.A units."""
    atoms = Atoms(positions=np.asarray(positions), cell=np.eye(3), pbc=False)
    calc = jaxpme.Ewald(prefactor=COULOMB_CONSTANT)
    # Non-periodic path ignores cutoff/smearing (bare sum), but prepare needs values.
    inputs = calc.prepare(atoms, jnp.asarray(charges), 1.0, 0.5, 0.2)
    return float(calc.energy(*inputs))


def test_jaxpme_nonperiodic_equals_our_direct_sum():
    """jax-pme(pbc=False) == our non-periodic CoulombModule (sigma=None): both are
    the same bare 1/r all-pairs sum, so this anchors jax-pme to the torch-validated
    convention we already match."""
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((4, 3)) * 2.0
    charges = rng.standard_normal(4)
    charges = charges - charges.mean()  # neutral

    jg = _nonperiodic_jax_graph(positions, [4])
    ours = float(CoulombModule(direct_coulomb_erf_damping_sigma=None)(
        jnp.asarray(charges)[:, None], jg
    )[0])
    theirs = _jaxpme_nonperiodic_energy(positions, charges)
    np.testing.assert_allclose(theirs, ours, rtol=1e-6)


def test_periodic_converges_to_isolated_energy():
    """jax-pme periodic energy -> isolated-molecule energy as the box grows
    (monotonic error decrease), mirroring torch test_vacuum_gap_convergence."""
    rng = np.random.default_rng(1)
    positions = rng.standard_normal((3, 3)) * 0.8 + 5.0  # near box centre
    charges = rng.standard_normal(3)
    charges = charges - charges.mean()

    iso = _jaxpme_nonperiodic_energy(positions, charges)

    errors = []
    for box in [12.0, 20.0, 35.0, 60.0]:
        atoms = Atoms(positions=positions, cell=box * np.eye(3), pbc=True)
        calc = jaxpme.Ewald(prefactor=COULOMB_CONSTANT)
        smearing = box / 12.0
        inputs = calc.prepare(atoms, jnp.asarray(charges), box / 4, 0.5 * smearing, smearing)
        e = float(calc.energy(*inputs))
        errors.append(abs(e - iso))

    assert errors == sorted(errors, reverse=True), f"not converging: {errors}"
    assert errors[-1] < 1e-2, f"largest box still far from isolated: {errors[-1]}"
