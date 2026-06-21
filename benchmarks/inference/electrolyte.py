"""Scalable aqueous NaCl electrolyte builder for the MD benchmark.

A periodic cubic water box at ~liquid density with a few Na+/Cl- pairs swapped in.
The geometry is grid-packed + rattled, NOT equilibrated -- good enough to give a
realistic neighbour count (hence realistic edge/PME-pair budgets) for benchmarking
*compute*, which is weight- and exact-geometry-independent.

Size is set by `n_side`: an n_side**3 grid of water molecules.
  n_side=3 ->   27 H2O ~   81 atoms      n_side=6 ->  216 H2O ~  648 atoms
  n_side=4 ->   64 H2O ~  192 atoms      n_side=7 ->  343 H2O ~ 1029 atoms
  n_side=5 ->  125 H2O ~  375 atoms      n_side=8 ->  512 H2O ~ 1536 atoms
(ion swaps drop the 2 H of a water, so atom counts dip slightly below 3*n_side**3.)
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

# Liquid water: ~33.4 molecules / nm^3 -> ~3.106 Angstrom per molecule on a cubic grid.
_SPACING = 3.106  # Angstrom between grid sites
_OH = 0.957  # O-H bond length, Angstrom
_HOH = np.deg2rad(104.5)  # H-O-H angle


def _water(center: np.ndarray, rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """One randomly-oriented water molecule (O + 2H) about `center`."""
    half = _HOH / 2.0
    local = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [_OH * np.sin(half), _OH * np.cos(half), 0.0],  # H
            [-_OH * np.sin(half), _OH * np.cos(half), 0.0],  # H
        ]
    )
    # random rotation via a uniform quaternion
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    pos = local @ rot.T + center
    return ["O", "H", "H"], pos


def build_electrolyte(
    n_side: int = 4,
    ion_fraction: float = 0.06,
    rattle: float = 0.1,
    seed: int = 1,
) -> Atoms:
    """Periodic cubic NaCl(aq) box; `n_side**3` water sites, a fraction swapped to ions.

    `ion_fraction` is the fraction of sites turned into ions; rounded down to an even
    number so the box stays charge-neutral (equal Na+ and Cl-).
    """
    rng = np.random.default_rng(seed)
    box = n_side * _SPACING
    grid = (np.arange(n_side) + 0.5) * _SPACING

    sites = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    n_sites = len(sites)

    n_ion = int(n_sites * ion_fraction)
    n_pair = n_ion // 2  # Na/Cl pairs -> neutral
    ion_idx = set(rng.choice(n_sites, size=2 * n_pair, replace=False).tolist())
    na_idx = set(list(ion_idx)[:n_pair])  # first half Na, rest Cl

    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for i, site in enumerate(sites):
        if i in ion_idx:
            symbols.append("Na" if i in na_idx else "Cl")
            positions.append(site)
        else:
            syms, pos = _water(site, rng)
            symbols.extend(syms)
            positions.extend(pos)

    atoms = Atoms(symbols=symbols, positions=np.array(positions))
    atoms.set_cell([box, box, box])
    atoms.set_pbc(True)
    atoms.wrap()
    if rattle:
        atoms.rattle(rattle, seed=seed + 1)
    return atoms
