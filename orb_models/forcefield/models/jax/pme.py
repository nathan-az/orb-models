"""Host-side prep for the periodic Coulomb (PME/Ewald) branch.

This is the NON-JITTABLE boundary: it builds the fixed-shape arrays the traced
`jax-pme` engine consumes -- the real-space neighbour list, the reciprocal k-grid,
the smearing, and all the padding masks/index maps. None of it is differentiable
(neighbour topology + grid shape are held fixed during a force evaluation, exactly
like the GNN neighbour list), so it lives in numpy on the host.

We drive `jaxpme.batched_mixed.Ewald`, whose batched container handles mixed
periodic / non-periodic systems plus padding. The differentiable energy itself is
applied in `coulomb_module.py`; here we only assemble the container.

Engine note: we build the periodic branch on `jax-pme` (pure-JAX, CPU, autodiff-
clean) rather than the torch nvalchemiops PME (GPU-only Warp FFI, non-
differentiable). Energies are NOT bit-parity with nvalchemiops -- both are correct
Ewald, but use different smearing/alpha/k-grid conventions -- so reproducing the
released orbmol-v2 checkpoint exactly is a separate, deferred concern. Correctness
is anchored on the analytic Madelung constant and large-box convergence to the
non-periodic direct sum (see tests/.../test_periodic_coulomb.py).
"""

from __future__ import annotations

import numpy as np

import jax

COULOMB_CONSTANT = 14.3996  # eV*A/e^2 (matches coulomb_module.COULOMB_CONSTANT)


def periodic_coulomb_energy(
    charges: jax.Array,  # (n_pad,) per-atom charges in our atom layout
    positions: jax.Array,  # (n_pad, 3) LIVE differentiable positions
    cell: jax.Array,  # (n_struct, 3, 3) LIVE differentiable cells
    sr_batch,  # jax-pme Batch namedtuple (host-prepped)
    nonperiodic_batch,
    periodic_batch,
    *,
    prefactor: float = COULOMB_CONSTANT,
) -> jax.Array:
    """Per-structure periodic electrostatic energy ``(n_struct,)`` from jax-pme.

    The host-prepped `sr_batch` carries the fixed neighbour list / masks / index
    maps; here we swap in the LIVE `positions`/`cell` (the namedtuple is a pytree,
    so `_replace` is a functional write) so `jax.grad` w.r.t. them flows -> forces
    (dE/dpos) and stress (dE/dcell, dE/dpos). The neighbour topology + k-grid stay
    fixed during the grad, exactly like the GNN's neighbour list.

    Energy convention matches orb: jax-pme returns ``E = sum_i q_i V_i`` with the
    1/2 absorbed, i.e. the same ``0.5 * k`` as the non-periodic direct sum.
    """
    from jaxpme.batched_mixed import Ewald

    calc = Ewald(prefactor=prefactor)
    sr_live = sr_batch._replace(positions=positions, cell=cell)
    return calc.energy(charges, sr_live, nonperiodic_batch, periodic_batch)


def _orb_neighbor_list(
    positions: np.ndarray,  # (n, 3)
    cell: np.ndarray,  # (3, 3)
    pbc: np.ndarray,  # (3,) bool
    cutoff: float,
    max_num_neighbors: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full real-space neighbour list (centers, others, cell_shifts) at `cutoff`.

    Reuses orb's `ForcefieldAtomsAdapter` so the PME real-space list is built by
    the same machinery as the GNN graph. The convention matches jax-pme exactly:
    `r_ij = pos[others] + cell_shifts @ cell - pos[centers]` (verified to ~1e-15),
    so `centers = senders`, `others = receivers`, `cell_shifts = unit_shifts`.

    NOTE: for a full JAX migration this could be swapped for jax-pme's own
    vesin-backed `to_structure` (`jaxpme.batched_mixed.batching.to_structure`),
    which builds the same `i, j, S` list directly from an ASE `Atoms`.
    """
    import ase
    import torch

    from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter

    atoms = ase.Atoms(
        numbers=np.ones(positions.shape[0], dtype=int),
        positions=positions,
        cell=cell,
        pbc=bool(pbc.all()),
    )
    adapter = ForcefieldAtomsAdapter(radius=float(cutoff), max_num_neighbors=max_num_neighbors)
    graph = adapter.from_ase_atoms(atoms)
    centers = graph.senders.cpu().numpy().astype(int)
    others = graph.receivers.cpu().numpy().astype(int)
    cell_shifts = graph.edge_features["unit_shifts"].cpu().numpy()
    return centers, others, np.rint(cell_shifts).astype(int)


def build_pme_structure(
    positions: np.ndarray,  # (n, 3)
    cell: np.ndarray,  # (3, 3)
    pbc: np.ndarray,  # (3,) bool
    lr_wavelength: float,
    *,
    smearing: float | None = None,
    cutoff: float | None = None,
    halfspace: bool = True,
) -> dict:
    """One jax-pme `batched_mixed` structure dict (periodic or non-periodic).

    Mirrors `jaxpme.batched_mixed.batching.prepare`, but takes raw arrays and uses
    orb's neighbour list for the periodic real-space pairs. `lr_wavelength` sets the
    reciprocal cutoff; jax-pme's defaults are cutoff = 8*lr_wavelength, smearing =
    2*lr_wavelength.
    """
    from jaxpme.batched_mixed.batching import to_lr

    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)
    is_periodic = bool(pbc.all())

    if cutoff is None:
        cutoff = lr_wavelength * 8.0
    if smearing is None:
        smearing = lr_wavelength * 2.0

    structure: dict = {
        "positions": positions,
        "cell": cell if not (cell == 0).all() else np.eye(3),
        "pbc": pbc,
        "charges": np.zeros(positions.shape[0]),  # placeholder; model supplies real q
    }

    if is_periodic:
        centers, others, shifts = _orb_neighbor_list(positions, cell, pbc, cutoff)
    else:
        # Non-periodic: jax-pme uses an all-pairs (upper-triangular) list, no cutoff.
        n = positions.shape[0]
        j, i = np.triu_indices(n, k=1)
        centers, others, shifts = i, j, np.zeros((i.shape[0], 3), dtype=int)

    structure["centers"] = centers
    structure["others"] = others
    structure["cell_shifts"] = shifts

    smearing_out, lr = to_lr(structure, lr_wavelength, smearing, halfspace=halfspace)
    structure["lr"] = lr
    if smearing_out is not None:
        structure["smearing"] = smearing_out
    return structure


def build_pme_batch(structures: list[dict], **size_overrides):
    """Assemble per-system structures into padded jax-pme batch containers.

    Returns `(sr_batch, nonperiodic_batch, periodic_batch)` -- the args
    `jaxpme.batched_mixed.Ewald().energy(charges, *these)` consumes. `size_overrides`
    (e.g. `num_atoms`, `num_structures`, `num_pairs`) pin the padded shapes so the
    container lines up with our `JaxAtomGraphs` buckets and jit caches once.
    """
    from jaxpme.batched_mixed.batching import get_batch

    _charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(
        structures, **size_overrides
    )
    return sr_batch, nonperiodic_batch, periodic_batch
