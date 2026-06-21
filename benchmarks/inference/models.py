"""Model registry for the inference benchmark.

One place that maps a clean, unambiguous model key to everything the bench needs to
vary between models -- so `bench.py` dispatches on a `ModelSpec` instead of scattered
`if args.model == "..."` string checks.

Naming: keys mirror the library's own (`orb_models.forcefield.models.jax.port_weights`)
so there is no ambiguity. In particular `orbmol_v2` is **OrbMol version 2**, NOT orb-v2
-- the old `v2` / `conservative` CLI names conflated those and are gone.

The jax builders are wrapped in lazy functions that import inside the call: `bench.py`
runs one framework per process (torch and jax can't share a CUDA context), so importing
jax at registry-load time would break the torch run. Only `build_jax()` pulls jax in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelSpec:
    """How the bench builds and feeds one model, across both frameworks.

    The two behavioural flags coincide for today's models but are kept separate
    because they drive different things:
      * `electrostatics` -- torch `has_electrostatics`; on the jax side it selects the
        periodic-PME input path (padded graph + `pme_prep`).
      * `charge_spin`    -- attach `atoms.info["charge"/"spin"]` before graph build so
        the adapter carries them onto the graph.
    """

    description: str
    electrostatics: bool
    charge_spin: bool
    _build_jax: Callable[..., Any]

    def build_jax(self, *, key):
        """Build the jax model (imports jax/orb_models lazily, inside the call)."""
        return self._build_jax(key=key)


def _orb_v3_conservative_jax(*, key):
    from orb_models.forcefield.models.jax.port_weights import build_orb_v3_conservative_jax

    return build_orb_v3_conservative_jax(key=key)


def _orbmol_v2_jax(*, key):
    from orb_models.forcefield.models.jax.port_weights import build_orbmol_v2_jax

    return build_orbmol_v2_jax(key=key)


MODELS: dict[str, ModelSpec] = {
    "orb_v3_conservative": ModelSpec(
        description="orb-v3 conservative backbone (materials; no electrostatics).",
        electrostatics=False,
        charge_spin=False,
        _build_jax=_orb_v3_conservative_jax,
    ),
    "orbmol_v2": ModelSpec(
        description="OrbMol-v2 (molecules; periodic-PME electrostatics + charge/spin).",
        electrostatics=True,
        charge_spin=True,
        _build_jax=_orbmol_v2_jax,
    ),
}

DEFAULT_MODEL = "orb_v3_conservative"
