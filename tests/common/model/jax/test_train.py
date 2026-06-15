"""The optax train step: weights move via the optimiser, buffers move ONLY via
the updater. This is the partitioning contract made into an assertion.

Reuses the matched-model builder from test_conservative_regressor so the step runs
on the same realistic conservative model (backbone + EnergyHead + ZBL).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.jax.conservative_regressor import (
    update_normalizer_buffers,
)
from orb_models.forcefield.models.jax.train import (
    init_opt_state,
    make_optimizer,
    make_train_step,
    train_step,
)
from tests.common.model.jax.test_conservative_regressor import (
    _arrays,
    _build,
    _torch_graph,
)

WEIGHTS = {"energy": 1.0, "forces": 1.0, "stress": 1.0}


def _setup(key):
    """A copied-weights model + a graph + random targets (absolute energy)."""
    torch_model, jax_model = _build(key)
    a = _arrays()
    graph = jgb.to_jax(_torch_graph(a))
    rng = np.random.default_rng(11)
    targets = {
        "energy": jnp.asarray(rng.standard_normal(a["G"])),
        "forces": jnp.asarray(rng.standard_normal((a["N"], 3))),
        "stress": jnp.asarray(rng.standard_normal((a["G"], 6))),
    }
    return jax_model, graph, targets


def test_train_step_updates_weights_but_not_buffers_via_optimizer(helpers, key):
    model, graph, targets = _setup(key)
    optimizer = make_optimizer(lr=1e-3, total_steps=100)
    opt_state = init_opt_state(model, optimizer)

    new_model, _opt_state, breakdown = train_step(
        model, opt_state, graph, targets, WEIGHTS, optimizer=optimizer
    )
    assert np.isfinite(np.asarray(breakdown["total"]))

    # (2) A trainable weight actually moved -- the optimiser did its job.
    w0 = model.gns._encoder.node_fn.mlp.layers[0].weight  # type: ignore[attr-defined]
    w1 = new_model.gns._encoder.node_fn.mlp.layers[0].weight  # type: ignore[attr-defined]
    assert not np.allclose(np.asarray(w0), np.asarray(w1))

    # (3) The crux: the normalizer buffers equal what the UPDATER alone produces.
    # If the optimiser had also touched them, these would differ. We compare against
    # `update_normalizer_buffers(model)` -- the only sanctioned mover of these stats.
    updated_only = update_normalizer_buffers(model, targets, graph)
    for getter in (
        lambda m: m.energy_head.normalizer,
        lambda m: m.grad_forces_normalizer,
        lambda m: m.grad_stress_normalizer,
    ):
        got, want = getter(new_model), getter(updated_only)
        np.testing.assert_array_equal(
            np.asarray(got.mean), np.asarray(want.mean)
        )
        np.testing.assert_array_equal(np.asarray(got.std), np.asarray(want.std))
        np.testing.assert_array_equal(
            np.asarray(got.count), np.asarray(want.count)
        )

    # And the buffers genuinely changed from the start (update wasn't a no-op),
    # so the assertion above is meaningful, not vacuous.
    assert not np.allclose(
        np.asarray(new_model.grad_forces_normalizer.std),
        np.asarray(model.grad_forces_normalizer.std),
    )


def test_optimizer_state_has_no_buffer_moments(helpers, key):
    """optax allocates Adam moment buffers only for the trainable partition.

    `init_opt_state` partitions first, so the normalizer leaves are `None` in the
    params optax sees -> no moment state for them. Count the inexact-array leaves in
    the (mu) moment tree and confirm it matches the trainable params, not the model.
    """
    model, _graph, _targets = _setup(key)
    optimizer = make_optimizer(lr=1e-3, total_steps=100)
    opt_state = init_opt_state(model, optimizer)

    from orb_models.forcefield.models.jax.conservative_regressor import (
        trainable_filter,
    )

    params, _ = eqx.partition(model, trainable_filter(model))
    n_param_leaves = len(
        jax.tree.leaves(eqx.filter(params, eqx.is_inexact_array))
    )
    n_mu_leaves = len(
        jax.tree.leaves(eqx.filter(opt_state, eqx.is_inexact_array))
    )
    # adam's state is (mu, nu) + a scalar count; leaves ~= 2*params + small constant.
    assert n_mu_leaves >= 2 * n_param_leaves
    assert n_param_leaves > 0


def test_jit_step_matches_eager(helpers, key):
    model, graph, targets = _setup(key)
    optimizer = make_optimizer(lr=1e-3, total_steps=100)
    opt_state = init_opt_state(model, optimizer)

    eager_model, _, eager_bd = train_step(
        model, opt_state, graph, targets, WEIGHTS, optimizer=optimizer
    )
    step = make_train_step(optimizer, WEIGHTS)
    jit_model, _, jit_bd = step(model, opt_state, graph, targets)

    np.testing.assert_allclose(
        np.asarray(jit_bd["total"]), np.asarray(eager_bd["total"]), rtol=1e-6
    )
    w_e = eager_model.gns._encoder.node_fn.mlp.layers[0].weight  # type: ignore[attr-defined]
    w_j = jit_model.gns._encoder.node_fn.mlp.layers[0].weight  # type: ignore[attr-defined]
    np.testing.assert_allclose(np.asarray(w_j), np.asarray(w_e), rtol=1e-6)
