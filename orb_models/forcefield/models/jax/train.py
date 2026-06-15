"""JAX/optax training step for the conservative forcefield.

This is the consumer of everything in `conservative_regressor.py`: it takes one
gradient pass (`compute_grads_jvp` by default) and turns it into an optimiser
update. The whole file exists to make ONE distinction concrete -- the difference
between the three categories of leaf in a `ConservativeRegressor`:

  1. static config (strings/bools)  -> baked into the treedef, grad never sees it
  2. trainable params (weights)     -> grad AND optimiser
  3. stateful buffers (normalizers) -> grad/optimiser must SKIP, but they DO change
                                       (advanced by `update_normalizer_buffers`)

Category 3 is the reason this file partitions. The buffers are float arrays in the
same pytree as the weights, so nothing structural separates them -- only the
runtime `trainable_filter` spec does. optax's `update`/`apply_updates` are
tree_maps over every leaf they're handed, so "don't let Adam touch the buffers"
is not a default you get for free; it is implemented by giving optax ONLY the
trainable partition.

Optimiser matches torch `get_optim` (common/training/util.py): plain Adam (no
weight decay) + OneCycleLR(max_lr=lr*10, pct_start=0.05, div/final_div=10).
"""

from __future__ import annotations

import equinox as eqx
import optax

import jax
from orb_models.common.atoms.jax.graph_batch import JaxAtomGraphs
from orb_models.forcefield.models.jax.conservative_regressor import (
    ConservativeRegressor,
    compute_grads_jvp,
    trainable_filter,
    update_normalizer_buffers,
)


def make_optimizer(
    lr: float,
    total_steps: int,
    *,
    div_factor: float = 10.0,
    final_div_factor: float = 10.0,
    pct_start: float = 0.05,
) -> optax.GradientTransformation:
    """Plain Adam + cosine OneCycle, mirroring torch `get_optim`.

    torch passes `max_lr=lr*div_factor`; optax's `peak_value` IS that max. Then
    initial_lr = peak/div_factor = lr and final_lr = initial/final_div_factor.
    optax.adam (not adamw) has no weight decay, matching orb.
    """
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=lr * div_factor,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    return optax.adam(schedule)


def init_opt_state(
    model: ConservativeRegressor, optimizer: optax.GradientTransformation
) -> optax.OptState:
    """Initialise optimiser state on the TRAINABLE partition only.

    `eqx.partition` returns `params` with every non-trainable leaf (buffers AND
    static config) replaced by `None`. optax treats `None` as an empty subtree, so
    it allocates Adam moment buffers for the weights and *nothing* for the
    normalizer stats -- the partition decision is baked in here, once.
    """
    params, _static = eqx.partition(model, trainable_filter(model))
    return optimizer.init(params)


def train_step(
    model: ConservativeRegressor,
    opt_state: optax.OptState,
    graph: JaxAtomGraphs,
    targets: dict[str, jax.Array],
    weights: dict[str, float],
    *,
    optimizer: optax.GradientTransformation,
    has_stress: bool = True,
) -> tuple[ConservativeRegressor, optax.OptState, dict[str, jax.Array]]:
    """One supervised step: buffers, grads, partition, optimiser, recombine.

    Order matters: the buffers are advanced from the TARGETS first (torch's
    side-effect-inside-the-loss made explicit), then the loss/grads are taken on
    that updated model -- so the normalization the loss uses is this step's stats.
    """
    spec = trainable_filter(model)

    # (3) Advance the category-3 buffers. This is the ONLY thing allowed to move
    #     the normalizer stats; the optimiser below never will.
    model = update_normalizer_buffers(model, targets, graph)

    # d(loss)/d(model). The jvp path already returns a zero cotangent for the
    # buffers; the reverse path would return a *nonzero* one. We do not rely on
    # that -- step (partition) drops the buffer grads either way.
    grads, breakdown = compute_grads_jvp(
        model, graph, targets, weights, has_stress=has_stress
    )

    # (1)+(2)+(3) split. params = trainable weights (rest None); static = buffers
    #     + config (trainable None). Same spec applied to grads => buffer grads -> None.
    params, static = eqx.partition(model, spec)
    grad_params, _ = eqx.partition(grads, spec)

    # The optimiser only ever sees `params` and `grad_params` -- the trainable half.
    updates, opt_state = optimizer.update(grad_params, opt_state, params)
    params = eqx.apply_updates(params, updates)

    # Recombine: new weights from the optimiser + the buffer-updated static half.
    model = eqx.combine(params, static)
    return model, opt_state, breakdown


def make_train_step(
    optimizer: optax.GradientTransformation,
    weights: dict[str, float],
    *,
    has_stress: bool = True,
):
    """jit-compiled closure over the non-array config (optimizer/weights/has_stress).

    `eqx.filter_jit` traces the array leaves of (model, opt_state, graph, targets)
    and treats everything else as static, so closing over `optimizer`/`weights`
    keeps the jitted signature clean. This is the form the throughput/memory
    benchmark will time.
    """

    @eqx.filter_jit
    def step(model, opt_state, graph, targets):
        return train_step(
            model,
            opt_state,
            graph,
            targets,
            weights,
            optimizer=optimizer,
            has_stress=has_stress,
        )

    return step
