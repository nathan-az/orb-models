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
import jax.numpy as jnp
from orb_models.common.atoms.jax.graph_batch import JaxAtomGraphs
from orb_models.forcefield.models.jax.conservative_regressor import (
    ConservativeRegressor,
    compute_grads_jvp,
    trainable_filter,
    update_normalizer_buffers,
)

# The two interchangeable d(loss)/d(model) paths. `compute_grads_jvp` is the
# default everywhere; `compute_grads_reverse` exists so the benchmark can time
# the naive reverse-over-reverse path against the forward-over-reverse one.
GradFn = type(compute_grads_jvp)


def _onecycle_cos_schedule(
    initial: float, peak: float, final: float, total_steps: int, pct_start: float
):
    """Torch `OneCycleLR` (anneal_strategy='cos', two-phase) as a step->value fn.

    optax's stock `cosine_onecycle_schedule` is NOT this curve -- it agrees with
    torch only at step 0 and diverges from step 1 (different warmup ramp), so we
    replicate torch's exact two-phase formula instead. Torch builds:
        phase 0 (warmup): step 0 .. pct_start*total_steps-1, anneal initial -> peak
        phase 1 (anneal): .. total_steps-1,                  anneal peak    -> final
    with `anneal_cos(a, b, p) = b + (a-b)/2 * (cos(pi*p)+1)`. The same shape drives
    both the LR (initial->peak->final) and, reversed, the momentum.
    """
    step0_end = pct_start * total_steps - 1.0
    phase1_len = (total_steps - 1) - step0_end

    def cos(a, b, p):
        return b + (a - b) / 2.0 * (jnp.cos(jnp.pi * p) + 1.0)

    def schedule(step):
        step = step.astype(jnp.float64) if hasattr(step, "astype") else float(step)
        warm = cos(initial, peak, step / step0_end)
        anneal = cos(peak, final, (step - step0_end) / phase1_len)
        return jnp.where(step <= step0_end, warm, anneal)

    return schedule


def make_optimizer(
    lr: float,
    total_steps: int,
    *,
    div_factor: float = 10.0,
    final_div_factor: float = 10.0,
    pct_start: float = 0.05,
    max_momentum: float = 0.95,
    base_momentum: float = 0.85,
) -> optax.GradientTransformation:
    """Plain Adam + OneCycle, faithfully mirroring torch `get_optim`.

    torch passes `max_lr=lr*div_factor` to `OneCycleLR`, so:
        peak_lr    = lr * div_factor      (the cycle's max)
        initial_lr = peak_lr / div_factor = lr
        final_lr   = initial_lr / final_div_factor
    optax.adam (not adamw) has no weight decay, matching orb.

    Two things the stock optax one-cycle does NOT reproduce, both fixed here:
      * the LR *curve* -- replicated exactly via `_onecycle_cos_schedule` (the stock
        schedule only matches torch at step 0);
      * `OneCycleLR`'s default `cycle_momentum=True` -- torch anneals Adam's beta1
        between `max_momentum` and `base_momentum`, in the OPPOSITE direction to the
        LR (high momentum at low LR). beta1 cancels in the step-1 bias correction, so
        this only bites from step 2 on. We schedule beta1 with `inject_hyperparams`
        so optax accumulates/bias-corrects with the same per-step beta1 as torch.
    """
    peak_lr = lr * div_factor
    lr_schedule = _onecycle_cos_schedule(
        initial=peak_lr / div_factor,
        peak=peak_lr,
        final=(peak_lr / div_factor) / final_div_factor,
        total_steps=total_steps,
        pct_start=pct_start,
    )
    # Momentum cycles the OTHER way: starts at max, dips to base at peak LR, back up.
    b1_schedule = _onecycle_cos_schedule(
        initial=max_momentum,
        peak=base_momentum,
        final=max_momentum,
        total_steps=total_steps,
        pct_start=pct_start,
    )
    return optax.inject_hyperparams(optax.adam)(
        learning_rate=lr_schedule, b1=b1_schedule
    )


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
    grad_fn: GradFn = compute_grads_jvp,
) -> tuple[ConservativeRegressor, optax.OptState, dict[str, jax.Array]]:
    """One supervised step: buffers, grads, partition, optimiser, recombine.

    Order matters: the buffers are advanced from the TARGETS first (torch's
    side-effect-inside-the-loss made explicit), then the loss/grads are taken on
    that updated model -- so the normalization the loss uses is this step's stats.

    `grad_fn` selects the gradient path; it defaults to the forward-over-reverse
    `compute_grads_jvp` and can be swapped for `compute_grads_reverse` (same
    signature) to benchmark the naive path. The partition below drops any buffer
    grads, so either path is safe here.
    """
    spec = trainable_filter(model)

    # (3) Advance the category-3 buffers. This is the ONLY thing allowed to move
    #     the normalizer stats; the optimiser below never will.
    model = update_normalizer_buffers(model, targets, graph)

    # d(loss)/d(model). The jvp path already returns a zero cotangent for the
    # buffers; the reverse path would return a *nonzero* one. We do not rely on
    # that -- step (partition) drops the buffer grads either way.
    grads, breakdown = grad_fn(
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
    grad_fn: GradFn = compute_grads_jvp,
):
    """jit-compiled closure over the non-array config (optimizer/weights/has_stress).

    `eqx.filter_jit` traces the array leaves of (model, opt_state, graph, targets)
    and treats everything else as static, so closing over `optimizer`/`weights`
    keeps the jitted signature clean. This is the form the throughput/memory
    benchmark times; `grad_fn` picks the jvp (default) or reverse path.
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
            grad_fn=grad_fn,
        )

    return step
