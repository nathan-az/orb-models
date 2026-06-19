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
    # Guard the phase denominators: at `total_steps == 1/pct_start` (e.g. 20 with the
    # default 0.05) `step0_end` is exactly 0, so `step/step0_end` is 0/0 = NaN
    warm_denom = step0_end if step0_end > 0.0 else 1.0
    anneal_denom = phase1_len if phase1_len > 0.0 else 1.0

    def cos(a, b, p):
        return b + (a - b) / 2.0 * (jnp.cos(jnp.pi * p) + 1.0)

    def schedule(step):
        # int count -> float at the ambient precision (fp32 in training; the fp64
        # equivalence tests run under jax_enable_x64 and need the schedule in fp64
        # to match torch's OneCycleLR to tolerance).
        step = step * 1.0 if hasattr(step, "astype") else float(step)
        warm = cos(initial, peak, step / warm_denom)
        anneal = cos(peak, final, (step - step0_end) / anneal_denom)
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

    # (3) Advance the category-3 buffers into `updated`. This is the ONLY thing
    #     allowed to move the normalizer stats; the optimiser below never will.
    #     We keep `model` (PRE-update stats) around on purpose -- see next.
    updated = update_normalizer_buffers(model, targets, graph)

    # d(loss)/d(model). FORWARD uses `model` (PRE-update stats), so the energy
    # prediction is denormalized with this step's *old* stats -- matching torch,
    # which runs the forward before advancing the running stats inside its loss.
    # The LOSS NORMALIZATION uses `updated` (POST-update stats) via `loss_model`.
    # The jvp path returns a zero cotangent for the buffers; reverse a nonzero one;
    # the partition below drops buffer grads either way.
    grads, metrics = grad_fn(
        model, graph, targets, weights, has_stress=has_stress, loss_model=updated
    )

    # (1)+(2)+(3) split. params = trainable weights (rest None); static = buffers
    #     + config (trainable None). Same spec applied to grads => buffer grads -> None.
    #     Partition `updated` so the kept buffers are the POST-update stats; its
    #     trainable weights are identical to `model`'s (the update only moved buffers),
    #     so they align with `grads` (taken w.r.t. `model`).
    params, static = eqx.partition(updated, spec)
    grad_params, _ = eqx.partition(grads, spec)

    # The optimiser only ever sees `params` and `grad_params` -- the trainable half.
    grad_norm = optax.global_norm(grad_params)
    updates, opt_state = optimizer.update(grad_params, opt_state, params)
    params = eqx.apply_updates(params, updates)

    # Recombine: new weights from the optimiser + the buffer-updated static half.
    model = eqx.combine(params, static)
    metrics["grad_norm"] = grad_norm
    return model, opt_state, metrics


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


def make_accum_train_step(
    optimizer: optax.GradientTransformation,
    weights: dict[str, float],
    *,
    has_stress: bool = True,
    grad_fn: GradFn = compute_grads_jvp,
):
    """Like `make_train_step` but accumulates grads over several micro-batches.

    The fused `make_train_step` does grad+update in one jitted call; gradient
    accumulation needs the two halves separated so several buckets contribute to
    one optimiser update. This decomposes `train_step` accordingly:

      * `_micro` (jitted): advance the normalizer buffers from this micro-batch,
        take d(loss)/d(weights) on the buffer-updated model, and return the
        *trainable-partition* grads (buffer grads dropped, as in `train_step`).
        The returned model carries the advanced buffers, which chain into the next
        micro-batch -- mirroring torch, where each forward advances BN stats once.
      * `_apply` (jitted): one optimiser update from the summed grads.

    The returned `step(model, opt_state, batches)` takes a list of
    ``(graph, targets)`` (each padded to the SAME bucket shape, so `_micro`
    compiles once) and returns ``(model, opt_state, metrics)`` with grads
    averaged over the micro-batches. With a one-element list it is equivalent to
    `make_train_step` (one buffer advance, one update). Same `grad_fn` choice, so
    accumulation works for both the jvp and reverse paths.
    """

    @eqx.filter_jit
    def _micro(model, graph, targets):
        spec = trainable_filter(model)
        updated = update_normalizer_buffers(model, targets, graph)
        grads, metrics = grad_fn(
            model, graph, targets, weights, has_stress=has_stress, loss_model=updated
        )
        grad_params, _ = eqx.partition(grads, spec)
        return updated, grad_params, metrics

    @eqx.filter_jit
    def _apply(model, opt_state, grad_params):
        spec = trainable_filter(model)
        params, static = eqx.partition(model, spec)
        grad_norm = optax.global_norm(grad_params)
        updates, opt_state = optimizer.update(grad_params, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return eqx.combine(params, static), opt_state, grad_norm

    def step(model, opt_state, batches):
        acc_grads = None
        acc_bd: dict | None = None
        for graph, targets in batches:
            model, grad_params, metrics = _micro(model, graph, targets)
            acc_grads = (
                grad_params
                if acc_grads is None
                else jax.tree.map(jnp.add, acc_grads, grad_params)
            )
            acc_bd = (
                dict(metrics)
                if acc_bd is None
                else {k: acc_bd[k] + v for k, v in metrics.items()}
            )
        n = len(batches)
        if n > 1:
            acc_grads = jax.tree.map(lambda g: g / n, acc_grads)
            acc_bd = {k: v / n for k, v in acc_bd.items()}  # type: ignore[union-attr]
        model, opt_state, grad_norm = _apply(model, opt_state, acc_grads)
        acc_bd["grad_norm"] = grad_norm  # type: ignore[index]
        return model, opt_state, acc_bd

    return step
