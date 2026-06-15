"""Two-step train-loop equivalence: jax vs torch, on the full orb-v3 model.

This is the end-to-end "does a real training step agree with torch" check. It sits
on top of the pieces already tested in isolation and ties them together across an
*optimiser update*, which is the only thing none of the existing tests exercise:

  * predictions parity        -> test_conservative_regressor (energy/forces/stress)
  * grads three ways          -> test_grad_equivalence (torch / reverse / jvp)
  * optax step self-consistency -> test_train (partition contract, jit==eager)

What is NEW here, and why TWO steps:

  1. LOSS VALUE parity. The grad test only compares *grads*; a constant offset in
     the loss has zero grad, so the scalar loss value has never been checked against
     torch. We compare the per-term + total loss directly.

  2. OPTIMISER-MIRROR parity. `make_optimizer` (optax) must match torch `get_optim`
     (Adam + OneCycleLR). Two pieces of that only diverge AFTER step 0, so a
     single-step test cannot see them:
       - the LR *curve* (the optax stock one-cycle agreed with torch only at step 0);
       - OneCycleLR's `cycle_momentum=True`, which anneals Adam's beta1 and cancels in
         the step-1 bias correction. We assert the LR and beta1 actually used each
         step equal torch's, exactly -- this is the decisive mirror check.

  3. EQUIVALENCE PRESERVED ACROSS THE UPDATE. We re-check loss + grad parity on the
     optimiser-moved weights (after step 0 AND after step 1). This is the user's
     "confirm the same is true after an update step".

Note we deliberately do NOT compare raw weights elementwise after a step. Adam's
first update is ~`lr * sign(g)`, so on parameters with near-zero gradient (which the
loss is, by definition, insensitive to) the update is eps-conditioned and the two
frameworks' weights drift by ~1e-4 even with a perfect optimiser mirror. That drift
lives entirely in loss-insensitive directions, so the loss/grad re-checks above stay
tight (~1e-7) while a fp64 weight comparison would be spuriously flaky. Schedule
parity + identical grads + identical Adam rule already pin the update; the loss/grad
re-checks confirm it functionally.

Run once per jax grad path (jvp / reverse) so an update driven by either stays
locked to torch -- the two paths are already proven equal in test_grad_equivalence,
so we don't re-compare them to each other here.

Normalizers are FROZEN (online=False / inference no-op) so the only thing moving
between steps is the weights via the optimiser. Buffer-update parity (torch BatchNorm
momentum vs `update_normalizer_buffers`) is a separate equivalence question; freezing
keeps a divergence here unambiguously about the optimiser/grads, not the stats.
"""

import copy

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.common.training.util import get_optim
from orb_models.forcefield.models.jax.conservative_regressor import (
    compute_grads_jvp,
    compute_grads_reverse,
    total_loss,
    trainable_filter,
)
from orb_models.forcefield.models.jax.train import init_opt_state, make_optimizer
from tests.common.model.jax.test_conservative_regressor import (
    _arrays,
    _build_real_features,
    _torch_graph,
)
from tests.common.model.jax.test_grad_equivalence import (
    _freeze_norm_stats,
    _targets_np,
    _trainable_leaves,
)

WEIGHTS = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
LR, TOTAL_STEPS, N_STEPS = 1e-3, 100, 2

# torch <-> jax loss-name correspondence (both weighted; weight=1.0 here).
LOSS_TERMS = {"energy": "energy_loss", "forces": "forces_loss", "stress": "stress_loss"}


def _set_targets(torch_graph, targets):
    """Place absolute-energy / forces / Voigt-stress targets on the torch batch."""
    torch_graph.system_targets["energy"] = torch.tensor(targets["energy"])
    torch_graph.node_targets["forces"] = torch.tensor(targets["forces"])
    torch_graph.system_targets["stress"] = torch.tensor(targets["stress"])


def _grads_to_jax(jax_template, torch_model, helpers):
    """Map the torch model's CURRENT `.grad`s into a jax pytree (same trick as
    test_grad_equivalence, reusing the grads `.backward()` just produced). A trainable
    param with `.grad is None` didn't affect the loss -> gradient is zero, NOT the
    param value."""
    grads_by_name = {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in torch_model.named_parameters()
    }
    grad_model = copy.deepcopy(torch_model)
    for n, p in grad_model.named_parameters():
        if n in grads_by_name:
            p.data = grads_by_name[n]
    return helpers.copy_conservative_regressor(jax_template, grad_model)


def _jax_update(model, opt_state, optimizer, grad_fn, graph, targets):
    """One optax update via `grad_fn`, mirroring `train.train_step` MINUS the
    normalizer-buffer advance (frozen here). Returns (new_model, opt_state)."""
    spec = trainable_filter(model)
    grads, _ = grad_fn(model, graph, targets, WEIGHTS)
    params, static = eqx.partition(model, spec)
    grad_params, _ = eqx.partition(grads, spec)
    updates, opt_state = optimizer.update(grad_params, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return eqx.combine(params, static), opt_state


@pytest.mark.parametrize(
    "grad_fn", [compute_grads_jvp, compute_grads_reverse], ids=["jvp", "reverse"]
)
def test_two_train_steps_match_torch(helpers, key, grad_fn):
    torch_model, jax_model = _build_real_features(key)
    _freeze_norm_stats(torch_model)  # non-trivial, FROZEN stats on both sides
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    # train mode so forces/stress (= dE/dx) get create_graph=True; stats stay frozen.
    torch_model.train()
    spec = trainable_filter(jax_model)

    a = _arrays()
    targets = _targets_np(a)
    jax_targets = {k: jnp.asarray(v) for k, v in targets.items()}
    # jax graph is immutable across forwards (tree_at copies), so build it once.
    jax_graph = jgb.to_jax(_torch_graph(a))

    optimizer = make_optimizer(lr=LR, total_steps=TOTAL_STEPS)
    opt_state = init_opt_state(jax_model, optimizer)
    torch_opt, torch_sched = get_optim(LR, TOTAL_STEPS, torch_model)

    n_torch_trainable = sum(p.requires_grad for p in torch_model.parameters())

    def check_loss_and_grads(label):
        """Loss + grad parity vs torch on the CURRENT weights. Leaves torch grads
        populated on `torch_model` so the caller can step the torch optimiser."""
        torch_graph = _torch_graph(a)  # torch forward mutates the batch -> fresh one
        _set_targets(torch_graph, targets)
        torch_opt.zero_grad(set_to_none=True)
        torch_out = torch_model.loss(torch_graph)

        # (1) LOSS VALUES match (fp64, shared weights -> tight).
        _, jax_bd = total_loss(jax_model, jax_graph, jax_targets, WEIGHTS)
        for jax_key, torch_key in LOSS_TERMS.items():
            np.testing.assert_allclose(
                np.asarray(jax_bd[jax_key]),
                torch_out.log[torch_key].detach().numpy(),
                atol=1e-7,
                rtol=1e-6,
                err_msg=f"{label}: {jax_key} loss",
            )
        np.testing.assert_allclose(
            np.asarray(jax_bd["total"]),
            torch_out.loss.detach().numpy(),
            atol=1e-7,
            rtol=1e-6,
            err_msg=f"{label}: total loss",
        )

        # (2) GRADS match (jax `grad_fn` vs torch backward) on the trainable partition.
        torch_out.loss.backward()
        torch_grads = _grads_to_jax(jax_model, torch_model, helpers)
        jax_grads, _ = grad_fn(jax_model, jax_graph, jax_targets, WEIGHTS)
        j_leaves = _trainable_leaves(jax_grads, spec)
        # jax trainable partition must be exactly torch's trainable set (no drift).
        assert len(j_leaves) == n_torch_trainable
        t_leaves = _trainable_leaves(torch_grads, spec)
        assert len(j_leaves) == len(t_leaves) > 0
        for lj, lt in zip(j_leaves, t_leaves):
            assert lj.shape == lt.shape
            np.testing.assert_allclose(
                np.asarray(lj), np.asarray(lt), atol=1e-9, rtol=1e-7, err_msg=label
            )

    w0 = np.asarray(jax_model.gns._encoder.node_fn.mlp.layers[0].weight)  # type: ignore[attr-defined]

    for step in range(N_STEPS):
        check_loss_and_grads(f"step {step}")  # also populates torch_model.grad

        # (3) Optimiser-mirror: the LR and beta1 torch is ABOUT to use this step ...
        lr_t = torch_opt.param_groups[0]["lr"]
        b1_t = torch_opt.param_groups[0]["betas"][0]
        torch_opt.step()
        torch_sched.step()
        jax_model, opt_state = _jax_update(
            jax_model, opt_state, optimizer, grad_fn, jax_graph, jax_targets
        )
        # ... must equal the LR/beta1 optax JUST used (stored in the injected state).
        np.testing.assert_allclose(
            float(opt_state.hyperparams["learning_rate"]), lr_t, rtol=1e-8,
            err_msg=f"step {step}: learning rate",
        )
        np.testing.assert_allclose(
            float(opt_state.hyperparams["b1"]), b1_t, rtol=1e-8,
            err_msg=f"step {step}: adam beta1 (momentum cycling)",
        )

    # (3, cont.) Equivalence preserved AFTER the second update too.
    check_loss_and_grads("after step 1")

    # Sanity: the optimiser actually moved the weights (not a vacuous pass).
    w1 = np.asarray(jax_model.gns._encoder.node_fn.mlp.layers[0].weight)  # type: ignore[attr-defined]
    assert not np.allclose(w0, w1)
