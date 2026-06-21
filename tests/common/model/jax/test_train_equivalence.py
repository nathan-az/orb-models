"""Two-step train-loop equivalence: jax vs torch, on the full orb-v3 model.

Ties together the isolated pieces (predictions, grads, optax step) across an
*optimiser update*. Each step checks, against torch:

  * loss value parity (per-term + total) -- grads alone can't catch a constant offset;
  * optimiser-mirror parity -- `make_optimizer` (optax) vs torch `get_optim`
    (Adam + OneCycleLR): the LR curve and `cycle_momentum` beta1 only diverge after
    step 0, so we assert the LR + beta1 used each step equal torch's exactly;
  * loss + grad parity re-checked on the optimiser-moved weights after each step.

Two steps because the optimiser pieces above only diverge after step 0. We do NOT
compare raw weights elementwise: Adam's update is eps-conditioned on near-zero-grad
(loss-insensitive) directions, so weights drift ~1e-4 even with a perfect mirror,
while the loss/grad re-checks stay tight (~1e-7).

The `padded` parametrization additionally runs the jax side through `to_padded_numpy`
(same systems + a dummy padding graph), proving a packed+padded+masked step matches
torch's unpadded batch. Normalizers are frozen so only the weights move between steps.
"""

import copy

import equinox as eqx
import jax
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
from orb_models.forcefield.models.jax.conservative_regressor import (
    update_normalizer_buffers,
)
from orb_models.forcefield.models.jax.train import (
    init_opt_state,
    make_optimizer,
    train_step,
)
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

pytestmark = pytest.mark.equivalence

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


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
@pytest.mark.parametrize(
    "grad_fn", [compute_grads_jvp, compute_grads_reverse], ids=["jvp", "reverse"]
)
def test_two_train_steps_match_torch(helpers, key, grad_fn, padded):
    torch_model, jax_model = _build_real_features(key)
    _freeze_norm_stats(torch_model)  # non-trivial, FROZEN stats on both sides
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    # train mode so forces/stress (= dE/dx) get create_graph=True; stats stay frozen.
    torch_model.train()
    spec = trainable_filter(jax_model)

    a = _arrays()
    targets = _targets_np(a)
    # jax graph is immutable across forwards (tree_at copies), so build it once.
    if padded:
        # Top the SAME systems up to a larger fixed bucket; torch stays unpadded.
        # The masks must make every comparison below identical to the unpadded run.
        # `to_padded_numpy` is the production conversion+padding path (it reads the
        # targets straight off the torch batch, so set them first).
        n_pad, e_pad, g_pad = a["N"] + 6, a["E"] + 9, a["G"] + 2
        tg = _torch_graph(a)
        _set_targets(tg, targets)
        graph_np, targets_np = jgb.to_padded_numpy(tg, n_pad, e_pad, g_pad, has_stress=True)
        jax_graph = jax.device_put(graph_np)
        jax_targets = jax.device_put(targets_np)
    else:
        jax_graph = jgb.to_jax(_torch_graph(a))
        jax_targets = {k: jnp.asarray(v) for k, v in targets.items()}

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


# --- live-normalizer equivalence (the buffer-update / masked-stats parity) ----
#
# The test above FREEZES the normalizers to isolate the optimiser. This one does
# the opposite: it lets the running stats MOVE (torch BatchNorm momentum=None vs
# jax `update_normalizer_buffers`) across two real `train_step`s, so a stat desync
# would show up as a step-1 loss divergence well outside tolerance -- plus we assert
# the mean/std/count of all three normalizers match torch each step. Under `padded`,
# this is the missing check: that the MASKED jax update (padding atoms/graphs excluded)
# equals torch's pooled update on the same systems. Single grad path (jvp); the stat
# update is independent of the grad path, and the optimiser mirror is owned above.

NORM_GETTERS = (
    lambda m: m.energy_head.normalizer,
    lambda m: m.grad_forces_normalizer,
    lambda m: m.grad_stress_normalizer,
)
_TORCH_NORMS = (
    lambda tm: tm.heads["energy"].normalizer,
    lambda tm: tm.grad_forces_normalizer,
    lambda tm: tm.grad_stress_normalizer,
)


def _set_online_stats(torch_model, count: int):
    """Non-trivial ONLINE stats + a starting count, so the cumulative-average factor
    1/(count+1) is actually exercised (not wiped by a first-step factor of 1.0)."""
    for norm, (mean, std) in zip(
        [t(torch_model) for t in _TORCH_NORMS], [(0.5, 2.0), (-0.3, 1.7), (0.1, 0.8)]
    ):
        norm.bn.running_mean = torch.tensor([mean])
        norm.bn.running_var = torch.tensor([std**2])
        norm.bn.num_batches_tracked = torch.tensor(count)
        norm.online = True


def _seed_jax_counts(jax_model, count: int):
    """Mirror torch's starting `num_batches_tracked` into the jax normalizers (the
    weight-copy shares mean/std but not the count)."""
    for getter in NORM_GETTERS:
        jax_model = eqx.tree_at(
            lambda m: getter(m).count, jax_model, jnp.asarray(float(count))
        )
    return jax_model


def _assert_norms_match(jax_model, torch_model, label):
    for jget, tget in zip(NORM_GETTERS, _TORCH_NORMS):
        jn, tn = jget(jax_model), tget(torch_model)
        np.testing.assert_allclose(
            float(np.asarray(jn.mean).reshape(())), tn.bn.running_mean.item(),
            atol=1e-9, rtol=1e-7, err_msg=f"{label}: running mean")
        np.testing.assert_allclose(
            float(np.asarray(jn.std).reshape(())), torch.sqrt(tn.bn.running_var).item(),
            atol=1e-9, rtol=1e-7, err_msg=f"{label}: running std")
        np.testing.assert_allclose(
            float(jn.count), float(tn.bn.num_batches_tracked),
            err_msg=f"{label}: count")


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_live_normalizer_two_steps_match_torch(helpers, key, padded):
    torch_model, jax_model = _build_real_features(key)
    _set_online_stats(torch_model, count=5)  # non-trivial, NOT frozen
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    jax_model = _seed_jax_counts(jax_model, count=5)
    torch_model.train()  # online stat updates happen; create_graph for forces/stress

    a = _arrays()
    targets = _targets_np(a)
    if padded:
        n_pad, e_pad, g_pad = a["N"] + 6, a["E"] + 9, a["G"] + 2
        tg = _torch_graph(a)
        _set_targets(tg, targets)
        graph_np, targets_np = jgb.to_padded_numpy(tg, n_pad, e_pad, g_pad, has_stress=True)
        jax_graph = jax.device_put(graph_np)
        jax_targets = jax.device_put(targets_np)
    else:
        jax_graph = jgb.to_jax(_torch_graph(a))
        jax_targets = {k: jnp.asarray(v) for k, v in targets.items()}

    optimizer = make_optimizer(lr=LR, total_steps=TOTAL_STEPS)
    opt_state = init_opt_state(jax_model, optimizer)
    torch_opt, torch_sched = get_optim(LR, TOTAL_STEPS, torch_model)

    for step in range(N_STEPS):
        # torch: model.loss ADVANCES the stats (online) then computes the loss.
        torch_graph = _torch_graph(a)  # torch forward mutates the batch
        _set_targets(torch_graph, targets)
        torch_opt.zero_grad(set_to_none=True)
        torch_out = torch_model.loss(torch_graph)

        # jax: train_step advances stats (masked, under padding) then loss+grads+optim.
        jax_model, opt_state, jax_bd = train_step(
            jax_model, opt_state, jax_graph, jax_targets, WEIGHTS, optimizer=optimizer
        )

        # (a) loss parity -- sensitive to BOTH weights and the just-advanced stats.
        for jk, tk in LOSS_TERMS.items():
            np.testing.assert_allclose(
                np.asarray(jax_bd[jk]), torch_out.log[tk].detach().numpy(),
                atol=1e-7, rtol=1e-6, err_msg=f"step {step}: {jk} loss")
        np.testing.assert_allclose(
            np.asarray(jax_bd["total"]), torch_out.loss.detach().numpy(),
            atol=1e-7, rtol=1e-6, err_msg=f"step {step}: total loss")

        # (b) the running stats themselves match (masked jax update == torch pooled).
        _assert_norms_match(jax_model, torch_model, f"step {step}")

        torch_out.loss.backward()
        torch_opt.step()
        torch_sched.step()

    # Sanity: the stats actually moved from their seeded start (not a vacuous pass).
    assert float(jax_model.grad_forces_normalizer.count) == 5 + N_STEPS
    assert not np.isclose(float(np.asarray(jax_model.grad_forces_normalizer.std).reshape(())), 1.7)
