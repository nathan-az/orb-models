"""d(loss)/d(model) three ways must agree: torch backward, jax reverse-over-reverse,
jax forward-over-reverse (jvp) -- ON THE TRAINABLE PARTITION ONLY.

Why "trainable partition only": the loss genuinely depends on the frozen buffers
(energy target = energy - reference; every term divides by a normalizer std), so
reverse-mode returns *nonzero* grads for the reference/normalizer leaves while the
jvp path returns zero and torch returns None (requires_grad=False). Those leaves are
masked out of the optimiser by `trainable_filter`, so the contract we test is
agreement on exactly the leaves the optimiser actually steps -- gns + energy_head.mlp.

Setup mirrors the forward equivalence test: matched fp64 models, weights shared, plus
non-trivial *frozen* normalizer stats so the 1/std folding is exercised. Online stat
updates are turned off on both sides (covered separately by test_loss); this test
isolates the autograd of the loss.
"""

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from orb_models.common.atoms.jax import graph_batch as jgb
from orb_models.forcefield.models.jax.conservative_regressor import (
    compute_grads_jvp,
    compute_grads_reverse,
    trainable_filter,
)
from tests.common.model.jax.test_conservative_regressor import (
    _arrays,
    _build,
    _torch_graph,
)

WEIGHTS = {"energy": 1.0, "forces": 1.0, "stress": 1.0}


def _freeze_norm_stats(torch_model):
    """Non-trivial, FROZEN normalizer stats on all three normalizers.

    Non-trivial std so the loss's 1/std actually shows up in the grads; online=False
    so neither framework moves the stats during this test.
    """
    for norm, (mean, std) in (
        (torch_model.heads["energy"].normalizer, (0.5, 2.0)),
        (torch_model.grad_forces_normalizer, (-0.3, 1.7)),
        (torch_model.grad_stress_normalizer, (0.1, 0.8)),
    ):
        norm.bn.running_mean = torch.tensor([mean])
        norm.bn.running_var = torch.tensor([std**2])
        norm.online = False


def _targets_np(a, seed=23):
    rng = np.random.default_rng(seed)
    return {
        "energy": rng.standard_normal(a["G"]),  # absolute (G,)
        "forces": rng.standard_normal((a["N"], 3)),  # (N, 3)
        "stress": rng.standard_normal((a["G"], 6)),  # Voigt-6
    }


def _torch_grads_as_jax(jax_template, torch_model, torch_graph, targets, helpers):
    """torch d(loss)/d(params), mapped into the jax pytree via the weight-copy machinery.

    Trick: run backward, then build a torch model whose every param *data* is that
    param's *grad*, and feed it through the same `copy_conservative_regressor` used for
    weights. The result is a jax ConservativeRegressor whose trainable leaves hold torch
    grads (frozen leaves hold torch values, but we mask those out before comparing).

    torch must be in TRAIN mode: forces/stress are themselves dE/dx, so the outer
    backward needs `create_graph=True`, which orb gates on `self.training`.
    """
    torch_model.train()
    torch_graph.system_targets["energy"] = torch.tensor(targets["energy"])
    torch_graph.node_targets["forces"] = torch.tensor(targets["forces"])
    torch_graph.system_targets["stress"] = torch.tensor(targets["stress"])

    torch_model.zero_grad(set_to_none=True)
    torch_model.loss(torch_graph).loss.backward()

    # A trainable param with `.grad is None` after backward simply didn't affect the
    # loss -> its gradient is ZERO (e.g. the decoder, whose `pred` output the
    # conservative energy never consumes). Map that to zeros, NOT the param value.
    grads_by_name = {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in torch_model.named_parameters()
    }
    grad_model = copy.deepcopy(torch_model)
    for n, p in grad_model.named_parameters():
        if n in grads_by_name:
            p.data = grads_by_name[n]
    return helpers.copy_conservative_regressor(jax_template, grad_model)


def _trainable_leaves(grads, spec):
    params, _ = eqx.partition(grads, spec)
    return jax.tree.leaves(eqx.filter(params, eqx.is_inexact_array))


def test_grad_equivalence_torch_jvp_reverse(helpers, key):
    torch_model, jax_model = _build(key)
    _freeze_norm_stats(torch_model)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)

    a = _arrays()
    targets = _targets_np(a)
    jax_graph = jgb.to_jax(_torch_graph(a))
    jax_targets = {k: jnp.asarray(v) for k, v in targets.items()}

    # Same loss, differentiated three ways w.r.t. the model.
    torch_grads = _torch_grads_as_jax(
        jax_model, torch_model, _torch_graph(a), targets, helpers
    )
    jvp_grads, _ = compute_grads_jvp(jax_model, jax_graph, jax_targets, WEIGHTS)
    rev_grads, _ = compute_grads_reverse(jax_model, jax_graph, jax_targets, WEIGHTS)

    spec = trainable_filter(jax_model)
    t_leaves = _trainable_leaves(torch_grads, spec)
    j_leaves = _trainable_leaves(jvp_grads, spec)
    r_leaves = _trainable_leaves(rev_grads, spec)

    # The jax trainable partition must be exactly torch's trainable param set --
    # else a leaf torch freezes (rbf/reference/ZBL/normalizers) would be trained,
    # or vice versa. Catches future freeze/trainable drift automatically.
    n_torch_trainable = sum(p.requires_grad for p in torch_model.parameters())
    assert len(j_leaves) == n_torch_trainable
    assert len(t_leaves) == len(j_leaves) == len(r_leaves) > 0
    for lt, lj, lr in zip(t_leaves, j_leaves, r_leaves):
        assert lt.shape == lj.shape == lr.shape
        t, j, r = np.asarray(lt), np.asarray(lj), np.asarray(lr)
        # jax (both paths) vs torch: fp64 with shared weights -> agree tightly.
        np.testing.assert_allclose(j, t, atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(r, t, atol=1e-9, rtol=1e-7)
        # the two jax paths are the same maths -> agree to ~machine fp64.
        np.testing.assert_allclose(j, r, atol=1e-11, rtol=1e-9)


def test_reverse_grads_buffers_nonzero_but_jvp_zero(helpers, key):
    """The reason the comparison is partition-scoped, made explicit.

    On the FROZEN normalizer std, reverse-mode returns a nonzero grad (every term
    divides by std, so the loss depends on it) while the jvp path returns zero. Both
    are 'correct' -- they only agree once the buffers are masked out, which is what
    `trainable_filter` does. (The normalizer *mean* cancels in pred - target, so it
    is std, not mean, that carries the spurious reverse grad for forces/stress.)
    """
    torch_model, jax_model = _build(key)
    _freeze_norm_stats(torch_model)
    jax_model = helpers.copy_conservative_regressor(jax_model, torch_model)
    a = _arrays()
    jax_graph = jgb.to_jax(_torch_graph(a))
    jax_targets = {k: jnp.asarray(v) for k, v in _targets_np(a).items()}

    jvp_grads, _ = compute_grads_jvp(jax_model, jax_graph, jax_targets, WEIGHTS)
    rev_grads, _ = compute_grads_reverse(jax_model, jax_graph, jax_targets, WEIGHTS)

    rev_std = rev_grads.grad_forces_normalizer.std
    jvp_std = jvp_grads.grad_forces_normalizer.std
    assert np.abs(np.asarray(rev_std)).max() > 1e-6  # reverse: spuriously nonzero
    assert np.allclose(np.asarray(jvp_std), 0.0)  # jvp: exactly zero
