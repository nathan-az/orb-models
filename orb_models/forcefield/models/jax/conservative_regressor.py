"""JAX conservative forcefield: energy -> forces/stress via jax.grad.

This is the consumer of compute_differentiable_edge_vectors and the place the
autograd boundary lives. Port of the torch ConservativeForcefieldRegressor
forward (conservative_regressor.py:160-249), but expressed the JAX way: a pure
`energy_fn` that we differentiate, instead of torch.autograd.grad with
create_graph bookkeeping.

Equinox note: there is no separate `params` pytree (that's a flax/haiku idiom).
An `eqx.Module` *is* the parameter pytree, so we pass the `model` directly and
call `model.gns(graph)`. We differentiate w.r.t. positions/disp/generator with
plain `jax.grad(..., argnums)`, and w.r.t. the model with `eqx.filter_grad`
(which masks the module's static, non-array fields).

Three differentiation targets of `energy_fn`, all evaluated at zero:
    positions          -> forces          = -dE/d(positions)
    stress_displacement-> stress          =  dE/d(disp) / volume
    generator          -> rotational_grad =  dE/d(generator)        (equigrad)

Two ways to get d(loss)/d(model), both built on the same `_total_loss`:
    compute_grads_reverse  naive reverse-over-reverse (differentiate the loss,
                           which itself contains the inner energy->forces grad).
                           Simple; the reference for correctness tests.
    compute_grads_jvp      forward-over-reverse. The loss only needs directional
                           derivatives of E, not the full mixed Hessian. We
                           contract each derivative term against a frozen
                           cotangent and push all of them through E in ONE
                           forward-mode pass (jax.jvp), whose primal we reuse for
                           the (zeroth-order) energy term.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from orb_models.common.atoms.jax.graph_batch import (
    JaxAtomGraphs,
    compute_differentiable_edge_vectors,
)
from orb_models.common.models.jax.gns import MoleculeGNS
from orb_models.forcefield.models.jax.forcefield_heads import ScalarNormalizer
from orb_models.forcefield.models.jax.loss import (
    forces_loss,
    full_3x3_to_voigt_6,
    mean_error,
    stress_loss,
)


class ConservativeRegressor(eqx.Module):
    """Bundles the weight-bearing pieces into a single pytree (the "model").

    `energy_head` contract: given the backbone node features (N, latent) and the
    graph, return the per-graph *interaction* energy (G,) -- i.e. it does the
    per-atom -> per-graph aggregation internally, like the torch EnergyHead. For
    the autodiff tests this can be a stand-in (Linear(latent->1) + segment mean);
    swap in the real EnergyHead (+ reference energy / normalizer / ZBL) later.
    """

    gns: MoleculeGNS
    energy_head: eqx.Module
    pair_repulsion: eqx.Module | None = None
    # Force/stress target normalizers (torch grad_forces/grad_stress_normalizer).
    # Fixed buffers, not params -- default to identity until copied/fit.
    grad_forces_normalizer: ScalarNormalizer = eqx.field(
        default_factory=lambda: ScalarNormalizer(mean=jnp.zeros(1), std=jnp.ones(1))
    )
    grad_stress_normalizer: ScalarNormalizer = eqx.field(
        default_factory=lambda: ScalarNormalizer(mean=jnp.zeros(1), std=jnp.ones(1))
    )
    forces_loss_type: str = eqx.field(static=True, default="condhuber_0.01")


class Predictions(eqx.Module):
    """Physical outputs for inference (forces/stress already scaled)."""

    energy: jax.Array  # (G,) per-graph interaction energy
    forces: jax.Array  # (N, 3)
    rotational_grad: jax.Array  # (G, 3, 3)
    stress: jax.Array | None  # (G, 3, 3) or None


def energy_fn(
    positions: jax.Array,  # (N, 3)    grad target -> forces
    stress_displacement: jax.Array,  # (G, 3, 3) grad target -> stress      (pass zeros)
    generator: jax.Array,  # (G, 3, 3) grad target -> rotational_grad (pass zeros)
    graph: JaxAtomGraphs,  # connectivity, Z, cell, unit_shifts, ...
    model: ConservativeRegressor,
) -> jax.Array:
    """Per-graph interaction energy (G,), differentiable w.r.t. the first three args.

    `positions` enters the energy ONLY through the edge `vectors` (the backbone
    featurizes nodes from atomic embeddings, not coordinates), so computing the
    differentiable vectors and injecting them into the graph carries the full
    position/strain/rotation dependence -- exactly torch's
    `batch.edge_features["vectors"] = vectors; self.model(batch)`, but with an
    immutable `eqx.tree_at` "write" instead of in-place mutation.

    Returns per-graph energy rather than its sum so the same forward pass feeds
    both the (per-graph) energy loss and the differentiated force/stress terms.
    Summing happens at the differentiation site: forces/stress are local, so
    d(sum_g E_g)/d(positions of graph k) == dE_k/d(positions of graph k).
    """
    vectors = compute_differentiable_edge_vectors(
        positions,
        graph.edge_features["unit_shifts"],
        graph.system_features["cell"],
        graph.senders,
        graph.receivers,
        graph.per_node_graph_index,
        graph.per_edge_graph_index,
        stress_displacement,
        generator,
    )
    graph = eqx.tree_at(lambda g: g.edge_features["vectors"], graph, vectors)

    out = model.gns(graph)
    interaction = model.energy_head(out["node_features"], graph)  # (G,)
    # ZBL repulsion (if present) is just another energy term sharing the same
    # differentiable `vectors`, so it flows into forces/stress via the outer grad.
    if model.pair_repulsion is not None:
        interaction = interaction + model.pair_repulsion(graph)
    return interaction


def _zero_grad_targets(graph: JaxAtomGraphs) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(positions, displacement=0, generator=0) -- the points we differentiate at."""
    positions = graph.node_features["positions"]
    G = graph.n_node.shape[0]
    disp = jnp.zeros((G, 3, 3), dtype=positions.dtype)
    gen = jnp.zeros((G, 3, 3), dtype=positions.dtype)
    return positions, disp, gen


def _energy_and_grads(
    graph: JaxAtomGraphs, model: ConservativeRegressor
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One value_and_grad pass: (per-graph energy, dE/dpos, dE/ddisp, dE/dgen).

    These are the *raw* derivatives. The sign flip (forces) and volume scaling
    (stress) live in `_total_loss`/`predict`, so there is a single place that
    knows the physics-variable <-> raw-derivative mapping.
    """
    positions, disp, gen = _zero_grad_targets(graph)

    def scalar_energy(p, d, g):
        per_graph = energy_fn(p, d, g, graph, model)  # (G,)
        return per_graph.sum(), per_graph  # scalar to differentiate; (G,) as aux

    (_, per_graph_energy), (dE_dpos, dE_ddisp, dE_dgen) = jax.value_and_grad(
        scalar_energy, argnums=(0, 1, 2), has_aux=True
    )(positions, disp, gen)
    return per_graph_energy, dE_dpos, dE_ddisp, dE_dgen


def predict(
    graph: JaxAtomGraphs,
    model: ConservativeRegressor,
    *,
    has_stress: bool = True,
) -> Predictions:
    """Inference: energy + conservative forces/stress/equigrad in one grad pass."""
    energy, dE_dpos, dE_ddisp, dE_dgen = _energy_and_grads(graph, model)
    volume = jnp.abs(jnp.linalg.det(graph.system_features["cell"]))  # (G,)
    return Predictions(
        energy=energy,
        forces=-dE_dpos,
        rotational_grad=dE_dgen,
        stress=dE_ddisp / volume[:, None, None] if has_stress else None,
    )


def _total_loss(
    energy: jax.Array,
    dE_dpos: jax.Array,
    dE_ddisp: jax.Array,
    dE_dgen: jax.Array,
    graph: JaxAtomGraphs,
    targets: dict[str, jax.Array],
    weights: dict[str, float],
    model: ConservativeRegressor,
    has_stress: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Scalar training loss as a function of the *raw* energy + derivatives.

    Written in terms of (energy, dE_dpos, dE_ddisp, dE_dgen) on purpose: running
    `jax.grad` over this w.r.t. those four args yields the cotangents for the jvp
    path with the sign (forces = -dE_dpos), the 1/volume (stress = dE_ddisp/V), the
    normalizer 1/std, the condhuber clip, and the loss weights all folded in. `model`
    supplies the (fixed-buffer) energy head + force/stress normalizers; it is an
    extra arg, NOT differentiated by the cotangent `jax.grad(argnums=(0,1,2,3))`.

    Targets: `energy` (G,) absolute, `forces` (N,3), `stress` (G,6) Voigt.
    """
    head = model.energy_head

    # Energy: huber on reference-subtracted, normalized interaction energy.
    reference = head.reference(
        graph.node_features["atomic_numbers"], graph.per_node_graph_index, graph.n_node.shape[0]
    )
    interaction_target = targets["energy"] - reference
    e_pred = head.normalize_for_loss(energy, graph)
    e_target = head.normalize_for_loss(interaction_target, graph)
    energy_l = weights["energy"] * mean_error(e_pred, e_target, head.loss_type)

    # Forces = -dE/dpos: normalize then condhuber.
    forces_l = weights["forces"] * forces_loss(
        -dE_dpos, targets["forces"], model.grad_forces_normalizer, model.forces_loss_type
    )

    total = energy_l + forces_l
    breakdown = {"energy": energy_l, "forces": forces_l}

    if has_stress:
        volume = jnp.abs(jnp.linalg.det(graph.system_features["cell"]))  # (G,)
        stress = full_3x3_to_voigt_6(dE_ddisp / volume[:, None, None])  # (G,6)
        stress_l = weights["stress"] * stress_loss(
            stress, targets["stress"], model.grad_stress_normalizer, head.loss_type
        )
        total = total + stress_l
        breakdown["stress"] = stress_l

    # An equigrad / rotational_grad regulariser would consume `dE_dgen` here; absent,
    # jax.grad gives a zero `dE_dgen` cotangent and the generator tangent is a no-op.
    breakdown["total"] = total
    return total, breakdown


def total_loss(
    model: ConservativeRegressor,
    graph: JaxAtomGraphs,
    targets: dict[str, jax.Array],
    weights: dict[str, float],
    *,
    has_stress: bool = True,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """`_total_loss` composed with the energy/grad forward pass."""
    energy, dE_dpos, dE_ddisp, dE_dgen = _energy_and_grads(graph, model)
    return _total_loss(
        energy, dE_dpos, dE_ddisp, dE_dgen, graph, targets, weights, model, has_stress
    )


def compute_grads_reverse(
    model: ConservativeRegressor,
    graph: JaxAtomGraphs,
    targets: dict[str, jax.Array],
    weights: dict[str, float],
    *,
    has_stress: bool = True,
) -> tuple[ConservativeRegressor, dict[str, jax.Array]]:
    """Naive reverse-over-reverse: differentiate the loss directly.

    *** The forward-over-backward / second-order spot. *** `total_loss` already
    takes an inner grad (E -> forces/stress); this outer grad differentiates
    *through* that inner grad. JAX does nested differentiation natively -- no
    create_graph / retain_graph. Correct and simple; the reference the jvp path
    is benchmarked and cross-checked against.
    """
    grads, breakdown = eqx.filter_grad(total_loss, has_aux=True)(
        model, graph, targets, weights, has_stress=has_stress
    )
    return grads, breakdown


def compute_grads_jvp(
    model: ConservativeRegressor,
    graph: JaxAtomGraphs,
    targets: dict[str, jax.Array],
    weights: dict[str, float],
    *,
    has_stress: bool = True,
) -> tuple[ConservativeRegressor, dict[str, jax.Array]]:
    """Forward-over-reverse: cotangents from `jax.grad`, one jvp through E.

    d(loss)/d(model) = <dL/d energy, d energy/d model>          (zeroth order)
                     + <dL/d(dE/dX), d(dE/dX)/d model>  for X in {pos, disp, gen}

    Each <., .> contracts a *frozen* cotangent against a derivative of E. Because
    jax.jvp is linear in its tangents, all three derivative terms superpose into a
    single directional derivative, computed in one forward-mode pass whose primal
    is the per-graph energy we reuse for the zeroth-order term.
    """
    # 1. One reverse pass for the prediction VALUES the cotangents depend on.
    energy, dE_dpos, dE_ddisp, dE_dgen = _energy_and_grads(graph, model)

    # 2. Cotangents in *raw-grad space*. jax.grad over `_total_loss` folds in the
    #    sign (forces), 1/volume (stress), per-term 1/n, and weights automatically
    #    -- for whatever `_huber` is replaced with. Freeze them: we contract
    #    against them, we do not differentiate them.
    frozen = jax.tree.map(jax.lax.stop_gradient, (energy, dE_dpos, dE_ddisp, dE_dgen))
    (c_energy, c_pos, c_disp, c_gen), breakdown = jax.grad(
        _total_loss, argnums=(0, 1, 2, 3), has_aux=True
    )(*frozen, graph, targets, weights, model, has_stress)

    # 3. Single forward-mode pass. primal = per-graph energy (-> energy term),
    #    tangent (G,) = per-graph directional derivative; summing it gives
    #    <cotangents, d(sum_g E_g)/d inputs> = the force/stress/equigrad term.
    positions, disp, gen = _zero_grad_targets(graph)

    def surrogate(m: ConservativeRegressor) -> jax.Array:
        per_graph_energy, tangent = jax.jvp(
            lambda p, d, g: energy_fn(p, d, g, graph, m),
            (positions, disp, gen),
            (c_pos, c_disp, c_gen),
        )
        return jnp.vdot(c_energy, per_graph_energy) + tangent.sum()

    grads = eqx.filter_grad(surrogate)(model)
    return grads, breakdown
