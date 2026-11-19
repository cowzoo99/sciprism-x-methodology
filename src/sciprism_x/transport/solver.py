from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from .schema import ObjectiveComponents, TransportDiagnostics, TransportResult


PreparedStructure = tuple[tuple[sparse.csr_matrix, sparse.csr_matrix], ...] | None


def masked_log_kernel(
    cost: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    log_kernel = np.full(cost.shape, -np.inf, dtype=np.float64)
    log_kernel[mask] = -cost[mask] / epsilon
    return log_kernel


def _structural_cost(
    plan: np.ndarray,
    source_adjacency: np.ndarray | None,
    target_adjacency: np.ndarray | None,
) -> float:
    prepared = _prepare_structure(plan, source_adjacency, target_adjacency)
    linear = _prepared_structural_linear_cost(plan, prepared)
    return float(np.sum(plan * linear))


def _prepare_structure(
    plan: np.ndarray,
    source_adjacency: np.ndarray | None,
    target_adjacency: np.ndarray | None,
) -> PreparedStructure:
    if source_adjacency is None and target_adjacency is None:
        return None
    if source_adjacency is None or target_adjacency is None:
        raise ValueError("both structural adjacency tensors are required")
    if source_adjacency.ndim != 3 or target_adjacency.ndim != 3:
        raise ValueError("structural adjacency inputs must be rank-three tensors")
    if source_adjacency.shape[0] == 0:
        raise ValueError("at least one structural relation channel is required")
    if source_adjacency.shape[1:] != (plan.shape[0], plan.shape[0]):
        raise ValueError("source adjacency shape does not match the transport plan")
    if target_adjacency.shape[1:] != (plan.shape[1], plan.shape[1]):
        raise ValueError("target adjacency shape does not match the transport plan")
    if source_adjacency.shape[0] != target_adjacency.shape[0]:
        raise ValueError("structural adjacency relation counts must match")
    return tuple(
        (
            sparse.csr_matrix(source_relation, dtype=np.float64),
            sparse.csr_matrix(target_relation, dtype=np.float64),
        )
        for source_relation, target_relation in zip(
            source_adjacency,
            target_adjacency,
            strict=True,
        )
    )


def _prepared_structural_linear_cost(
    plan: np.ndarray,
    prepared: PreparedStructure,
) -> np.ndarray:
    if prepared is None:
        return np.zeros_like(plan)
    source_mass = plan.sum(axis=1)
    target_mass = plan.sum(axis=0)
    linear = np.zeros_like(plan)
    for source_relation, target_relation in prepared:
        source_square = source_relation.multiply(source_relation) @ source_mass
        target_square = target_relation.multiply(target_relation) @ target_mass
        cross = (target_relation @ (source_relation @ plan).T).T
        linear += source_square[:, None] + target_square[None, :] - 2.0 * cross
    return linear / len(prepared)


def _structural_linear_cost(
    plan: np.ndarray,
    source_adjacency: np.ndarray | None,
    target_adjacency: np.ndarray | None,
) -> np.ndarray:
    prepared = _prepare_structure(plan, source_adjacency, target_adjacency)
    return _prepared_structural_linear_cost(plan, prepared)


def _objective_components(
    *,
    plan: np.ndarray,
    node_cost: np.ndarray,
    prepared_structure: PreparedStructure,
    soft_constraint_cost: np.ndarray,
    epsilon: float,
    lambda_structure: float,
    lambda_constraint: float,
) -> ObjectiveComponents:
    positive = plan > 0.0
    negative_entropy = float(np.sum(plan[positive] * (np.log(plan[positive]) - 1.0)))
    node = float(np.sum(plan * node_cost))
    structural_linear = _prepared_structural_linear_cost(plan, prepared_structure)
    structural = float(np.sum(plan * structural_linear))
    soft_value = float(np.sum(plan * soft_constraint_cost))
    total = (
        node
        + lambda_structure * structural
        + lambda_constraint * soft_value
        + epsilon * negative_entropy
    )
    return ObjectiveComponents(
        node=node,
        structural=structural,
        soft_constraint=soft_value,
        negative_entropy=negative_entropy,
        total=float(total),
    )


def transport_objective(
    *,
    plan: np.ndarray,
    node_cost: np.ndarray,
    source_adjacency: np.ndarray | None,
    target_adjacency: np.ndarray | None,
    soft_constraint_cost: np.ndarray | None,
    epsilon: float,
    lambda_structure: float,
    lambda_constraint: float,
) -> ObjectiveComponents:
    if lambda_structure < 0.0 or lambda_constraint < 0.0:
        raise ValueError("transport penalty weights must be nonnegative")
    soft = np.zeros_like(plan) if soft_constraint_cost is None else soft_constraint_cost
    return _objective_components(
        plan=plan,
        node_cost=node_cost,
        prepared_structure=_prepare_structure(
            plan,
            source_adjacency,
            target_adjacency,
        ),
        soft_constraint_cost=soft,
        epsilon=epsilon,
        lambda_structure=lambda_structure,
        lambda_constraint=lambda_constraint,
    )


def _mask_is_feasible(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    mask: np.ndarray,
    tolerance: float,
) -> bool:
    if np.any((source_mass > tolerance) & ~mask.any(axis=1)):
        return False
    if np.any((target_mass > tolerance) & ~mask.any(axis=0)):
        return False
    graph = nx.DiGraph()
    total = float(source_mass.sum())
    for row, mass in enumerate(source_mass):
        graph.add_edge("source", f"row:{row}", capacity=float(mass))
    for column, mass in enumerate(target_mass):
        graph.add_edge(f"column:{column}", "sink", capacity=float(mass))
    for row, column in zip(*np.where(mask)):
        graph.add_edge(f"row:{row}", f"column:{column}", capacity=total)
    flow, _ = nx.maximum_flow(graph, "source", "sink")
    return flow >= total - tolerance


def _validate_inputs(
    node_cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    hard_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cost = np.asarray(node_cost, dtype=np.float64)
    source = np.asarray(source_mass, dtype=np.float64)
    target = np.asarray(target_mass, dtype=np.float64)
    mask = np.asarray(hard_mask, dtype=np.bool_)
    if cost.ndim != 2 or mask.shape != cost.shape:
        raise ValueError("node cost and mask must have the same matrix shape")
    if source.shape != (cost.shape[0],) or target.shape != (cost.shape[1],):
        raise ValueError("mass vectors do not match the cost matrix")
    if not np.all(np.isfinite(cost)) or np.any(cost < 0.0):
        raise ValueError("node costs must be finite and nonnegative")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("masses must be finite")
    if np.any(source < 0.0) or np.any(target < 0.0):
        raise ValueError("masses must be nonnegative")
    if not math.isclose(float(source.sum()), float(target.sum()), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("source and target masses must have equal total mass")
    if source.sum() <= 0.0:
        raise ValueError("total mass must be positive")
    return cost, source, target, mask


def solve_transport(
    *,
    node_cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    hard_mask: np.ndarray,
    epsilon: float,
    max_iterations: int,
    marginal_tolerance: float,
    objective_tolerance: float,
    stable_iterations: int,
    source_adjacency: np.ndarray | None = None,
    target_adjacency: np.ndarray | None = None,
    soft_constraint_cost: np.ndarray | None = None,
    lambda_structure: float = 0.0,
    lambda_constraint: float = 0.0,
) -> TransportResult:
    cost, source, target, mask = _validate_inputs(
        node_cost,
        source_mass,
        target_mass,
        hard_mask,
    )
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if max_iterations < 1 or stable_iterations < 1:
        raise ValueError("iteration controls must be positive")
    if marginal_tolerance < 0.0 or objective_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    soft = (
        np.zeros_like(cost)
        if soft_constraint_cost is None
        else np.asarray(soft_constraint_cost, dtype=np.float64)
    )
    if soft.shape != cost.shape or np.any(soft < 0.0) or not np.all(np.isfinite(soft)):
        raise ValueError("soft constraint cost must be a finite nonnegative cost matrix")
    if not _mask_is_feasible(source, target, mask, marginal_tolerance):
        return TransportResult(
            plan=np.zeros_like(cost),
            objective=None,
            diagnostics=TransportDiagnostics(
                feasible=False,
                converged=False,
                iterations=0,
                marginal_error=math.inf,
                objective_change=None,
                consecutive_stable_iterations=0,
                reason="infeasible_hard_mask",
            ),
        )
    prepared_structure = _prepare_structure(
        cost,
        source_adjacency,
        target_adjacency,
    )

    plan = np.outer(source, target)
    plan[~mask] = 0.0
    plan /= plan.sum()
    log_source = np.full_like(source, -np.inf)
    log_target = np.full_like(target, -np.inf)
    active_source = source > 0.0
    active_target = target > 0.0
    log_source[active_source] = np.log(source[active_source])
    log_target[active_target] = np.log(target[active_target])
    log_u = np.full_like(source, -np.inf)
    log_v = np.full_like(target, -np.inf)
    log_u[active_source] = 0.0
    log_v[active_target] = 0.0
    previous_total: float | None = None
    stable = 0
    objective_change: float | None = None
    components: ObjectiveComponents | None = None
    marginal_error = math.inf

    for iteration in range(1, max_iterations + 1):
        structural_linear = _prepared_structural_linear_cost(plan, prepared_structure)
        effective_cost = (
            cost
            + lambda_structure * structural_linear
            + lambda_constraint * soft
        )
        log_kernel = masked_log_kernel(effective_cost, mask, epsilon)
        row_normalizer = logsumexp(log_kernel + log_v[None, :], axis=1)
        log_u.fill(-np.inf)
        log_u[active_source] = log_source[active_source] - row_normalizer[active_source]
        column_normalizer = logsumexp(log_kernel + log_u[:, None], axis=0)
        log_v.fill(-np.inf)
        log_v[active_target] = (
            log_target[active_target] - column_normalizer[active_target]
        )
        log_plan = log_kernel + log_u[:, None] + log_v[None, :]
        plan = np.zeros_like(cost)
        plan[mask] = np.exp(log_plan[mask])

        row_error = float(np.max(np.abs(plan.sum(axis=1) - source)))
        column_error = float(np.max(np.abs(plan.sum(axis=0) - target)))
        marginal_error = max(row_error, column_error)
        components = _objective_components(
            plan=plan,
            node_cost=cost,
            prepared_structure=prepared_structure,
            soft_constraint_cost=soft,
            epsilon=epsilon,
            lambda_structure=lambda_structure,
            lambda_constraint=lambda_constraint,
        )
        if previous_total is not None:
            objective_change = abs(components.total - previous_total)
            if (
                marginal_error <= marginal_tolerance
                and objective_change <= objective_tolerance
            ):
                stable += 1
            else:
                stable = 0
        previous_total = components.total
        if stable >= stable_iterations:
            return TransportResult(
                plan=plan,
                objective=components,
                diagnostics=TransportDiagnostics(
                    feasible=True,
                    converged=True,
                    iterations=iteration,
                    marginal_error=marginal_error,
                    objective_change=objective_change,
                    consecutive_stable_iterations=stable,
                    reason="converged",
                ),
            )

    return TransportResult(
        plan=plan,
        objective=components,
        diagnostics=TransportDiagnostics(
            feasible=True,
            converged=False,
            iterations=max_iterations,
            marginal_error=marginal_error,
            objective_change=objective_change,
            consecutive_stable_iterations=stable,
            reason="maximum_iterations_reached",
        ),
    )
