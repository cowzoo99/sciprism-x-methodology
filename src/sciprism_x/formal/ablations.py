from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Literal, Protocol

import numpy as np
from scipy.optimize import linprog


class AblationKind(str, Enum):
    WITHOUT_MOTIF_STRUCTURE = "without_motif_structure"
    WITHOUT_HARD_CONSTRAINTS = "without_hard_constraints"
    WITHOUT_SOFT_PENALTIES = "without_soft_penalties"
    WITHOUT_RESIDUAL_CHANNELS = "without_residual_channels"
    WITHOUT_STATE_FIELD = "without_state_field"
    WITHOUT_BOUNDARY_ESTIMATION = "without_boundary_estimation"
    WITHOUT_OPEN_SET = "without_open_set"
    WITHOUT_TRANSPORT = "without_transport"


@dataclass(frozen=True, slots=True)
class AblationControls:
    motif_structure: bool = True
    hard_constraints: bool = True
    soft_penalties: bool = True
    residual_channels: bool = True
    state_field: bool = True
    boundary_estimation: bool = True
    open_set: bool = True
    transport_mode: Literal["optimal", "greedy"] = "optimal"


@dataclass(frozen=True, slots=True)
class AblationPolicy:
    kind: AblationKind
    controls: AblationControls
    recompute_stages: tuple[str, ...]


_MAIN_CONTROLS = AblationControls()
_PIPELINE = (
    "motif_graph",
    "transport",
    "residuals",
    "state_field",
    "boundary",
    "open_set",
    "metrics",
)


def _downstream(stage: str) -> tuple[str, ...]:
    return _PIPELINE[_PIPELINE.index(stage) :]


_POLICIES = {
    AblationKind.WITHOUT_MOTIF_STRUCTURE: AblationPolicy(
        AblationKind.WITHOUT_MOTIF_STRUCTURE,
        replace(_MAIN_CONTROLS, motif_structure=False),
        _downstream("motif_graph"),
    ),
    AblationKind.WITHOUT_HARD_CONSTRAINTS: AblationPolicy(
        AblationKind.WITHOUT_HARD_CONSTRAINTS,
        replace(_MAIN_CONTROLS, hard_constraints=False),
        _downstream("transport"),
    ),
    AblationKind.WITHOUT_SOFT_PENALTIES: AblationPolicy(
        AblationKind.WITHOUT_SOFT_PENALTIES,
        replace(_MAIN_CONTROLS, soft_penalties=False),
        _downstream("transport"),
    ),
    AblationKind.WITHOUT_RESIDUAL_CHANNELS: AblationPolicy(
        AblationKind.WITHOUT_RESIDUAL_CHANNELS,
        replace(_MAIN_CONTROLS, residual_channels=False),
        _downstream("residuals"),
    ),
    AblationKind.WITHOUT_STATE_FIELD: AblationPolicy(
        AblationKind.WITHOUT_STATE_FIELD,
        replace(_MAIN_CONTROLS, state_field=False),
        _downstream("state_field"),
    ),
    AblationKind.WITHOUT_BOUNDARY_ESTIMATION: AblationPolicy(
        AblationKind.WITHOUT_BOUNDARY_ESTIMATION,
        replace(_MAIN_CONTROLS, boundary_estimation=False),
        _downstream("boundary"),
    ),
    AblationKind.WITHOUT_OPEN_SET: AblationPolicy(
        AblationKind.WITHOUT_OPEN_SET,
        replace(_MAIN_CONTROLS, open_set=False),
        _downstream("open_set"),
    ),
    AblationKind.WITHOUT_TRANSPORT: AblationPolicy(
        AblationKind.WITHOUT_TRANSPORT,
        replace(_MAIN_CONTROLS, transport_mode="greedy"),
        _downstream("transport"),
    ),
}


def resolve_ablation_policy(kind: AblationKind) -> AblationPolicy:
    return _POLICIES[kind]


@dataclass(frozen=True, slots=True)
class PerCaseMetric:
    case_id: str
    dataset: str
    metric: str
    value: float


class AblationEvaluator(Protocol):
    resolved_controls: AblationControls

    def evaluate(self, bundle: object) -> Sequence[PerCaseMetric]: ...


EvaluatorFactory = Callable[[AblationKind | None], AblationEvaluator]


@dataclass(frozen=True, slots=True)
class AblationRow:
    case_id: str
    dataset: str
    ablation: str
    metric: str
    main_value: float
    metric_value: float
    delta: float
    source: Literal["computed"] = "computed"


def _index_metrics(
    rows: Sequence[PerCaseMetric],
    *,
    label: str,
) -> dict[tuple[str, str, str], float]:
    indexed: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row.case_id, row.dataset, row.metric)
        if key in indexed:
            raise ValueError(f"{label} has duplicate per-case metric: {key}")
        if not math.isfinite(row.value):
            raise ValueError(f"{label} has non-finite per-case metric: {key}")
        indexed[key] = float(row.value)
    if not indexed:
        raise ValueError(f"{label} produced no per-case metrics")
    return indexed


def run_formal_ablations(
    bundle: object,
    evaluator_factory: EvaluatorFactory,
) -> tuple[AblationRow, ...]:
    main_evaluator = evaluator_factory(None)
    if main_evaluator.resolved_controls != _MAIN_CONTROLS:
        raise ValueError("main evaluator does not use the complete method controls")
    main = _index_metrics(main_evaluator.evaluate(bundle), label="main method")

    output: list[AblationRow] = []
    for kind in AblationKind:
        policy = resolve_ablation_policy(kind)
        evaluator = evaluator_factory(kind)
        if evaluator.resolved_controls != policy.controls:
            raise ValueError(f"{kind.value} evaluator controls differ from its policy")
        ablated = _index_metrics(evaluator.evaluate(bundle), label=kind.value)
        if set(ablated) != set(main):
            missing = sorted(set(main) - set(ablated))
            extra = sorted(set(ablated) - set(main))
            raise ValueError(
                f"{kind.value} is not paired with main outcomes; "
                f"missing={missing}, extra={extra}"
            )
        output.extend(
            AblationRow(
                case_id=case_id,
                dataset=dataset,
                ablation=kind.value,
                metric=metric,
                main_value=main[(case_id, dataset, metric)],
                metric_value=ablated[(case_id, dataset, metric)],
                delta=(
                    main[(case_id, dataset, metric)]
                    - ablated[(case_id, dataset, metric)]
                ),
            )
            for case_id, dataset, metric in sorted(main)
        )
    return tuple(output)


def _maximum_feasible_candidate_mass(
    *,
    candidates: Sequence[tuple[float, int, int]],
    remaining_source: np.ndarray,
    remaining_target: np.ndarray,
    lp_method: str,
    lp_presolve: bool,
    lp_primal_feasibility_tolerance: float,
    lp_dual_feasibility_tolerance: float,
    lp_edge_weight_strategy: str,
) -> float:
    variable_count = len(candidates)
    row_count = remaining_source.size
    column_count = remaining_target.size
    equality = np.zeros(
        (row_count + column_count, variable_count),
        dtype=np.float64,
    )
    for index, (_, row, column) in enumerate(candidates):
        equality[row, index] = 1.0
        equality[row_count + column, index] = 1.0
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[0] = -1.0
    result = linprog(
        c=objective,
        A_eq=equality,
        b_eq=np.concatenate((remaining_source, remaining_target)),
        bounds=(0.0, None),
        method=lp_method,
        options={
            "presolve": lp_presolve,
            "primal_feasibility_tolerance": lp_primal_feasibility_tolerance,
            "dual_feasibility_tolerance": lp_dual_feasibility_tolerance,
            "simplex_dual_edge_weight_strategy": lp_edge_weight_strategy,
        },
    )
    if not result.success or result.x is None:
        raise ValueError("typed compatible graph cannot satisfy the declared marginals")
    return float(result.x[0])


def typed_greedy_minimum_cost_matching(
    *,
    node_cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    compatible: np.ndarray,
    tolerance: float,
    numerical_tolerance: float,
    lp_method: str,
    lp_presolve: bool,
    lp_primal_feasibility_tolerance: float,
    lp_dual_feasibility_tolerance: float,
    lp_edge_weight_strategy: str,
) -> np.ndarray:
    costs = np.asarray(node_cost, dtype=np.float64)
    source = np.asarray(source_mass, dtype=np.float64)
    target = np.asarray(target_mass, dtype=np.float64)
    mask = np.asarray(compatible, dtype=np.bool_)
    if costs.ndim != 2 or costs.size == 0:
        raise ValueError("node_cost must be a non-empty matrix")
    if source.shape != (costs.shape[0],) or target.shape != (costs.shape[1],):
        raise ValueError("mass vectors do not match node_cost")
    if mask.shape != costs.shape:
        raise ValueError("compatible mask does not match node_cost")
    if (
        not np.isfinite(costs).all()
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("greedy matching inputs must be finite")
    if np.any(source < 0.0) or np.any(target < 0.0):
        raise ValueError("greedy matching masses must be nonnegative")
    if (
        tolerance <= 0.0
        or numerical_tolerance <= 0.0
        or not math.isfinite(tolerance)
        or not math.isfinite(numerical_tolerance)
    ):
        raise ValueError("greedy tolerances must be finite and positive")
    if numerical_tolerance < tolerance:
        raise ValueError("numerical_tolerance must not be smaller than mass tolerance")
    if not math.isclose(
        float(source.sum()),
        float(target.sum()),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("source and target masses must have equal totals")

    remaining_source = source.copy()
    remaining_target = target.copy()
    plan = np.zeros_like(costs, dtype=np.float64)
    candidates = sorted(
        (
            (float(costs[row, column]), row, column)
            for row, column in zip(*np.where(mask), strict=True)
        ),
        key=lambda value: (value[0], value[1], value[2]),
    )
    for index, (_, row, column) in enumerate(candidates):
        if (
            remaining_source[row] <= tolerance
            or remaining_target[column] <= tolerance
        ):
            continue
        mass = _maximum_feasible_candidate_mass(
            candidates=candidates[index:],
            remaining_source=remaining_source,
            remaining_target=remaining_target,
            lp_method=lp_method,
            lp_presolve=lp_presolve,
            lp_primal_feasibility_tolerance=lp_primal_feasibility_tolerance,
            lp_dual_feasibility_tolerance=lp_dual_feasibility_tolerance,
            lp_edge_weight_strategy=lp_edge_weight_strategy,
        )
        mass = min(mass, remaining_source[row], remaining_target[column])
        if mass <= tolerance:
            continue
        plan[row, column] += mass
        remaining_source[row] -= mass
        remaining_target[column] -= mass

    remaining_source[np.abs(remaining_source) <= numerical_tolerance] = 0.0
    remaining_target[np.abs(remaining_target) <= numerical_tolerance] = 0.0
    if np.any(remaining_source != 0.0) or np.any(remaining_target != 0.0):
        raise ValueError("typed greedy matching cannot satisfy the declared marginals")
    plan.flags.writeable = False
    return plan
