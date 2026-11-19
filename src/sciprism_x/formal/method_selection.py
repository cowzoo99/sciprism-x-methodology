from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from sciprism_x.config import RunConfig
from sciprism_x.data.schema import SciCUCase
from sciprism_x.residuals.schema import ResidualChannel
from sciprism_x.responses.schema import ResponseRecord
from sciprism_x.tuning import build_trials

from .evaluator import (
    FormalEvaluationContext,
    MethodParameters,
    evaluate_scicu_components,
    evaluate_scicu_response,
)


@dataclass(frozen=True, slots=True)
class MethodSelectionExample:
    case: SciCUCase
    response: ResponseRecord
    structural_validity: bool


@dataclass(frozen=True, slots=True)
class MethodTrialAudit:
    trial_index: int
    parameters: MethodParameters
    loss_normalizers: dict[str, float]
    validation_brier: float
    feasible_rate: float


@dataclass(frozen=True, slots=True)
class MethodSelectionResult:
    selected_trial: int
    trial_count: int
    parameters: MethodParameters
    loss_normalizers: dict[str, float]
    residual_normalizers: dict[ResidualChannel, float]
    trials: tuple[MethodTrialAudit, ...]


def _parameters(trial: dict[str, float]) -> MethodParameters:
    return MethodParameters(**{key: float(value) for key, value in trial.items()})


def _validity_probability(pcs: float, consistency: float, feasible: bool) -> float:
    if not feasible or not math.isfinite(pcs) or not math.isfinite(consistency):
        return 0.0
    evidence = max(pcs, 0.0) * min(max(consistency, 0.0), 1.0)
    return evidence / (1.0 + evidence)


def _fit_normalizers(
    examples: Sequence[MethodSelectionExample],
    context: FormalEvaluationContext,
) -> dict[ResidualChannel, float]:
    raw = {channel: [] for channel in ResidualChannel}
    for example in examples:
        components = evaluate_scicu_components(
            case=example.case,
            response=example.response,
            context=context,
        )
        if components.residuals is None:
            continue
        for channel in ResidualChannel:
            raw[channel].append(components.residuals.components[channel].raw)
    percentile = context.config.residuals.normalizer_percentile
    floor = context.config.residuals.normalizer_floor
    return {
        channel: max(
            float(np.quantile(values, percentile, method="linear")) if values else floor,
            floor,
        )
        for channel, values in raw.items()
    }


def _fit_loss_normalizers(
    examples: Sequence[MethodSelectionExample],
    context: FormalEvaluationContext,
) -> dict[str, float]:
    raw: dict[str, list[float]] = {
        "node": [],
        "structure": [],
        "constraint": [],
    }
    for example in examples:
        components = evaluate_scicu_components(
            case=example.case,
            response=example.response,
            context=context,
        )
        objective = components.transport.objective
        if objective is None or not components.transport.diagnostics.feasible:
            continue
        raw["node"].append(objective.node)
        raw["structure"].append(objective.structural)
        raw["constraint"].append(objective.soft_constraint)
    percentile = context.config.transport.loss_normalizer_percentile
    floor = context.config.transport.loss_normalizer_floor
    return {
        name: max(
            float(np.quantile(values, percentile, method="linear"))
            if values
            else floor,
            floor,
        )
        for name, values in raw.items()
    }


def select_method_parameters(
    examples: Sequence[MethodSelectionExample],
    context: FormalEvaluationContext,
    config: RunConfig,
    *,
    seed: int,
) -> MethodSelectionResult:
    """Select all method coefficients on validation data, then freeze normalizers."""

    if not examples:
        raise ValueError("method selection requires validation examples")
    trials = build_trials(
        {
            "epsilon": config.transport.epsilon_grid,
            "eta_constraint": config.residuals.pcs_coefficient_grid,
            "eta_node": config.residuals.pcs_coefficient_grid,
            "eta_structure": config.residuals.pcs_coefficient_grid,
            "lambda_constraint": config.transport.lambda_constraint_grid,
            "lambda_structure": config.transport.lambda_structure_grid,
            "minimum_support": config.residuals.gamma_grid,
            "omega_missing": config.residuals.missing_penalty_grid,
            "omega_unsupported": config.residuals.unsupported_penalty_grid,
        },
        count=config.paper.hyperparameter_trials,
        seed=seed,
    )
    unit_normalizers = {channel: 1.0 for channel in ResidualChannel}
    audits = []
    for trial_index, trial in enumerate(trials):
        parameters = _parameters(trial)
        trial_context = replace(
            context,
            parameters=parameters,
            loss_normalizers={
                "node": 1.0,
                "structure": 1.0,
                "constraint": 1.0,
            },
            residual_normalizers=unit_normalizers,
        )
        loss_normalizers = _fit_loss_normalizers(examples, trial_context)
        trial_context = replace(
            trial_context,
            loss_normalizers=loss_normalizers,
        )
        rows = tuple(
            evaluate_scicu_response(
                case=example.case,
                response=example.response,
                context=trial_context,
            )
            for example in examples
        )
        probabilities = np.asarray(
            [
                _validity_probability(
                    row.pcs,
                    row.residual_consistency,
                    row.hard_feasible,
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        labels = np.asarray(
            [float(example.structural_validity) for example in examples],
            dtype=np.float64,
        )
        audits.append(
            MethodTrialAudit(
                trial_index=trial_index,
                parameters=parameters,
                loss_normalizers=loss_normalizers,
                validation_brier=float(np.mean(np.square(probabilities - labels))),
                feasible_rate=float(np.mean([row.hard_feasible for row in rows])),
            )
        )
    selected = min(
        audits,
        key=lambda item: (
            item.validation_brier,
            -item.feasible_rate,
            item.trial_index,
        ),
    )
    selected_context = replace(
        context,
        parameters=selected.parameters,
        loss_normalizers=selected.loss_normalizers,
        residual_normalizers=unit_normalizers,
    )
    return MethodSelectionResult(
        selected_trial=selected.trial_index,
        trial_count=len(trials),
        parameters=selected.parameters,
        loss_normalizers=selected.loss_normalizers,
        residual_normalizers=_fit_normalizers(examples, selected_context),
        trials=tuple(audits),
    )
