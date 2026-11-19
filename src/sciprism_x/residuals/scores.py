from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sciprism_x.transport.schema import ObjectiveComponents

from .schema import ResidualVector, ScoreBundle


def compute_tsr(endpoint_success: Sequence[bool]) -> float:
    if not endpoint_success:
        raise ValueError("at least one endpoint result is required")
    return sum(bool(value) for value in endpoint_success) / len(endpoint_success)


def compute_pcs(
    *,
    node_loss: float,
    structure_loss: float,
    constraint_loss: float,
    eta_node: float,
    eta_structure: float,
    eta_constraint: float,
) -> float:
    """Literal manuscript formula: exp(+η1 L_node - η2 L_str - η3 L_con)."""

    return math.exp(
        eta_node * node_loss
        - eta_structure * structure_loss
        - eta_constraint * constraint_loss
    )


def compute_residual_consistency(
    residual: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Literal manuscript formula: exp(βᵀ residual)."""

    residual_vector = np.asarray(residual, dtype=np.float64)
    coefficients = np.asarray(beta, dtype=np.float64)
    if residual_vector.shape != (6,) or coefficients.shape != (6,):
        raise ValueError("residual and beta must each have six entries")
    return math.exp(float(coefficients @ residual_vector))


def direction_audit(
    *,
    eta_node: float,
    eta_structure: float,
    eta_constraint: float,
    beta: np.ndarray,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if eta_node > 0.0:
        warnings.append("pcs_node_loss_increases_score")
    if eta_structure < 0.0:
        warnings.append("negative_eta_structure_increases_score_with_structure_loss")
    if eta_constraint < 0.0:
        warnings.append("negative_eta_constraint_increases_score_with_constraint_loss")
    coefficients = np.asarray(beta, dtype=np.float64)
    if coefficients.shape != (6,):
        raise ValueError("beta must have six entries")
    if np.any(coefficients > 0.0):
        warnings.append("positive_beta_increases_consistency_with_residual")
    return tuple(warnings)


def build_score_bundle(
    *,
    endpoint_success: Sequence[bool],
    transport_objective: ObjectiveComponents,
    residuals: ResidualVector,
    eta_node: float,
    eta_structure: float,
    eta_constraint: float,
    beta: np.ndarray,
) -> ScoreBundle:
    warnings = direction_audit(
        eta_node=eta_node,
        eta_structure=eta_structure,
        eta_constraint=eta_constraint,
        beta=beta,
    )
    return ScoreBundle(
        tsr=compute_tsr(endpoint_success),
        pcs=compute_pcs(
            node_loss=transport_objective.node,
            structure_loss=transport_objective.structural,
            constraint_loss=transport_objective.soft_constraint,
            eta_node=eta_node,
            eta_structure=eta_structure,
            eta_constraint=eta_constraint,
        ),
        residual_consistency=compute_residual_consistency(residuals.as_array(), beta),
        node_loss=transport_objective.node,
        structure_loss=transport_objective.structural,
        constraint_loss=transport_objective.soft_constraint,
        direction_warnings=warnings,
    )
