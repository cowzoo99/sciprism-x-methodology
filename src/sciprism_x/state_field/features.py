from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sciprism_x.residuals.schema import ResidualVector

STATE_FEATURE_NAMES = (
    "pcs",
    "residual_consistency",
    "node_loss",
    "structure_loss",
    "constraint_loss",
    "transport_entropy",
    "motif_signature_discrepancy",
    "residual_physical",
    "residual_logical",
    "residual_unit_dimensional",
    "residual_causal",
    "residual_modal_evidential",
    "residual_uncertainty",
    "task_graph_complexity",
    "response_graph_complexity",
)


@dataclass(frozen=True, slots=True)
class StateVector:
    feature_names: tuple[str, ...]
    values: np.ndarray


def build_state_vector(
    *,
    pcs: float,
    residual_consistency: float,
    node_loss: float,
    structure_loss: float,
    constraint_loss: float,
    transport_entropy: float,
    motif_signature_discrepancy: float,
    residuals: ResidualVector,
    task_graph_complexity: float,
    response_graph_complexity: float,
) -> StateVector:
    values = np.array(
        [
            pcs,
            residual_consistency,
            node_loss,
            structure_loss,
            constraint_loss,
            transport_entropy,
            motif_signature_discrepancy,
            *residuals.as_array().tolist(),
            task_graph_complexity,
            response_graph_complexity,
        ],
        dtype=np.float64,
    )
    if values.shape != (15,):
        raise ValueError("reasoning-state vector must contain 15 features")
    if not np.all(np.isfinite(values)):
        raise ValueError("reasoning-state features must be finite")
    return StateVector(feature_names=STATE_FEATURE_NAMES, values=values)
