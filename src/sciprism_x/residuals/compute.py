from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from sciprism_x.motifs.schema import MotifFamily, MotifNode, TypedReasoningGraph
from sciprism_x.transport.schema import TransportResult
from sciprism_x.validators.schema import ValidationResult, ValidationStatus

from .schema import ResidualChannel, ResidualComponent, ResidualVector


_MOTIF_CHANNEL = {
    MotifFamily.SYMBOLIC_TRANSFORMATION: ResidualChannel.LOGICAL,
    MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY: ResidualChannel.UNIT_DIMENSIONAL,
    MotifFamily.CAUSAL_MECHANISTIC_RELATION: ResidualChannel.CAUSAL,
    MotifFamily.EVIDENCE_BINDING: ResidualChannel.MODAL_EVIDENTIAL,
    MotifFamily.MODALITY_GROUNDING: ResidualChannel.MODAL_EVIDENTIAL,
    MotifFamily.UNCERTAINTY_QUALIFICATION: ResidualChannel.UNCERTAINTY,
}


def _reason_channel(result: ValidationResult, task_node: MotifNode) -> ResidualChannel:
    reason = result.reason_code.lower()
    if "physical" in reason or "boundary" in reason:
        return ResidualChannel.PHYSICAL
    if "unit" in reason or "dimension" in reason:
        return ResidualChannel.UNIT_DIMENSIONAL
    if "causal" in reason or "mechanism" in reason:
        return ResidualChannel.CAUSAL
    if any(token in reason for token in ("evidence", "modal", "anchor", "ground")):
        return ResidualChannel.MODAL_EVIDENTIAL
    if "uncert" in reason or "qualifier" in reason:
        return ResidualChannel.UNCERTAINTY
    if "logic" in reason or "symbol" in reason or "contrad" in reason:
        return ResidualChannel.LOGICAL
    return _MOTIF_CHANNEL[task_node.motif_family]


def _validate_matrix(
    validations: Sequence[Sequence[ValidationResult]],
    task_count: int,
    response_count: int,
) -> None:
    if len(validations) != task_count or any(
        len(row) != response_count for row in validations
    ):
        raise ValueError("validation matrix shape does not match graph nodes")


def compute_residuals(
    *,
    transport: TransportResult,
    task_graph: TypedReasoningGraph,
    response_graph: TypedReasoningGraph,
    validations: Sequence[Sequence[ValidationResult]],
    omega_missing: float,
    omega_unsupported: float,
    minimum_support: float,
    normalizers: Mapping[ResidualChannel, float],
    normalizer_floor: float,
) -> ResidualVector:
    if transport.objective is None or not transport.diagnostics.feasible:
        raise ValueError("residuals require a feasible transport result")
    if omega_missing < 0.0 or omega_unsupported < 0.0:
        raise ValueError("residual weights must be nonnegative")
    if not 0.0 <= minimum_support <= 1.0:
        raise ValueError("minimum_support must be in [0, 1]")
    if normalizer_floor <= 0.0:
        raise ValueError("normalizer_floor must be positive")
    if set(normalizers) != set(ResidualChannel):
        raise ValueError("normalizers must cover all six residual channels")
    if any(value < 0.0 for value in normalizers.values()):
        raise ValueError("normalizers must be nonnegative")

    plan = np.asarray(transport.plan, dtype=np.float64)
    expected_shape = (len(task_graph.nodes), len(response_graph.nodes))
    if plan.shape != expected_shape:
        raise ValueError("transport plan shape does not match graph nodes")
    _validate_matrix(validations, *expected_shape)

    matched = {channel: 0.0 for channel in ResidualChannel}
    missing = {channel: 0.0 for channel in ResidualChannel}
    unsupported = {channel: 0.0 for channel in ResidualChannel}

    for task_index, task_node in enumerate(task_graph.nodes):
        row_mass = float(plan[task_index].sum())
        pass_mass = 0.0
        for response_index, result in enumerate(validations[task_index]):
            pair_mass = float(plan[task_index, response_index])
            if result.status is ValidationStatus.FAIL:
                channel = _reason_channel(result, task_node)
                matched[channel] += pair_mass * result.confidence
            elif result.status is ValidationStatus.PASS:
                pass_mass += pair_mass
        support = 1.0 if row_mass == 0.0 else pass_mass / row_mass
        if support < minimum_support:
            missing[_MOTIF_CHANNEL[task_node.motif_family]] += row_mass

    for response_index, response_node in enumerate(response_graph.nodes):
        structurally_supported = any(
            task_node.motif_family is response_node.motif_family
            and plan[task_index, response_index] > 0.0
            for task_index, task_node in enumerate(task_graph.nodes)
        )
        if not structurally_supported:
            unsupported[_MOTIF_CHANNEL[response_node.motif_family]] += float(
                plan[:, response_index].sum()
            )

    components: dict[ResidualChannel, ResidualComponent] = {}
    for channel in ResidualChannel:
        raw = (
            matched[channel]
            + omega_missing * missing[channel]
            + omega_unsupported * unsupported[channel]
        )
        normalizer = max(float(normalizers[channel]), normalizer_floor)
        components[channel] = ResidualComponent(
            matched=matched[channel],
            missing=missing[channel],
            unsupported=unsupported[channel],
            raw=raw,
            normalizer=normalizer,
            normalized=raw / normalizer,
        )
    return ResidualVector(components=components)
