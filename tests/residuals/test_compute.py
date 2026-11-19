import numpy as np


def _node(node_id, text, family):
    from sciprism_x.motifs.schema import MotifNode

    return MotifNode(
        node_id=node_id,
        text=text,
        motif_family=family,
        confidence=1.0,
        salience=0.5,
        source_span=(0, len(text)),
        variables=(),
        feature_evidence={},
    )


def _graph(source_id, side, nodes):
    from sciprism_x.motifs.schema import TypedReasoningGraph

    return TypedReasoningGraph(source_id=source_id, side=side, nodes=nodes, edges=())


def _transport(plan):
    from sciprism_x.transport.schema import (
        ObjectiveComponents,
        TransportDiagnostics,
        TransportResult,
    )

    return TransportResult(
        plan=np.asarray(plan, dtype=np.float64),
        objective=ObjectiveComponents(
            node=0.0,
            structural=0.0,
            soft_constraint=0.0,
            negative_entropy=0.0,
            total=0.0,
        ),
        diagnostics=TransportDiagnostics(
            feasible=True,
            converged=True,
            iterations=1,
            marginal_error=0.0,
            objective_change=0.0,
            consecutive_stable_iterations=5,
            reason="converged",
        ),
    )


def test_residuals_preserve_matched_missing_and_unsupported_terms() -> None:
    from sciprism_x.motifs.schema import MotifFamily
    from sciprism_x.residuals.compute import compute_residuals
    from sciprism_x.residuals.schema import ResidualChannel
    from sciprism_x.validators.schema import ValidationResult

    task = _graph(
        "task",
        "task",
        (
            _node("T1", "The result is 12 m/s.", MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY),
            _node(
                "T2",
                "Under ideal conditions, report approximately.",
                MotifFamily.UNCERTAINTY_QUALIFICATION,
            ),
        ),
    )
    response = _graph(
        "response",
        "response",
        (
            _node("R1", "The result is 12 kg.", MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY),
            _node("R2", "An unsupported table claim.", MotifFamily.EVIDENCE_BINDING),
        ),
    )
    validations = (
        (
            ValidationResult.fail("unit_dimension_mismatch", confidence=0.8),
            ValidationResult.unresolved("not_applicable", confidence=0.0),
        ),
        (
            ValidationResult.unresolved("not_applicable", confidence=0.0),
            ValidationResult.fail("uncertainty_qualifier_missing", confidence=1.0),
        ),
    )
    normalizers = {channel: 2.0 for channel in ResidualChannel}

    result = compute_residuals(
        transport=_transport([[0.4, 0.1], [0.1, 0.4]]),
        task_graph=task,
        response_graph=response,
        validations=validations,
        omega_missing=2.0,
        omega_unsupported=3.0,
        minimum_support=0.5,
        normalizers=normalizers,
        normalizer_floor=1.0e-12,
    )

    unit = result.components[ResidualChannel.UNIT_DIMENSIONAL]
    uncertainty = result.components[ResidualChannel.UNCERTAINTY]
    modal = result.components[ResidualChannel.MODAL_EVIDENTIAL]
    assert unit.matched == 0.4 * 0.8
    assert unit.missing == 0.5
    assert unit.unsupported == 0.0
    assert uncertainty.matched == 0.4
    assert uncertainty.missing == 0.5
    assert modal.unsupported == 0.5
    assert unit.raw == unit.matched + 2.0 * unit.missing
    assert unit.normalized == unit.raw / 2.0
    assert result.as_array().shape == (6,)


def test_ambiguous_validator_outputs_do_not_create_matched_violations() -> None:
    from sciprism_x.motifs.schema import MotifFamily
    from sciprism_x.residuals.compute import compute_residuals
    from sciprism_x.residuals.schema import ResidualChannel
    from sciprism_x.validators.schema import ValidationResult

    task = _graph(
        "task",
        "task",
        (_node("T1", "Implicit mechanism.", MotifFamily.CAUSAL_MECHANISTIC_RELATION),),
    )
    response = _graph(
        "response",
        "response",
        (_node("R1", "Implicit response.", MotifFamily.CAUSAL_MECHANISTIC_RELATION),),
    )
    result = compute_residuals(
        transport=_transport([[1.0]]),
        task_graph=task,
        response_graph=response,
        validations=((ValidationResult.unresolved("implicit", confidence=0.0),),),
        omega_missing=1.0,
        omega_unsupported=1.0,
        minimum_support=0.0,
        normalizers={channel: 1.0 for channel in ResidualChannel},
        normalizer_floor=1.0e-12,
    )

    assert result.components[ResidualChannel.CAUSAL].matched == 0.0


def test_normalizer_floor_is_explicit() -> None:
    from sciprism_x.motifs.schema import MotifFamily
    from sciprism_x.residuals.compute import compute_residuals
    from sciprism_x.residuals.schema import ResidualChannel
    from sciprism_x.validators.schema import ValidationResult

    task = _graph(
        "task",
        "task",
        (_node("T1", "A causes B.", MotifFamily.CAUSAL_MECHANISTIC_RELATION),),
    )
    response = _graph(
        "response",
        "response",
        (_node("R1", "B causes A.", MotifFamily.CAUSAL_MECHANISTIC_RELATION),),
    )
    result = compute_residuals(
        transport=_transport([[1.0]]),
        task_graph=task,
        response_graph=response,
        validations=((ValidationResult.fail("causal_direction_reversed", confidence=1.0),),),
        omega_missing=0.0,
        omega_unsupported=0.0,
        minimum_support=0.0,
        normalizers={channel: 0.0 for channel in ResidualChannel},
        normalizer_floor=1.0e-12,
    )

    causal = result.components[ResidualChannel.CAUSAL]
    assert causal.normalizer == 1.0e-12
    assert causal.normalized == 1.0e12
