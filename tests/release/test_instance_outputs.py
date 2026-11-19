from __future__ import annotations

import numpy as np

from sciprism_x.data.schema import SciCUCase
from sciprism_x.formal.evaluator import EvaluatedComponents
from sciprism_x.motifs.schema import MotifFamily, MotifNode, TypedReasoningGraph
from sciprism_x.release.instance_outputs import serialize_evaluated_components
from sciprism_x.residuals.schema import (
    ResidualChannel,
    ResidualComponent,
    ResidualVector,
)
from sciprism_x.responses.schema import ResponseRecord
from sciprism_x.transport.schema import (
    ObjectiveComponents,
    TransportDiagnostics,
    TransportResult,
)
from sciprism_x.validators.schema import ValidationResult


def _node(node_id: str, text: str) -> MotifNode:
    return MotifNode(
        node_id=node_id,
        text=text,
        motif_family=MotifFamily.EVIDENCE_BINDING,
        confidence=1.0,
        salience=1.0,
        source_span=(0, len(text)),
        variables=(),
        feature_evidence={"lexical_cues": ("evidence",)},
    )


def test_component_serialization_preserves_graph_validator_transport_and_residual_links() -> None:
    case = SciCUCase(
        task_id="SCICU-0001",
        domain="physics",
        competency="evidence",
        motif_family="evidence binding",
        prompt="Use the evidence.",
        reference_context="Table 1 reports a positive result.",
        reference_answer="Positive.",
        source_hash="0" * 64,
    )
    response = ResponseRecord(
        task_id=case.task_id,
        model_id="anonymized_generator",
        replicate=0,
        response_text="According to Table 1, the result is positive.",
    )
    task_node = _node("N1", "Table 1 reports a positive result")
    response_node = _node("N1", response.response_text)
    component = ResidualComponent(
        matched=0.1,
        missing=0.0,
        unsupported=0.0,
        raw=0.1,
        normalizer=1.0,
        normalized=0.1,
    )
    components = EvaluatedComponents(
        case=case,
        response=response,
        task_graph=TypedReasoningGraph(
            source_id=case.task_id,
            side="task",
            nodes=(task_node,),
            edges=(),
        ),
        response_graph=TypedReasoningGraph(
            source_id=case.task_id,
            side="response",
            nodes=(response_node,),
            edges=(),
        ),
        validations=((ValidationResult.pass_(
            "evidence_anchor_match",
            confidence=1.0,
            validator_id="deterministic_evidence_binding",
        ),),),
        transport=TransportResult(
            plan=np.asarray([[1.0]], dtype=np.float64),
            objective=ObjectiveComponents(
                node=0.1,
                structural=0.0,
                soft_constraint=0.0,
                negative_entropy=0.0,
                total=0.1,
            ),
            diagnostics=TransportDiagnostics(
                feasible=True,
                converged=True,
                iterations=5,
                marginal_error=0.0,
                objective_change=0.0,
                consecutive_stable_iterations=5,
                reason="converged",
            ),
        ),
        residuals=ResidualVector(
            components={channel: component for channel in ResidualChannel}
        ),
    )

    serialized = serialize_evaluated_components(
        components,
        model_id="gpt-4o-2024-08-06",
        replicate=0,
    )

    assert serialized.instance_key == "SCICU-0001::gpt-4o-2024-08-06::0"
    assert serialized.link_row["plan_key"] == serialized.plan_key
    assert serialized.plan.shape == (1, 1)
    assert len(serialized.validator_rows) == 1
    assert len(serialized.residual_rows) == 6
    assert serialized.validator_rows[0]["validator_id"] == (
        "deterministic_evidence_binding"
    )
