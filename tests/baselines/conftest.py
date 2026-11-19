import numpy as np
import pytest


@pytest.fixture
def evaluated_case():
    from sciprism_x.baselines.schema import EvaluatedCase, GraphEdge, GraphView

    source = GraphView(
        nodes=("a", "b"),
        edges=(GraphEdge("a", "supports", "b"),),
    )
    target = GraphView(
        nodes=("x", "y"),
        edges=(GraphEdge("x", "supports", "y"),),
    )
    adjacency = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64)
    return EvaluatedCase(
        case_id="case-1",
        dataset="scicueval",
        task="What follows from the evidence?",
        response="The supported conclusion is B.",
        reference_context="Evidence A supports conclusion B.",
        predicted_target="B",
        reference_target="B",
        response_segments=("Evidence A", "conclusion B"),
        evidence_spans=("Evidence A", "supports conclusion B"),
        task_metadata=(("domain", "physics"),),
        confidence=0.8,
        entropy=0.2,
        representation=np.array([1.0, 2.0], dtype=np.float64),
        validation_support=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        source_graph=source,
        target_graph=target,
        pairwise_cost=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        source_mass=np.array([0.5, 0.5], dtype=np.float64),
        target_mass=np.array([0.5, 0.5], dtype=np.float64),
        source_adjacency=adjacency,
        target_adjacency=adjacency,
        candidate_scores=np.array([0.9, 0.2, 0.1], dtype=np.float64),
        correct_candidate_index=0,
        residuals=np.ones(6, dtype=np.float64),
        transport_plan=np.eye(2, dtype=np.float64),
        generator_identity="withheld-generator",
        motif_graph="withheld-motif-graph",
        scientific_constraint_mask=np.eye(2, dtype=np.bool_),
        boundary_label="withheld-boundary",
        open_set_category="withheld-open-set",
    )
