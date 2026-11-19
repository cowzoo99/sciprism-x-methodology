import pytest
from pydantic import ValidationError


def _case():
    from sciprism_x.data.schema import SciCUCase

    return SciCUCase(
        task_id="SCICU-0417",
        prompt=(
            "Using Table 1, derive v from qE = qvB and explain why the forces "
            "balance. State the applicable uncertainty."
        ),
        reference_context="Table 1 reports E = 24 V/m and B = 2 T.",
        reference_answer=(
            "Under ideal conditions, the forces balance because they oppose. "
            "According to Table 1, v = E/B = 12 m/s, approximately."
        ),
        domain="physics",
        competency="multi-source integration",
        motif_family="unit/dimensional consistency",
        source_hash="a" * 64,
    )


def _weights():
    from sciprism_x.motifs.salience import SalienceWeights

    return SalienceWeights(
        inverse_position=0.25,
        motif_confidence=0.25,
        variable_reuse=0.25,
        graph_centrality=0.25,
    )


def test_salience_is_normalized_and_explicit() -> None:
    from sciprism_x.motifs.extract import build_task_graph
    from sciprism_x.motifs.schema import MotifFamily

    graph = build_task_graph(
        _case(),
        short_fragment_token_threshold=4,
        explicit_cue_confidence=1.0,
        lexical_cue_confidence=0.8,
        salience_weights=_weights(),
    )

    assert graph.nodes
    assert all(0.0 <= node.salience <= 1.0 for node in graph.nodes)
    assert {node.motif_family for node in graph.nodes} <= set(MotifFamily)
    assert all(node.feature_evidence for node in graph.nodes)
    assert all(node.source_span[0] < node.source_span[1] for node in graph.nodes)


def test_six_motif_families_and_typed_edges_are_observable() -> None:
    from sciprism_x.motifs.extract import build_task_graph
    from sciprism_x.motifs.schema import EdgeRelation, MotifFamily

    graph = build_task_graph(
        _case(),
        short_fragment_token_threshold=4,
        explicit_cue_confidence=1.0,
        lexical_cue_confidence=0.8,
        salience_weights=_weights(),
    )

    assert {node.motif_family for node in graph.nodes} == set(MotifFamily)
    assert graph.edges
    assert {edge.relation for edge in graph.edges} <= set(EdgeRelation)
    assert all(edge.direction in {"forward", "bidirectional"} for edge in graph.edges)
    assert all(0.0 <= edge.confidence <= 1.0 for edge in graph.edges)


def test_response_graph_uses_only_observable_response_content() -> None:
    from sciprism_x.motifs.extract import build_response_graph
    from sciprism_x.motifs.schema import EdgeRelation
    from sciprism_x.responses.schema import ResponseRecord

    response = ResponseRecord(
        task_id="SCICU-0417",
        model_id="gpt-4o-2024-08-06",
        replicate=0,
        response_text=(
            "From Table 1, qE = qvB. Therefore v = E/B = 12 m/s because the "
            "forces balance. This is approximately valid under ideal conditions."
        ),
    )
    graph = build_response_graph(
        response,
        short_fragment_token_threshold=4,
        explicit_cue_confidence=1.0,
        lexical_cue_confidence=0.8,
        salience_weights=_weights(),
    )

    assert graph.side == "response"
    assert graph.source_id == "SCICU-0417:gpt-4o-2024-08-06:0"
    assert any(edge.relation is EdgeRelation.TEMPORAL_SUCCESSION for edge in graph.edges)
    assert all("model" not in key.lower() for node in graph.nodes for key in node.feature_evidence)


def test_salience_weights_must_be_nonnegative_and_sum_to_one() -> None:
    from sciprism_x.motifs.salience import SalienceWeights

    with pytest.raises(ValidationError):
        SalienceWeights(
            inverse_position=0.4,
            motif_confidence=0.4,
            variable_reuse=0.4,
            graph_centrality=-0.2,
        )

    with pytest.raises(ValidationError):
        SalienceWeights(
            inverse_position=0.2,
            motif_confidence=0.2,
            variable_reuse=0.2,
            graph_centrality=0.2,
        )
