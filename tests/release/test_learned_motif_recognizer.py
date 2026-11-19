from __future__ import annotations

from sciprism_x.motifs.extract import build_response_graph
from sciprism_x.motifs.salience import SalienceWeights
from sciprism_x.motifs.schema import MotifFamily
from sciprism_x.responses.schema import ResponseRecord


class PrototypeEncoder:
    def cosine(self, left: str, right: str) -> float:
        del left
        return 0.92 if "mechanism" in right.lower() else 0.10


def test_learned_recognizer_only_fills_units_without_explicit_cues() -> None:
    graph = build_response_graph(
        ResponseRecord(
            task_id="SCICU-0001",
            model_id="anonymized_generator",
            replicate=0,
            response_text="The intervention changes the downstream state.",
        ),
        short_fragment_token_threshold=4,
        explicit_cue_confidence=1.0,
        lexical_cue_confidence=0.8,
        learned_recognizer_threshold=0.75,
        learned_recognizer_margin=0.05,
        learned_recognizer=PrototypeEncoder(),
        salience_weights=SalienceWeights(
            inverse_position=0.25,
            motif_confidence=0.25,
            variable_reuse=0.25,
            graph_centrality=0.25,
        ),
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].motif_family is MotifFamily.CAUSAL_MECHANISTIC_RELATION
    assert graph.nodes[0].confidence == 0.92
    assert graph.nodes[0].feature_evidence["recognizer_route"] == (
        "validation_frozen_prototype_cosine",
    )
