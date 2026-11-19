from __future__ import annotations

import math
import re
from typing import Protocol

from sciprism_x.data.schema import SciCUCase
from sciprism_x.release.protocol import (
    compiled_family_patterns,
    edge_rules_by_id,
    family_aliases,
    family_recognizer_prototypes,
)
from sciprism_x.responses.schema import ResponseRecord

from .salience import SalienceWeights, apply_salience
from .schema import (
    EdgeRelation,
    MotifFamily,
    MotifNode,
    TraceUnit,
    TypedEdge,
    TypedReasoningGraph,
)
from .segment import segment_trace

_FAMILY_PATTERNS = compiled_family_patterns()
_FAMILY_ALIASES = family_aliases()
_FAMILY_PROTOTYPES = family_recognizer_prototypes()
_EDGE_RULES = edge_rules_by_id()


class LearnedMotifRecognizer(Protocol):
    def cosine(self, left: str, right: str) -> float: ...


def _prototype_candidates(
    unit: TraceUnit,
    recognizer: LearnedMotifRecognizer,
) -> list[tuple[float, str, MotifFamily, str]]:
    candidates: list[tuple[float, str, MotifFamily, str]] = []
    for family, prototypes in _FAMILY_PROTOTYPES.items():
        scored = []
        for prototype in prototypes:
            score = float(recognizer.cosine(unit.text, prototype))
            if not math.isfinite(score):
                raise ValueError("learned motif recognizer returned a non-finite cosine")
            scored.append((score, prototype))
        best_score, best_prototype = max(scored, key=lambda item: (item[0], item[1]))
        candidates.append((best_score, family.value, family, best_prototype))
    return sorted(candidates)


def _annotated_family(value: str) -> MotifFamily | None:
    normalized = re.sub(r"[^a-z]+", "_", value.lower()).strip("_")
    for token, family in _FAMILY_ALIASES.items():
        if token in normalized:
            return family
    return None


def _feature_evidence(
    unit: TraceUnit,
    family: MotifFamily,
    matches: tuple[str, ...],
    annotated: bool,
) -> dict[str, tuple[str, ...]]:
    evidence: dict[str, tuple[str, ...]] = {"lexical_cues": matches}
    if unit.variables:
        evidence["variables"] = unit.variables
    if unit.equations:
        evidence["equations"] = unit.equations
    if unit.units:
        evidence["units"] = unit.units
    if annotated:
        evidence["task_annotation"] = (family.value,)
    return evidence


def _extract_nodes(
    units: tuple[TraceUnit, ...],
    *,
    annotated_family: MotifFamily | None,
    explicit_cue_confidence: float,
    lexical_cue_confidence: float,
    learned_recognizer_threshold: float,
    learned_recognizer_margin: float,
    learned_recognizer: LearnedMotifRecognizer | None,
) -> tuple[MotifNode, ...]:
    nodes: list[MotifNode] = []
    annotated_emitted = False
    for unit in units:
        unit_node_start = len(nodes)
        for family, patterns in _FAMILY_PATTERNS.items():
            matches = tuple(
                dict.fromkeys(
                    match.group(0)
                    for pattern in patterns
                    for match in pattern.finditer(unit.text)
                )
            )
            annotation_applies = annotated_family is family and not annotated_emitted
            if not matches and not annotation_applies:
                continue
            explicit = bool(
                unit.equations
                or (family is MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY and unit.units)
                or family is MotifFamily.MODALITY_GROUNDING
                or annotation_applies
            )
            confidence = explicit_cue_confidence if explicit else lexical_cue_confidence
            nodes.append(
                MotifNode(
                    node_id=f"N{len(nodes) + 1}",
                    text=unit.text,
                    motif_family=family,
                    confidence=confidence,
                    salience=0.0,
                    source_span=(unit.start, unit.end),
                    variables=unit.variables,
                    feature_evidence=_feature_evidence(
                        unit,
                        family,
                        matches or (family.value,),
                        annotation_applies,
                    ),
                )
            )
            if annotation_applies:
                annotated_emitted = True
        if len(nodes) != unit_node_start or learned_recognizer is None:
            continue
        candidates = _prototype_candidates(unit, learned_recognizer)
        best_score, _family_name, best_family, best_prototype = candidates[-1]
        second_score = candidates[-2][0] if len(candidates) > 1 else -1.0
        confidence = min(max(best_score, 0.0), 1.0)
        if (
            confidence < learned_recognizer_threshold
            or confidence - second_score < learned_recognizer_margin
        ):
            continue
        evidence = _feature_evidence(unit, best_family, (), False)
        evidence["recognizer_route"] = (
            "validation_frozen_prototype_cosine",
        )
        evidence["recognizer_prototype"] = (best_prototype,)
        nodes.append(
            MotifNode(
                node_id=f"N{len(nodes) + 1}",
                text=unit.text,
                motif_family=best_family,
                confidence=confidence,
                salience=0.0,
                source_span=(unit.start, unit.end),
                variables=unit.variables,
                feature_evidence=evidence,
            )
        )
    return tuple(nodes)


def _make_edge(
    source: MotifNode,
    target: MotifNode,
    relation: EdgeRelation,
    *,
    direction: str = "forward",
) -> TypedEdge:
    return TypedEdge(
        source=source.node_id,
        target=target.node_id,
        relation=relation,
        direction=direction,
        confidence=min(source.confidence, target.confidence),
        source_span=(min(source.source_span[0], target.source_span[0]), max(source.source_span[1], target.source_span[1])),
    )


def _edge_spec(rule_id: str) -> tuple[EdgeRelation, str]:
    rule = _EDGE_RULES[rule_id]
    return EdgeRelation(rule.relation), rule.direction


def _build_edges(nodes: tuple[MotifNode, ...], side: str) -> tuple[TypedEdge, ...]:
    if len(nodes) < 2:
        return ()
    edges: list[TypedEdge] = []
    seen: set[tuple[str, str, EdgeRelation]] = set()

    def add(source: MotifNode, target: MotifNode, relation: EdgeRelation, direction: str = "forward") -> None:
        if source.node_id == target.node_id:
            return
        key = (source.node_id, target.node_id, relation)
        if key not in seen:
            edges.append(_make_edge(source, target, relation, direction=direction))
            seen.add(key)

    ordered = tuple(sorted(nodes, key=lambda node: (node.source_span, node.node_id)))
    sequence_rule = "response_order" if side == "response" else "task_order"
    sequence_relation, sequence_direction = _edge_spec(sequence_rule)
    for source, target in zip(ordered, ordered[1:]):
        add(source, target, sequence_relation, sequence_direction)
        if (
            source.motif_family is MotifFamily.SYMBOLIC_TRANSFORMATION
            or target.motif_family is MotifFamily.SYMBOLIC_TRANSFORMATION
        ) and set(source.variables).intersection(target.variables):
            relation, direction = _edge_spec("symbolic_shared_variable")
            add(source, target, relation, direction)
        if target.motif_family is MotifFamily.CAUSAL_MECHANISTIC_RELATION:
            relation, direction = _edge_spec("causal_target")
            add(source, target, relation, direction)
            relation, direction = _edge_spec("causal_justification")
            add(source, target, relation, direction)
        if source.motif_family is MotifFamily.EVIDENCE_BINDING:
            relation, direction = _edge_spec("evidence_source")
            add(source, target, relation, direction)
        if source.motif_family is MotifFamily.MODALITY_GROUNDING:
            relation, direction = _edge_spec("modality_source")
            add(source, target, relation, direction)

    variable_relation, variable_direction = _edge_spec("variable_reuse")
    for index, source in enumerate(ordered):
        for target in ordered[index + 1 :]:
            if set(source.variables).intersection(target.variables):
                add(source, target, variable_relation, variable_direction)
    if len(ordered) > 2:
        relation, direction = _edge_spec("local_to_global")
        add(ordered[0], ordered[-1], relation, direction)
    return tuple(edges)


def _build_graph(
    text: str,
    *,
    source_id: str,
    side: str,
    annotated_family: MotifFamily | None,
    short_fragment_token_threshold: int,
    explicit_cue_confidence: float,
    lexical_cue_confidence: float,
    learned_recognizer_threshold: float,
    learned_recognizer_margin: float,
    learned_recognizer: LearnedMotifRecognizer | None,
    salience_weights: SalienceWeights,
) -> TypedReasoningGraph:
    if not 0.0 <= explicit_cue_confidence <= 1.0:
        raise ValueError("explicit_cue_confidence must be in [0, 1]")
    if not 0.0 <= lexical_cue_confidence <= 1.0:
        raise ValueError("lexical_cue_confidence must be in [0, 1]")
    if not 0.0 <= learned_recognizer_threshold <= 1.0:
        raise ValueError("learned_recognizer_threshold must be in [0, 1]")
    if not 0.0 <= learned_recognizer_margin <= 1.0:
        raise ValueError("learned_recognizer_margin must be in [0, 1]")
    units = segment_trace(
        text,
        short_fragment_token_threshold=short_fragment_token_threshold,
    )
    nodes = _extract_nodes(
        units,
        annotated_family=annotated_family,
        explicit_cue_confidence=explicit_cue_confidence,
        lexical_cue_confidence=lexical_cue_confidence,
        learned_recognizer_threshold=learned_recognizer_threshold,
        learned_recognizer_margin=learned_recognizer_margin,
        learned_recognizer=learned_recognizer,
    )
    edges = _build_edges(nodes, side)
    nodes = apply_salience(nodes, edges, salience_weights)
    return TypedReasoningGraph(
        source_id=source_id,
        side=side,
        nodes=nodes,
        edges=edges,
    )


def build_task_graph(
    case: SciCUCase,
    *,
    short_fragment_token_threshold: int,
    explicit_cue_confidence: float,
    lexical_cue_confidence: float,
    salience_weights: SalienceWeights,
    learned_recognizer_threshold: float = 1.0,
    learned_recognizer_margin: float = 1.0,
    learned_recognizer: LearnedMotifRecognizer | None = None,
) -> TypedReasoningGraph:
    text = " ".join((case.prompt, case.reference_context, case.reference_answer))
    return _build_graph(
        text,
        source_id=case.task_id,
        side="task",
        annotated_family=_annotated_family(case.motif_family),
        short_fragment_token_threshold=short_fragment_token_threshold,
        explicit_cue_confidence=explicit_cue_confidence,
        lexical_cue_confidence=lexical_cue_confidence,
        learned_recognizer_threshold=learned_recognizer_threshold,
        learned_recognizer_margin=learned_recognizer_margin,
        learned_recognizer=learned_recognizer,
        salience_weights=salience_weights,
    )


def build_response_graph(
    response: ResponseRecord,
    *,
    short_fragment_token_threshold: int,
    explicit_cue_confidence: float,
    lexical_cue_confidence: float,
    salience_weights: SalienceWeights,
    learned_recognizer_threshold: float = 1.0,
    learned_recognizer_margin: float = 1.0,
    learned_recognizer: LearnedMotifRecognizer | None = None,
) -> TypedReasoningGraph:
    return _build_graph(
        response.response_text,
        source_id=f"{response.task_id}:{response.model_id}:{response.replicate}",
        side="response",
        annotated_family=None,
        short_fragment_token_threshold=short_fragment_token_threshold,
        explicit_cue_confidence=explicit_cue_confidence,
        lexical_cue_confidence=lexical_cue_confidence,
        learned_recognizer_threshold=learned_recognizer_threshold,
        learned_recognizer_margin=learned_recognizer_margin,
        learned_recognizer=learned_recognizer,
        salience_weights=salience_weights,
    )
