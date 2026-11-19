from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sciprism_x.motifs.schema import EdgeRelation, MotifFamily, MotifNode, TypedReasoningGraph
from sciprism_x.validators.schema import ValidationResult, ValidationStatus

from .schema import CostBundle


class ScientificTextEncoder(Protocol):
    def cosine(self, left: str, right: str) -> float: ...


class CostWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    motif: float = Field(ge=0.0)
    role: float = Field(ge=0.0)
    modality: float = Field(ge=0.0)
    evidence_anchor: float = Field(ge=0.0)
    variable_equation_anchor: float = Field(ge=0.0)
    scibert_cosine: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_unit_sum(self) -> "CostWeights":
        if not math.isclose(
            sum(self.model_dump().values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("node-cost weights must sum to one")
        return self


_ROLE = {
    MotifFamily.SYMBOLIC_TRANSFORMATION: "operation",
    MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY: "constraint",
    MotifFamily.CAUSAL_MECHANISTIC_RELATION: "mechanism",
    MotifFamily.EVIDENCE_BINDING: "support",
    MotifFamily.MODALITY_GROUNDING: "grounding",
    MotifFamily.UNCERTAINTY_QUALIFICATION: "qualifier",
}
_ANCHOR = re.compile(
    r"\b(table|figure|fig\.?|graph|image|record|passage|reference|source)"
    r"(?:\s+([A-Za-z0-9.-]+))?",
    re.IGNORECASE,
)
_MODALITY = re.compile(
    r"\b(table|figure|fig\.?|graph|image|record|passage)\b",
    re.IGNORECASE,
)


def _jaccard_mismatch(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def _pairwise(
    source: Sequence[MotifNode],
    target: Sequence[MotifNode],
    function,
) -> np.ndarray:
    return np.array(
        [[function(left, right) for right in target] for left in source],
        dtype=np.float64,
    )


def _modalities(node: MotifNode) -> set[str]:
    return {
        match.group(1).lower().rstrip(".").replace("fig", "figure")
        for match in _MODALITY.finditer(node.text)
    }


def _anchors(node: MotifNode) -> set[str]:
    return {
        f"{match.group(1).lower().rstrip('.').replace('fig', 'figure')}:"
        f"{_normalize_anchor_identifier(match.group(2) or '')}"
        for match in _ANCHOR.finditer(node.text)
    }


def _normalize_anchor_identifier(value: str) -> str:
    return value.lower().rstrip(".,;:)]}")


def _variable_equation_anchors(node: MotifNode) -> set[str]:
    values = {f"variable:{value}" for value in node.variables}
    for equation in node.feature_evidence.get("equations", ()):
        normalized = re.sub(r"\s+", "", equation).lower()
        values.add(f"equation:{normalized}")
    return values


def _cosine_mismatch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True)
    if np.any(source_norm == 0.0) or np.any(target_norm == 0.0):
        raise ValueError("SciBERT embeddings must have nonzero norm")
    cosine = (source / source_norm) @ (target / target_norm).T
    return np.clip((1.0 - cosine) / 2.0, 0.0, 1.0).astype(np.float64)


def _text_encoder_mismatch(
    source: Sequence[MotifNode],
    target: Sequence[MotifNode],
    encoder: ScientificTextEncoder,
) -> np.ndarray:
    return np.asarray(
        [
            [1.0 - encoder.cosine(left.text, right.text) for right in target]
            for left in source
        ],
        dtype=np.float64,
    ).clip(0.0, 1.0)


def _adjacency_tensor(graph: TypedReasoningGraph) -> np.ndarray:
    relations = tuple(EdgeRelation)
    relation_index = {relation: index for index, relation in enumerate(relations)}
    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    tensor = np.zeros(
        (len(relations), len(graph.nodes), len(graph.nodes)),
        dtype=np.float64,
    )
    for edge in graph.edges:
        relation = relation_index[edge.relation]
        source = node_index[edge.source]
        target = node_index[edge.target]
        tensor[relation, source, target] = max(
            tensor[relation, source, target], edge.confidence
        )
        if edge.direction == "bidirectional":
            tensor[relation, target, source] = max(
                tensor[relation, target, source], edge.confidence
            )
    for relation in range(tensor.shape[0]):
        maximum = float(tensor[relation].max(initial=0.0))
        if maximum > 0.0:
            tensor[relation] /= maximum
    return tensor


def _mass(nodes: Sequence[MotifNode]) -> np.ndarray:
    values = np.array([node.salience for node in nodes], dtype=np.float64)
    total = float(values.sum())
    if total == 0.0:
        return np.full(len(nodes), 1.0 / len(nodes), dtype=np.float64)
    return values / total


def build_costs(
    task_graph: TypedReasoningGraph,
    response_graph: TypedReasoningGraph,
    validations: Sequence[Sequence[ValidationResult]],
    *,
    weights: CostWeights,
    scibert_embeddings: tuple[np.ndarray, np.ndarray] | None,
    scientific_encoder: ScientificTextEncoder | None = None,
    unresolved_soft_penalty: float,
    unavailable_soft_penalty: float,
) -> CostBundle:
    if not task_graph.nodes or not response_graph.nodes:
        raise ValueError("transport costs require nonempty graphs")
    shape = (len(task_graph.nodes), len(response_graph.nodes))
    if len(validations) != shape[0] or any(len(row) != shape[1] for row in validations):
        raise ValueError("validation matrix shape does not match graph nodes")
    for name, value in (
        ("unresolved_soft_penalty", unresolved_soft_penalty),
        ("unavailable_soft_penalty", unavailable_soft_penalty),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    source = task_graph.nodes
    target = response_graph.nodes
    components = {
        "motif": _pairwise(
            source,
            target,
            lambda left, right: float(left.motif_family is not right.motif_family),
        ),
        "role": _pairwise(
            source,
            target,
            lambda left, right: float(_ROLE[left.motif_family] != _ROLE[right.motif_family]),
        ),
        "modality": _pairwise(
            source,
            target,
            lambda left, right: _jaccard_mismatch(_modalities(left), _modalities(right)),
        ),
        "evidence_anchor": _pairwise(
            source,
            target,
            lambda left, right: _jaccard_mismatch(_anchors(left), _anchors(right)),
        ),
        "variable_equation_anchor": _pairwise(
            source,
            target,
            lambda left, right: _jaccard_mismatch(
                _variable_equation_anchors(left),
                _variable_equation_anchors(right),
            ),
        ),
    }
    if weights.scibert_cosine > 0.0:
        if scibert_embeddings is not None:
            source_embeddings = np.asarray(scibert_embeddings[0], dtype=np.float64)
            target_embeddings = np.asarray(scibert_embeddings[1], dtype=np.float64)
            if source_embeddings.shape[0] != shape[0] or target_embeddings.shape[0] != shape[1]:
                raise ValueError("SciBERT embedding rows do not match graph nodes")
            if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
                raise ValueError("SciBERT embeddings must be matrices")
            if source_embeddings.shape[1] != target_embeddings.shape[1]:
                raise ValueError("SciBERT embedding dimensions must match")
            components["scibert_cosine"] = _cosine_mismatch(
                source_embeddings,
                target_embeddings,
            )
        elif scientific_encoder is not None:
            components["scibert_cosine"] = _text_encoder_mismatch(
                source,
                target,
                scientific_encoder,
            )
        else:
            raise ValueError(
                "SciBERT embeddings or a scientific text encoder are required by the nonzero cost weight"
            )
    else:
        components["scibert_cosine"] = np.zeros(shape, dtype=np.float64)

    weight_values = weights.model_dump()
    node_cost = sum(
        weight_values[name] * component for name, component in components.items()
    ).astype(np.float64)
    hard_mask = np.ones(shape, dtype=np.bool_)
    soft = np.zeros(shape, dtype=np.float64)
    for row_index, row in enumerate(validations):
        for column_index, result in enumerate(row):
            if result.status is ValidationStatus.FAIL:
                hard_mask[row_index, column_index] = False
                soft[row_index, column_index] = 1.0
            elif result.status is ValidationStatus.PASS:
                soft[row_index, column_index] = 1.0 - result.confidence
            elif result.status is ValidationStatus.UNRESOLVED:
                soft[row_index, column_index] = unresolved_soft_penalty
            else:
                soft[row_index, column_index] = unavailable_soft_penalty

    return CostBundle(
        node_components=components,
        node_cost=node_cost,
        source_adjacency=_adjacency_tensor(task_graph),
        target_adjacency=_adjacency_tensor(response_graph),
        soft_constraint_cost=soft,
        hard_mask=hard_mask,
        source_mass=_mass(source),
        target_mass=_mass(target),
    )
