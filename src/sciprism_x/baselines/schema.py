from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class BaselineKind(str, Enum):
    ENDPOINT = "endpoint"
    ATTRIBUTION = "attribution"
    FIXED_TAXONOMY = "fixed_taxonomy"
    UNCERTAINTY = "uncertainty"
    KNN_SUPPORT = "knn_support"
    CENTROID_SUPPORT = "centroid_support"
    GRAPH_EDIT = "graph_edit"
    SINKHORN = "sinkhorn"
    GROMOV = "gromov"
    STRUCTURE_AWARE = "structure_aware"
    RELATION_AWARE = "relation_aware"
    LLM_JUDGE = "llm_judge"
    HITS_AT_1 = "hits_at_1"
    MRR = "mrr"


class BaselineStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE_EXTERNAL = "unavailable_external"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True, slots=True)
class GraphView:
    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True, slots=True)
class EvaluatedCase:
    """Internal superset. It must never be passed directly to a baseline."""

    case_id: str
    dataset: Literal["scicueval", "dbp15k"]
    task: str
    response: str
    reference_context: str
    predicted_target: str
    reference_target: str
    response_segments: tuple[str, ...]
    evidence_spans: tuple[str, ...]
    task_metadata: tuple[tuple[str, str], ...]
    confidence: float
    entropy: float
    representation: FloatArray
    validation_support: FloatArray
    source_graph: GraphView
    target_graph: GraphView
    pairwise_cost: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_adjacency: FloatArray
    target_adjacency: FloatArray
    candidate_scores: FloatArray
    correct_candidate_index: int
    residuals: FloatArray
    transport_plan: FloatArray
    generator_identity: str
    motif_graph: object
    scientific_constraint_mask: BoolArray
    boundary_label: str
    open_set_category: str


@dataclass(frozen=True, slots=True)
class EndpointInput:
    case_id: str
    predicted_target: str
    reference_target: str


@dataclass(frozen=True, slots=True)
class AttributionInput:
    case_id: str
    response_segments: tuple[str, ...]
    evidence_spans: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaxonomyInput:
    case_id: str
    response: str
    task_metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class UncertaintyInput:
    case_id: str
    confidence: float
    entropy: float


@dataclass(frozen=True, slots=True)
class SupportInput:
    case_id: str
    representation: FloatArray
    validation_support: FloatArray


@dataclass(frozen=True, slots=True)
class GraphBaselineInput:
    case_id: str
    source_graph: GraphView
    target_graph: GraphView
    pairwise_cost: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_adjacency: FloatArray
    target_adjacency: FloatArray


@dataclass(frozen=True, slots=True)
class LLMJudgeInput:
    case_id: str
    task: str
    response: str
    reference_context: str


@dataclass(frozen=True, slots=True)
class RankingInput:
    case_id: str
    candidate_scores: FloatArray
    correct_candidate_index: int


BaselineInput: TypeAlias = (
    EndpointInput
    | AttributionInput
    | TaxonomyInput
    | UncertaintyInput
    | SupportInput
    | GraphBaselineInput
    | LLMJudgeInput
    | RankingInput
)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    kind: BaselineKind
    status: BaselineStatus
    metric: str
    score: float | None
    details: tuple[tuple[str, str], ...]


class BaselineAdapter(Protocol):
    kind: BaselineKind
    input_type: type[BaselineInput]

    def evaluate(self, value: BaselineInput) -> BaselineResult: ...


def _readonly_copy(value: FloatArray) -> FloatArray:
    copied = np.array(value, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def project_baseline_input(case: EvaluatedCase, kind: BaselineKind) -> BaselineInput:
    if kind is BaselineKind.ENDPOINT:
        return EndpointInput(case.case_id, case.predicted_target, case.reference_target)
    if kind is BaselineKind.ATTRIBUTION:
        return AttributionInput(case.case_id, case.response_segments, case.evidence_spans)
    if kind is BaselineKind.FIXED_TAXONOMY:
        return TaxonomyInput(case.case_id, case.response, case.task_metadata)
    if kind is BaselineKind.UNCERTAINTY:
        return UncertaintyInput(case.case_id, case.confidence, case.entropy)
    if kind in (BaselineKind.KNN_SUPPORT, BaselineKind.CENTROID_SUPPORT):
        return SupportInput(
            case.case_id,
            _readonly_copy(case.representation),
            _readonly_copy(case.validation_support),
        )
    if kind in (
        BaselineKind.GRAPH_EDIT,
        BaselineKind.SINKHORN,
        BaselineKind.GROMOV,
        BaselineKind.STRUCTURE_AWARE,
        BaselineKind.RELATION_AWARE,
    ):
        return GraphBaselineInput(
            case.case_id,
            case.source_graph,
            case.target_graph,
            _readonly_copy(case.pairwise_cost),
            _readonly_copy(case.source_mass),
            _readonly_copy(case.target_mass),
            _readonly_copy(case.source_adjacency),
            _readonly_copy(case.target_adjacency),
        )
    if kind is BaselineKind.LLM_JUDGE:
        return LLMJudgeInput(
            case.case_id,
            case.task,
            case.response,
            case.reference_context,
        )
    if kind in (BaselineKind.HITS_AT_1, BaselineKind.MRR):
        return RankingInput(
            case.case_id,
            _readonly_copy(case.candidate_scores),
            case.correct_candidate_index,
        )
    raise ValueError(f"unsupported baseline kind: {kind}")
