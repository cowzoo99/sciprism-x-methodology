from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MotifFamily(str, Enum):
    SYMBOLIC_TRANSFORMATION = "symbolic_transformation"
    UNIT_DIMENSIONAL_CONSISTENCY = "unit_dimensional_consistency"
    CAUSAL_MECHANISTIC_RELATION = "causal_mechanistic_relation"
    EVIDENCE_BINDING = "evidence_binding"
    MODALITY_GROUNDING = "modality_grounding"
    UNCERTAINTY_QUALIFICATION = "uncertainty_qualification"


class EdgeRelation(str, Enum):
    PREREQUISITE = "prerequisite"
    TEMPORAL_SUCCESSION = "temporal_succession"
    SYMBOLIC_DEPENDENCE = "symbolic_dependence"
    VARIABLE_REUSE = "variable_reuse"
    CAUSAL_DIRECTION = "causal_direction"
    JUSTIFICATORY_SUPPORT = "justificatory_support"
    EVIDENCE_SUPPORT = "evidence_support"
    LOCAL_TO_GLOBAL = "local_to_global"
    CROSS_MODAL_GROUNDING = "cross_modal_grounding"


class TraceUnit(StrictModel):
    unit_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    variables: tuple[str, ...]
    equations: tuple[str, ...]
    units: tuple[str, ...]
    protected_evidence: tuple[str, ...]

    @model_validator(mode="after")
    def validate_span(self) -> "TraceUnit":
        if self.end <= self.start:
            raise ValueError("trace-unit end must be greater than start")
        return self


class MotifNode(StrictModel):
    node_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    motif_family: MotifFamily
    confidence: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)
    source_span: tuple[int, int]
    variables: tuple[str, ...]
    feature_evidence: dict[str, tuple[str, ...]]


class TypedEdge(StrictModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: EdgeRelation
    direction: Literal["forward", "bidirectional"]
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: tuple[int, int]


class TypedReasoningGraph(StrictModel):
    source_id: str = Field(min_length=1)
    side: Literal["task", "response"]
    nodes: tuple[MotifNode, ...]
    edges: tuple[TypedEdge, ...]

    @model_validator(mode="after")
    def validate_edge_endpoints(self) -> "TypedReasoningGraph":
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError("edge endpoint is absent from nodes")
        return self
