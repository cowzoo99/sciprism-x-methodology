from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sciprism_x.motifs.schema import MotifNode


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"
    UNAVAILABLE_EXTERNAL = "UNAVAILABLE_EXTERNAL"


class EvidenceSpan(StrictModel):
    side: Literal["task", "response", "context"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)


class ValidationPair(StrictModel):
    task_node: MotifNode
    response_node: MotifNode
    reference_context: str


class ValidationResult(StrictModel):
    status: ValidationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(min_length=1)
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    validator_id: str = "unspecified"

    @classmethod
    def pass_(
        cls,
        reason_code: str,
        *,
        confidence: float,
        evidence_spans: tuple[EvidenceSpan, ...] = (),
        validator_id: str = "unspecified",
    ) -> "ValidationResult":
        return cls(
            status=ValidationStatus.PASS,
            confidence=confidence,
            reason_code=reason_code,
            evidence_spans=evidence_spans,
            validator_id=validator_id,
        )

    @classmethod
    def fail(
        cls,
        reason_code: str,
        *,
        confidence: float,
        evidence_spans: tuple[EvidenceSpan, ...] = (),
        validator_id: str = "unspecified",
    ) -> "ValidationResult":
        return cls(
            status=ValidationStatus.FAIL,
            confidence=confidence,
            reason_code=reason_code,
            evidence_spans=evidence_spans,
            validator_id=validator_id,
        )

    @classmethod
    def unresolved(
        cls,
        reason_code: str,
        *,
        confidence: float,
        evidence_spans: tuple[EvidenceSpan, ...] = (),
        validator_id: str = "unspecified",
    ) -> "ValidationResult":
        return cls(
            status=ValidationStatus.UNRESOLVED,
            confidence=confidence,
            reason_code=reason_code,
            evidence_spans=evidence_spans,
            validator_id=validator_id,
        )

    @classmethod
    def unavailable(
        cls,
        reason_code: str,
        *,
        evidence_spans: tuple[EvidenceSpan, ...] = (),
        validator_id: str = "unspecified",
    ) -> "ValidationResult":
        return cls(
            status=ValidationStatus.UNAVAILABLE_EXTERNAL,
            confidence=0.0,
            reason_code=reason_code,
            evidence_spans=evidence_spans,
            validator_id=validator_id,
        )


def pair_evidence(pair: ValidationPair) -> tuple[EvidenceSpan, ...]:
    return (
        EvidenceSpan(
            side="task",
            start=pair.task_node.source_span[0],
            end=pair.task_node.source_span[1],
            text=pair.task_node.text,
        ),
        EvidenceSpan(
            side="response",
            start=pair.response_node.source_span[0],
            end=pair.response_node.source_span[1],
            text=pair.response_node.text,
        ),
    )
