from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResponseRecord(StrictModel):
    """Ordinary response row; acquisition details never enter this schema."""

    task_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    replicate: int = Field(ge=0)
    response_text: str = Field(min_length=1)


class ResponseAcquisitionRecord(StrictModel):
    """Internal input-manifest record for one response acquisition route."""

    route: Literal[
        "supplied_file",
        "exact_official_model",
        "nearest_official_model",
        "deterministic_reconstruction",
    ]
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str = Field(min_length=1)
    endpoint: str | None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    settings: dict[str, Any]


class ReconstructionProfile(StrictModel):
    """Explicit response-transformation rates used when source outputs are absent."""

    physical_error: float = Field(ge=0.0, le=1.0)
    logical_error: float = Field(ge=0.0, le=1.0)
    unit_error: float = Field(ge=0.0, le=1.0)
    causal_error: float = Field(ge=0.0, le=1.0)
    modal_evidence_error: float = Field(ge=0.0, le=1.0)
    uncertainty_error: float = Field(ge=0.0, le=1.0)
    endpoint_error: float = Field(ge=0.0, le=1.0)
    motif_omission: float = Field(ge=0.0, le=1.0)
    unsupported_content: float = Field(ge=0.0, le=1.0)
    length_ratio: float = Field(ge=0.25, le=2.0)
    evidence_synthesis: float = Field(ge=0.0, le=1.0)


class ReconstructionAudit(StrictModel):
    seed: int = Field(ge=0)
    source_units: tuple[str, ...]
    output_units: tuple[str, ...]
    applied_transforms: tuple[str, ...]


class ReconstructionResult(StrictModel):
    record: ResponseRecord
    audit: ReconstructionAudit
