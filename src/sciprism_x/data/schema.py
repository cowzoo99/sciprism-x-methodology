from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceRecord(StrictModel):
    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    destination: str = Field(min_length=1)

    @field_validator("destination")
    @classmethod
    def require_relative_destination(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("destination must be a relative path without '..'")
        return normalized


class SciCUCase(StrictModel):
    task_id: str = Field(min_length=1)
    prompt: str
    reference_context: str
    reference_answer: str
    domain: str = Field(min_length=1)
    competency: str = Field(min_length=1)
    motif_family: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DBPCase(StrictModel):
    pair_id: str = Field(min_length=1)
    source_entity_id: str = Field(min_length=1)
    target_entity_id: str = Field(min_length=1)
    language_pair: Literal["ZH-EN", "JA-EN", "FR-EN"]
    source_relations: tuple[str, ...]
    target_relations: tuple[str, ...]
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RegimeSplits(StrictModel):
    seed: int
    iid: tuple[str, ...]
    motif: tuple[str, ...]
    domain: tuple[str, ...]
    boundary: tuple[str, ...]
