from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: str
    cache_hash: str
    staging_path: Path
    final_path: Path


@dataclass(frozen=True, slots=True)
class RunReceipt:
    run_id: str
    cache_hash: str
    track: Literal["A", "B"]
    final_path: Path
    artifact_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class StageReceipt:
    name: str
    cache_key: str
    content_hash: str
    path: Path
    resumed: bool
    input_hashes: tuple[tuple[str, str], ...]
    implementation_hash: str
    artifact_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FormalRunReceipt:
    run_id: str
    scale: Literal["smoke", "paper"]
    final_path: Path
    stage_receipts: tuple[StageReceipt, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    computed_response_count: int
    applicable_metric_cells: int
    nonfinite_metric_cells: int
    paper_sources_in_computed_rows: int

    @property
    def completed_stages(self) -> tuple[str, ...]:
        return tuple(receipt.name for receipt in self.stage_receipts)
