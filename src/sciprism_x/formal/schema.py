from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ScaleContract:
    scale: Literal["smoke", "paper"]
    scicu_nonstress_per_regime: int
    scicu_boundary: int
    dbp_nonstress_per_regime: int
    dbp_boundary: int
    generator_count: int
    responses_per_task: int
    seeds: tuple[int, ...]

    @property
    def scicu_task_count(self) -> int:
        return self.scicu_nonstress_per_regime * 3 + self.scicu_boundary

    @property
    def scicu_response_count(self) -> int:
        return self.scicu_task_count * self.generator_count * self.responses_per_task

    @property
    def dbp_case_count(self) -> int:
        return self.dbp_nonstress_per_regime * 3 + self.dbp_boundary
