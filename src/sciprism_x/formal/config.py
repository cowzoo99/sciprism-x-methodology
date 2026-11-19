from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sciprism_x.config import RunConfig

from .schema import ScaleContract


class FormalExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    split_fractions: tuple[float, float, float]
    smoke_nonstress_cases: int = Field(gt=0)
    smoke_scicu_boundary_cases: int = Field(gt=0)
    smoke_dbp_boundary_cases: int = Field(gt=0)
    response_profile_entropy_weight: float = Field(ge=0.0)
    stable_pcs_quantile: float = Field(ge=0.0, le=1.0)
    stable_residual_quantile: float = Field(ge=0.0, le=1.0)
    invalid_pcs_quantile: float = Field(ge=0.0, le=1.0)
    invalid_residual_quantile: float = Field(ge=0.0, le=1.0)
    proxy_pass_threshold: float = Field(ge=0.0, le=1.0)
    proxy_fail_threshold: float = Field(ge=0.0, le=1.0)
    resume: Literal[True]

    @model_validator(mode="after")
    def validate_protocol(self) -> "FormalExecutionConfig":
        if any(value <= 0.0 for value in self.split_fractions):
            raise ValueError("split fractions must be positive")
        if abs(sum(self.split_fractions) - 1.0) > 1.0e-12:
            raise ValueError("split fractions must sum to one")
        if self.smoke_scicu_boundary_cases % 15 != 0:
            raise ValueError("SciCUEval smoke boundary count must cover 5 families x 3 strengths")
        if self.smoke_dbp_boundary_cases % 36 != 0:
            raise ValueError("DBP smoke boundary count must cover 3 pairs x 4 families x 3 strengths")
        return self


def load_formal_config(path: str | Path) -> FormalExecutionConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("formal configuration must be a mapping")
    return FormalExecutionConfig.model_validate(payload)


def resolve_scale_contract(
    run: RunConfig,
    formal: FormalExecutionConfig,
    *,
    scale: Literal["smoke", "paper"],
) -> ScaleContract:
    if scale == "paper":
        scicu_nonstress = run.regimes.nonstress_cases
        scicu_boundary = run.regimes.boundary_stress_cases
        dbp_nonstress = run.regimes.nonstress_cases
        dbp_boundary = run.regimes.boundary_stress_cases
    elif scale == "smoke":
        scicu_nonstress = formal.smoke_nonstress_cases
        scicu_boundary = formal.smoke_scicu_boundary_cases
        dbp_nonstress = formal.smoke_nonstress_cases
        dbp_boundary = formal.smoke_dbp_boundary_cases
    else:
        raise ValueError(f"unsupported execution scale: {scale}")
    return ScaleContract(
        scale=scale,
        scicu_nonstress_per_regime=scicu_nonstress,
        scicu_boundary=scicu_boundary,
        dbp_nonstress_per_regime=dbp_nonstress,
        dbp_boundary=dbp_boundary,
        generator_count=len(run.generators.model_ids),
        responses_per_task=run.generators.responses_per_task,
        seeds=run.paper.seeds,
    )
