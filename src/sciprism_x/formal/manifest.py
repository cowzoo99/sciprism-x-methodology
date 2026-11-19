from __future__ import annotations

from dataclasses import dataclass

from sciprism_x.config import RunConfig

from .allocation import build_formal_allocations
from .config import FormalExecutionConfig, resolve_scale_contract
from .datasets import FormalDatasets


@dataclass(frozen=True, slots=True)
class PaperExperimentManifest:
    scicu_response_count: int
    scicu_nonstress_counts: dict[str, int]
    scicu_boundary_count: int
    dbp_nonstress_counts: dict[str, int]
    dbp_boundary_count: int
    seeds: tuple[int, ...]


def build_paper_manifest(
    config: RunConfig,
    formal_config: FormalExecutionConfig,
    datasets: FormalDatasets,
) -> PaperExperimentManifest:
    contract = resolve_scale_contract(config, formal_config, scale="paper")
    allocations = build_formal_allocations(
        datasets,
        contract,
        seed=config.paper.seeds[0],
    )
    nonstress = ("iid", "motif_or_structure", "domain_transfer")
    scicu_nonstress = {
        regime: allocations.scicu.count_regime(regime) for regime in nonstress
    }
    dbp_nonstress = {
        regime: allocations.dbp.count_regime(regime) for regime in nonstress
    }
    scicu_boundary = allocations.scicu.count_regime("boundary_stress")
    dbp_boundary = allocations.dbp.count_regime("boundary_stress")
    response_count = len(allocations.scicu.rows) * contract.generator_count
    response_count *= contract.responses_per_task
    if response_count != contract.scicu_response_count:
        raise ValueError("paper allocation response count differs from its scale contract")
    return PaperExperimentManifest(
        scicu_response_count=response_count,
        scicu_nonstress_counts=scicu_nonstress,
        scicu_boundary_count=scicu_boundary,
        dbp_nonstress_counts=dbp_nonstress,
        dbp_boundary_count=dbp_boundary,
        seeds=tuple(config.paper.seeds),
    )
