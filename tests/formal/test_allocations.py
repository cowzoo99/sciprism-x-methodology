from pathlib import Path

from sciprism_x.formal.allocation import (
    DBP_FAMILIES,
    SCICU_FAMILIES,
    STRENGTHS,
    build_formal_allocations,
)
from sciprism_x.formal.datasets import load_formal_datasets

from .helpers import paper_contract, smoke_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _datasets(tmp_path: Path):
    return load_formal_datasets(PROJECT_ROOT / "data", extraction_root=tmp_path)


def test_paper_allocations_have_exact_regime_and_family_counts(tmp_path: Path) -> None:
    allocations = build_formal_allocations(_datasets(tmp_path), paper_contract(), seed=42)

    assert allocations.scicu.count_regime("iid") == 600
    assert allocations.scicu.count_regime("motif_or_structure") == 600
    assert allocations.scicu.count_regime("domain_transfer") == 600
    assert allocations.scicu.count_regime("boundary_stress") == 2_400
    assert allocations.scicu.family_strength_counts() == {
        (family, strength): 160
        for family in SCICU_FAMILIES
        for strength in STRENGTHS
    }
    assert {
        regime: sum(row.bsi_regime == regime for row in allocations.scicu.rows)
        for regime in ("iid", "motif_or_structure", "domain_transfer", "boundary_stress")
    } == {
        "iid": 600,
        "motif_or_structure": 600,
        "domain_transfer": 600,
        "boundary_stress": 600,
    }
    assert allocations.dbp.count_regime("iid") == 600
    assert allocations.dbp.count_regime("motif_or_structure") == 600
    assert allocations.dbp.count_regime("domain_transfer") == 600
    assert allocations.dbp.count_regime("boundary_stress") == 2_400


def test_smoke_allocations_cover_every_family_strength_and_language_pair(
    tmp_path: Path,
) -> None:
    datasets = _datasets(tmp_path)
    allocations = build_formal_allocations(datasets, smoke_contract(), seed=42)

    assert allocations.scicu.count_regime("boundary_stress") == 15
    assert allocations.scicu.family_strength_counts() == {
        (family, strength): 1
        for family in SCICU_FAMILIES
        for strength in STRENGTHS
    }
    assert {
        row.bsi_regime
        for row in allocations.scicu.rows
        if row.regime == "boundary_stress"
    } == {"pooled_smoke"}
    assert allocations.dbp.count_regime("boundary_stress") == 36
    assert allocations.dbp.family_strength_language_counts() == {
        (language, family, strength): 1
        for language in ("ZH-EN", "JA-EN", "FR-EN")
        for family in DBP_FAMILIES
        for strength in STRENGTHS
    }

    scicu_ids = {
        regime: {
            row.case_id
            for row in allocations.scicu.rows
            if row.regime == regime
        }
        for regime in ("iid", "motif_or_structure", "domain_transfer")
    }
    assert scicu_ids["iid"].isdisjoint(scicu_ids["motif_or_structure"])
    assert scicu_ids["iid"].isdisjoint(scicu_ids["domain_transfer"])
    assert scicu_ids["motif_or_structure"].isdisjoint(scicu_ids["domain_transfer"])

    by_scicu_id = {case.task_id: case for case in datasets.scicueval}
    iid_domains = {by_scicu_id[value].domain for value in scicu_ids["iid"]}
    motif_domains = {
        by_scicu_id[value].domain for value in scicu_ids["motif_or_structure"]
    }
    transfer_domains = {
        by_scicu_id[value].domain for value in scicu_ids["domain_transfer"]
    }
    assert transfer_domains
    assert transfer_domains.isdisjoint(iid_domains | motif_domains)

    iid_combinations = {
        (by_scicu_id[value].domain, by_scicu_id[value].motif_family)
        for value in scicu_ids["iid"]
    }
    motif_combinations = {
        (by_scicu_id[value].domain, by_scicu_id[value].motif_family)
        for value in scicu_ids["motif_or_structure"]
    }
    assert motif_combinations
    assert motif_combinations.isdisjoint(iid_combinations)

    dbp_ids = {
        regime: {
            row.case_id
            for row in allocations.dbp.rows
            if row.regime == regime
        }
        for regime in ("iid", "motif_or_structure", "domain_transfer")
    }
    assert dbp_ids["iid"].isdisjoint(dbp_ids["motif_or_structure"])
    assert dbp_ids["iid"].isdisjoint(dbp_ids["domain_transfer"])
    assert dbp_ids["motif_or_structure"].isdisjoint(dbp_ids["domain_transfer"])


def test_allocations_are_deterministic(tmp_path: Path) -> None:
    datasets = _datasets(tmp_path)
    first = build_formal_allocations(datasets, smoke_contract(), seed=42)
    second = build_formal_allocations(datasets, smoke_contract(), seed=42)

    assert first == second


def test_boundary_strengths_reuse_the_same_paired_base_cases(tmp_path: Path) -> None:
    allocations = build_formal_allocations(_datasets(tmp_path), smoke_contract(), seed=42)

    for family in SCICU_FAMILIES:
        by_strength = {
            strength: {
                row.case_id
                for row in allocations.scicu.rows
                if row.family == family and row.strength == strength
            }
            for strength in STRENGTHS
        }
        assert len({frozenset(values) for values in by_strength.values()}) == 1
    for language in ("ZH-EN", "JA-EN", "FR-EN"):
        for family in DBP_FAMILIES:
            by_strength = {
                strength: {
                    row.case_id
                    for row in allocations.dbp.rows
                    if row.language_pair == language
                    and row.family == family
                    and row.strength == strength
                }
                for strength in STRENGTHS
            }
            assert len({frozenset(values) for values in by_strength.values()}) == 1
