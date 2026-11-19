import hashlib

from sciprism_x.data.schema import SciCUCase
from sciprism_x.formal.allocation import (
    DBP_FAMILIES,
    SCICU_FAMILIES,
    STRENGTHS,
    AllocationRow,
)
from sciprism_x.formal.datasets import DBPLanguageDataset
from sciprism_x.formal.perturbed_cases import (
    materialize_dbp_perturbation,
    materialize_scicu_perturbation,
)


def _case() -> SciCUCase:
    return SciCUCase(
        task_id="SCICU-MATERIALIZE-1",
        prompt="Under ideal conditions, derive v = E/B using Table 1 and explain the cause.",
        reference_context=(
            "Table 1 reports E = 4.0e3 V/m and B = 0.05 T. "
            "Figure 2 confirms perpendicular fields. The result is approximately ideal."
        ),
        reference_answer="Under ideal conditions qE=qvB, so v=E/B=8.0e4 m/s according to Table 1.",
        domain="physics",
        competency="Information_Integration",
        motif_family="evidence binding",
        source_hash=hashlib.sha256(b"SCICU-MATERIALIZE-1").hexdigest(),
    )


def test_every_scicu_family_changes_the_declared_input() -> None:
    original = _case()
    records = []
    for family in SCICU_FAMILIES:
        row = AllocationRow(
            dataset="scicueval",
            allocation_id=f"{family}-0.5",
            case_id=original.task_id,
            regime="boundary_stress",
            family=family,
            strength=0.5,
            language_pair=None,
        )
        records.append(materialize_scicu_perturbation(original, row, seed=42))

    assert {record.family for record in records} == set(SCICU_FAMILIES)
    assert all(record.changed_units for record in records)
    assert all(record.valid for record in records)
    assert all(all(dict(record.validity_checks).values()) for record in records)
    assert all(record.case.source_hash != original.source_hash for record in records)
    assert all(record.case.task_id != original.task_id for record in records)


def test_strength_levels_produce_distinct_audited_cases() -> None:
    original = _case()
    records = [
        materialize_scicu_perturbation(
            original,
            AllocationRow(
                dataset="scicueval",
                allocation_id=f"context-{strength}",
                case_id=original.task_id,
                regime="boundary_stress",
                family="context_composition_shift",
                strength=strength,
                language_pair=None,
            ),
            seed=42,
        )
        for strength in STRENGTHS
    ]

    assert len({record.case.source_hash for record in records}) == 3
    assert len({record.case.reference_context for record in records}) == 3
    assert all(record.changed_units for record in records)


def _dbp_dataset() -> DBPLanguageDataset:
    return DBPLanguageDataset(
        language_pair="ZH-EN",
        entity_ids_1=((0, "source-0"), (1, "source-1"), (2, "source-2")),
        entity_ids_2=((10, "target-10"), (11, "target-11"), (12, "target-12")),
        triples_1=((0, 1, 1), (1, 2, 2), (2, 3, 0)),
        triples_2=((10, 1, 11), (11, 2, 12), (12, 3, 10)),
        seed_pairs=((20, 30), (21, 31), (22, 32), (23, 33)),
        reference_pairs=((0, 10),),
        target_candidate_ids=(10, 11, 12, 30, 31, 32, 33),
    )


def test_every_dbp_family_materializes_immutable_audited_changes() -> None:
    original = _dbp_dataset()
    records = []
    for family in DBP_FAMILIES:
        row = AllocationRow(
            dataset="dbp15k",
            allocation_id=f"dbp-{family}-0.5",
            case_id="ZH-EN::0::10",
            regime="boundary_stress",
            family=family,
            strength=0.5,
            language_pair="ZH-EN",
        )
        records.append(materialize_dbp_perturbation(original, row, seed=42))

    assert {record.family for record in records} == set(DBP_FAMILIES)
    assert all(record.changed_edges or record.changed_seed_alignments for record in records)
    assert all(record.valid for record in records)
    assert all(all(dict(record.validity_checks).values()) for record in records)
    assert all(record.source_id == 0 and record.target_id == 10 for record in records)
    assert original == _dbp_dataset()
    assert any(record.triples_1 != original.triples_1 for record in records)
    assert any(record.seed_pairs != original.seed_pairs for record in records)
