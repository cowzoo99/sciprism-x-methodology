from pathlib import Path

from sciprism_x.formal.datasets import (
    extract_dbp15k,
    load_dbp15k_dataset,
    load_scicueval_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def test_scicueval_parser_reads_all_verified_records() -> None:
    cases = load_scicueval_cases(DATA_ROOT / "raw" / "scicueval")

    assert len(cases) == 11_343
    assert len({case.task_id for case in cases}) == 11_343
    assert all(case.prompt and case.reference_context and case.reference_answer for case in cases)
    assert {case.competency for case in cases} == {
        "Context-aware_Inference",
        "Information-absence_Detection",
        "Information_Integration",
        "Relevant_Information_Identification",
    }
    assert {case.domain for case in cases} == {
        "biology",
        "chemistry",
        "physics",
        "biomedicine",
        "materials_science",
    }


def test_dbp_parser_uses_0_3_full_candidate_protocol(tmp_path: Path) -> None:
    root = extract_dbp15k(
        DATA_ROOT / "raw" / "dbp15k" / "dbp15k.tar.gz",
        tmp_path,
    )
    dataset = load_dbp15k_dataset(root)

    assert set(dataset.language_pairs) == {"ZH-EN", "JA-EN", "FR-EN"}
    for pair in dataset.language_pairs.values():
        assert pair.seed_fraction == 0.30
        assert pair.candidate_pruning is False
        assert len(pair.seed_pairs) == 4_500
        assert len(pair.reference_pairs) == 10_500
        assert len(pair.target_candidate_ids) == 15_000
        assert pair.triples_1
        assert pair.triples_2


def test_dbp_extraction_is_idempotent_and_contained(tmp_path: Path) -> None:
    archive = DATA_ROOT / "raw" / "dbp15k" / "dbp15k.tar.gz"
    first = extract_dbp15k(archive, tmp_path)
    second = extract_dbp15k(archive, tmp_path)

    assert first == second
    assert first.parent == tmp_path.resolve()
    assert (first / "zh_en" / "0_3" / "ref_ent_ids").is_file()
