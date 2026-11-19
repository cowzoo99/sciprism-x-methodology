from collections import Counter
import json


def _make_cases(count: int):
    from sciprism_x.data.schema import SciCUCase

    domains = ("physics", "chemistry", "biology", "materials", "earth", "astronomy")
    competencies = ("symbolic", "unit", "causal", "evidence", "modality", "uncertainty")
    motifs = ("symbolic", "unit", "causal", "evidence", "modality", "uncertainty")
    return tuple(
        SciCUCase(
            task_id=f"SCICU-{index:04d}",
            prompt=f"Prompt {index}",
            reference_context=f"Context {index}",
            reference_answer=f"Answer {index}",
            domain=domains[index % len(domains)],
            competency=competencies[(index // len(domains)) % len(competencies)],
            motif_family=motifs[(index // (len(domains) * len(competencies))) % len(motifs)],
            source_hash="a" * 64,
        )
        for index in range(count)
    )


def test_regime_counts_disjointness_and_repeatability() -> None:
    from sciprism_x.data.splits import build_regime_splits

    cases = _make_cases(3600)
    first = build_regime_splits(
        cases,
        seed=42,
        nonstress_count=600,
        stress_count=2400,
    )
    second = build_regime_splits(
        cases,
        seed=42,
        nonstress_count=600,
        stress_count=2400,
    )

    assert first == second
    assert len(first.iid) == 600
    assert len(first.motif) == 600
    assert len(first.domain) == 600
    assert len(first.boundary) == 2400
    assert set(first.iid).isdisjoint(first.motif)
    assert set(first.iid).isdisjoint(first.domain)
    assert set(first.motif).isdisjoint(first.domain)


def test_iid_split_preserves_balanced_domain_strata() -> None:
    from sciprism_x.data.splits import build_regime_splits

    cases = _make_cases(3600)
    by_id = {case.task_id: case for case in cases}
    splits = build_regime_splits(
        cases,
        seed=42,
        nonstress_count=600,
        stress_count=2400,
    )

    counts = Counter(by_id[task_id].domain for task_id in splits.iid)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_split_rejects_insufficient_unique_nonstress_cases() -> None:
    from sciprism_x.data.splits import build_regime_splits

    cases = _make_cases(1000)
    try:
        build_regime_splits(cases, seed=42, nonstress_count=600, stress_count=800)
    except ValueError as error:
        assert "non-stress" in str(error)
    else:
        raise AssertionError("undersized non-stress pool must be rejected")


def test_dbp_splits_are_balanced_by_language_pair() -> None:
    from sciprism_x.data.schema import DBPCase
    from sciprism_x.data.splits import build_dbp_regime_splits

    cases = tuple(
        DBPCase(
            pair_id=f"{language_pair}-{index:04d}",
            source_entity_id=f"source-{index}",
            target_entity_id=f"target-{index}",
            language_pair=language_pair,
            source_relations=(),
            target_relations=(),
            source_hash="b" * 64,
        )
        for language_pair in ("ZH-EN", "JA-EN", "FR-EN")
        for index in range(900)
    )

    splits = build_dbp_regime_splits(
        cases,
        seed=42,
        nonstress_count=600,
        stress_count=2400,
    )
    by_id = {case.pair_id: case for case in cases}

    for split_ids, expected_per_pair in (
        (splits.iid, 200),
        (splits.motif, 200),
        (splits.domain, 200),
        (splits.boundary, 800),
    ):
        counts = Counter(by_id[pair_id].language_pair for pair_id in split_ids)
        assert set(counts.values()) == {expected_per_pair}


def test_split_manifest_records_seed_and_ids_atomically(tmp_path) -> None:
    from sciprism_x.data.schema import RegimeSplits
    from sciprism_x.data.splits import write_split_manifest

    splits = RegimeSplits(
        seed=42,
        iid=("a",),
        motif=("b",),
        domain=("c",),
        boundary=("d", "e"),
    )
    target = tmp_path / "splits" / "scicueval-42.json"

    result = write_split_manifest(splits, target)

    assert result == target
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "seed": 42,
        "iid": ["a"],
        "motif": ["b"],
        "domain": ["c"],
        "boundary": ["d", "e"],
    }
    assert not target.with_suffix(".json.partial").exists()
