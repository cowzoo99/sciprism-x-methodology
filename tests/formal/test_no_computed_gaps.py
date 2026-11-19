from pathlib import Path

from scripts.inspect_formal_run import inspect_formal_run


def test_smoke_run_has_no_applicable_computed_gaps(formal_smoke_run: Path) -> None:
    audit = inspect_formal_run(formal_smoke_run)

    assert audit.missing_applicable_cells == ()
    assert audit.nonfinite_cells == ()
    assert audit.paper_sources_in_computed == ()
    assert audit.missing_baselines == ()
    assert audit.missing_ablations == ()
    assert audit.missing_statistics == ()
    assert audit.artifact_hash_mismatches == ()
