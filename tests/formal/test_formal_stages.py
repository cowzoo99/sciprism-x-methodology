import json
from pathlib import Path

import pytest

from sciprism_x.formal.results import (
    REQUIRED_FORMAL_ARTIFACTS,
    FormalArtifactSpec,
    assemble_formal_results,
)
from sciprism_x.formal.stages import StageStore


def test_identical_stage_inputs_resume_without_recompute(tmp_path: Path) -> None:
    calls = 0

    def produce(path: Path) -> None:
        nonlocal calls
        calls += 1
        (path / "value.json").write_text('{"value": 1}\n', encoding="utf-8")

    store = StageStore(tmp_path)
    first = store.run(
        "responses",
        input_hashes={"cases": "a" * 64},
        implementation_hash="b" * 64,
        producer=produce,
    )
    second = store.run(
        "responses",
        input_hashes={"cases": "a" * 64},
        implementation_hash="b" * 64,
        producer=produce,
    )

    assert calls == 1
    assert first.content_hash == second.content_hash
    assert first.cache_key == second.cache_key
    assert first.resumed is False
    assert second.resumed is True


def test_changed_upstream_hash_invalidates_stage(tmp_path: Path) -> None:
    calls = 0

    def run(upstream: str):
        def produce(path: Path) -> None:
            nonlocal calls
            calls += 1
            (path / "value.txt").write_text(upstream, encoding="utf-8")

        return StageStore(tmp_path).run(
            "evaluation",
            input_hashes={"responses": upstream},
            implementation_hash="b" * 64,
            producer=produce,
        )

    first = run("a" * 64)
    second = run("c" * 64)

    assert calls == 2
    assert first.cache_key != second.cache_key
    assert first.path != second.path


def test_failed_partial_stage_is_never_resumed(tmp_path: Path) -> None:
    store = StageStore(tmp_path)

    def fail(path: Path) -> None:
        (path / "incomplete.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        store.run(
            "open_set",
            input_hashes={"evaluation": "a" * 64},
            implementation_hash="b" * 64,
            producer=fail,
        )

    receipt = store.run(
        "open_set",
        input_hashes={"evaluation": "a" * 64},
        implementation_hash="b" * 64,
        producer=lambda path: (path / "complete.txt").write_text(
            "complete", encoding="utf-8"
        ),
    )
    assert receipt.resumed is False
    assert not any(tmp_path.glob("*.partial"))


def test_changed_cached_artifact_is_rejected_instead_of_resumed(tmp_path: Path) -> None:
    store = StageStore(tmp_path)
    receipt = store.run(
        "statistics",
        input_hashes={"ablations": "a" * 64},
        implementation_hash="b" * 64,
        producer=lambda path: (path / "bootstrap.json").write_text(
            "original", encoding="utf-8"
        ),
    )
    (receipt.path / "bootstrap.json").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hashes"):
        store.run(
            "statistics",
            input_hashes={"ablations": "a" * 64},
            implementation_hash="b" * 64,
            producer=lambda _path: pytest.fail("corrupt cache must not resume or overwrite"),
        )


def test_formal_assembly_contains_every_required_namespaced_artifact(
    tmp_path: Path,
) -> None:
    store = StageStore(tmp_path / "stages")

    def produce(path: Path) -> None:
        for index, target in enumerate(REQUIRED_FORMAL_ARTIFACTS):
            source = path / "payload" / target.replace("/", "__")
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"artifact-{index}\n", encoding="utf-8")

    stage = store.run(
        "plotting_inputs",
        input_hashes={"statistics": "a" * 64},
        implementation_hash="b" * 64,
        producer=produce,
    )
    specs = tuple(
        FormalArtifactSpec(
            target=target,
            stage="plotting_inputs",
            source=f"payload/{target.replace('/', '__')}",
        )
        for target in REQUIRED_FORMAL_ARTIFACTS
    )
    receipt = assemble_formal_results(
        tmp_path / "runs",
        scale="smoke",
        stages=(stage,),
        artifacts=specs,
        computed_response_count=54,
        applicable_metric_cells=11,
        nonfinite_metric_cells=0,
        paper_sources_in_computed_rows=0,
    )

    assert set(dict(receipt.artifact_hashes)) == set(REQUIRED_FORMAL_ARTIFACTS)
    assert receipt.completed_stages == ("plotting_inputs",)
    assert receipt.computed_response_count == 54
    assert all((receipt.final_path / path).is_file() for path in REQUIRED_FORMAL_ARTIFACTS)
    saved = json.loads((receipt.final_path / "formal_run_receipt.json").read_text())
    assert saved["nonfinite_metric_cells"] == 0
