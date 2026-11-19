import json
from pathlib import Path


def test_writer_never_overwrites_existing_run(tmp_path: Path) -> None:
    from sciprism_x.results.writer import ArtifactWriter

    writer = ArtifactWriter(tmp_path)
    first = writer.create_run("a" * 64)
    second = writer.create_run("a" * 64)

    assert first.run_id != second.run_id
    assert first.cache_hash == second.cache_hash
    assert first.staging_path.is_dir()
    assert second.staging_path.is_dir()


def test_writer_fsyncs_hashes_and_atomically_finalizes(tmp_path: Path) -> None:
    from sciprism_x.results.writer import ArtifactWriter

    writer = ArtifactWriter(tmp_path)
    handle = writer.create_run("b" * 64)
    writer.write_json(handle, "aggregate_metrics.json", {"TSR": 1.0})
    writer.write_text(handle, "logs/workflow.log", "complete\n")
    receipt = writer.finalize(handle, track="B")

    assert receipt.final_path.is_dir()
    assert not handle.staging_path.exists()
    assert json.loads((receipt.final_path / "aggregate_metrics.json").read_text()) == {
        "TSR": 1.0
    }
    assert dict(receipt.artifact_hashes)["aggregate_metrics.json"]
    assert (receipt.final_path / "run_receipt.json").is_file()
