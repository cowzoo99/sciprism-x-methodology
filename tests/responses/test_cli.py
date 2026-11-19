import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_cli_writes_repeatable_ordinary_rows_and_manifest(tmp_path) -> None:
    import pyarrow.parquet as parquet

    from sciprism_x.responses.reconstruct import main

    output = tmp_path / "fixture"
    arguments = [
        "--config",
        str(PROJECT_ROOT / "configs" / "track_b.yaml"),
        "--fixture-count",
        "12",
        "--output-dir",
        str(output),
    ]

    assert main(arguments) == 0
    first_hashes = (_hash(output / "responses.parquet"), _hash(output / "input_manifest.json"))
    assert main(arguments) == 0
    second_hashes = (_hash(output / "responses.parquet"), _hash(output / "input_manifest.json"))

    assert first_hashes == second_hashes
    table = parquet.read_table(output / "responses.parquet")
    assert table.num_rows == 12
    assert set(table.column_names) == {
        "task_id",
        "model_id",
        "replicate",
        "response_text",
    }
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 12
    assert manifest["acquisition"]["route"] == "deterministic_reconstruction"
    assert manifest["responses_sha256"] == first_hashes[0]
