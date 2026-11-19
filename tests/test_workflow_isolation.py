import json
from pathlib import Path

import pyarrow.parquet as parquet
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_track_a_rejects_published_target_path(tmp_path: Path) -> None:
    from sciprism_x.config import TrackIsolationError, load_run_config
    from sciprism_x.workflow import run_track_a

    config = load_run_config(PROJECT_ROOT / "configs" / "track_a.yaml").model_copy(
        update={"published_targets": "data/manifests/published_targets.json"}
    )

    with pytest.raises(TrackIsolationError, match="published"):
        run_track_a(config, output_root=tmp_path, fixture_count=2)


def test_track_a_never_uses_deterministic_reconstruction(tmp_path: Path) -> None:
    from sciprism_x.config import TrackIsolationError, load_run_config
    from sciprism_x.workflow import run_track_a

    config = load_run_config(PROJECT_ROOT / "configs" / "track_a.yaml")

    with pytest.raises(TrackIsolationError, match="supplied or exactly acquired"):
        run_track_a(config, output_root=tmp_path, fixture_count=2)


def test_track_b_fixture_run_writes_complete_nonoverwriting_artifacts(
    tmp_path: Path,
) -> None:
    from sciprism_x.config import load_run_config
    from sciprism_x.workflow import REQUIRED_ARTIFACTS, run_track_b

    config = load_run_config(PROJECT_ROOT / "configs" / "track_b.yaml")
    first = run_track_b(config, output_root=tmp_path, fixture_count=2)
    second = run_track_b(config, output_root=tmp_path, fixture_count=2)

    assert first.run_id != second.run_id
    assert first.cache_hash == second.cache_hash
    assert first.final_path.is_dir()
    assert all((first.final_path / path).exists() for path in REQUIRED_ARTIFACTS)
    assert (first.final_path / "published_target_fit.json").is_file()
    assert parquet.read_table(first.final_path / "responses.parquet").num_rows == 2
    manifest = json.loads((first.final_path / "input_manifest.json").read_text())
    assert manifest["acquisition"]["route"] == "deterministic_reconstruction"
    assert "route" not in parquet.read_table(
        first.final_path / "responses.parquet"
    ).column_names
    assert any((first.final_path / "transport_plans").iterdir())
    assert any((first.final_path / "logs").iterdir())
    assert any((first.final_path / "figures").iterdir())
