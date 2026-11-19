import json
import math
from pathlib import Path

import pandas as pd
import pytest

from sciprism_x.config import load_run_config
from sciprism_x.workflow import run_track_b


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ARTIFACTS = {
    "responses.parquet",
    "task_graphs.jsonl",
    "response_graphs.jsonl",
    "validator_outputs.parquet",
    "transport_plans/manifest.json",
    "per_case_metrics.parquet",
    "aggregate_metrics.json",
    "bootstrap_results.csv",
    "significance_results.csv",
    "ablation_results.csv",
    "config_resolved.yaml",
    "implementation_settings.json",
    "environment.json",
    "input_manifest.json",
    "source_hashes.json",
    "published_target_fit.json",
    "figures/manifest.json",
    "logs/workflow.log",
}


@pytest.fixture(scope="module")
def repeated_runs(tmp_path_factory: pytest.TempPathFactory):
    output_root = tmp_path_factory.mktemp("track_b_acceptance")
    config = load_run_config(PROJECT_ROOT / "configs" / "track_b.yaml")
    first = run_track_b(config, output_root=output_root / "first", fixture_count=12)
    second = run_track_b(config, output_root=output_root / "second", fixture_count=12)
    return config, first, second, output_root


def test_track_b_repeated_runs_have_identical_deterministic_artifacts(
    repeated_runs,
) -> None:
    _, first, second, output_root = repeated_runs

    assert first.final_path != second.final_path
    assert first.cache_hash == second.cache_hash
    assert first.artifact_hashes == second.artifact_hashes
    assert REQUIRED_ARTIFACTS <= set(dict(first.artifact_hashes))
    assert not list(output_root.rglob("*.partial"))


def test_track_b_fixture_artifacts_and_paper_controls(repeated_runs) -> None:
    config, receipt, _, _ = repeated_runs
    run_dir = receipt.final_path

    responses = pd.read_parquet(run_dir / "responses.parquet")
    metrics = pd.read_parquet(run_dir / "per_case_metrics.parquet")
    manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    aggregate = json.loads(
        (run_dir / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    bootstrap = pd.read_csv(run_dir / "bootstrap_results.csv")

    assert tuple(responses.columns) == (
        "task_id",
        "model_id",
        "replicate",
        "response_text",
    )
    assert len(responses) == len(metrics) == 12
    assert manifest["fixture_only"] is True
    assert manifest["fixture_case_count"] == 12
    assert aggregate["fixture_only"] is True
    assert aggregate["case_count"] == 12
    assert set(metrics["diagnostic_regime"]) <= {"EVALUATED", "INVALID"}
    numeric = metrics.select_dtypes(include="number")
    assert all(math.isfinite(float(value)) for value in numeric.to_numpy().ravel())
    assert set(bootstrap["resamples"]) == {1000}
    assert (run_dir / "logs" / "workflow.log").read_text(encoding="utf-8").endswith(
        "status=complete\n"
    )

    assert config.paper.seeds == (42, 0, 1, 123, 1234)
    assert config.generators.responses_per_task == 3
    assert config.regimes.nonstress_cases == 600
    assert config.regimes.boundary_stress_cases == 2400
    assert config.paper.bootstrap_resamples == 1000
    assert config.paper.confidence_level == 0.95
    assert config.paper.correction == "holm_bonferroni"
    assert config.datasets.dbp15k.full_candidate_pool is True
    assert config.datasets.dbp15k.candidate_pruning is False
