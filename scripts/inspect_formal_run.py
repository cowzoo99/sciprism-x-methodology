from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sciprism_x.baselines.registry import PAPER_BASELINES
from sciprism_x.formal.ablations import AblationKind
from sciprism_x.formal.results import REQUIRED_FORMAL_ARTIFACTS


@dataclass(frozen=True, slots=True)
class FormalRunAudit:
    run_root: Path
    scale: str
    scicu_responses: int
    missing_applicable_cells: tuple[str, ...]
    nonfinite_cells: tuple[str, ...]
    paper_sources_in_computed: tuple[str, ...]
    missing_baselines: tuple[str, ...]
    missing_ablations: tuple[str, ...]
    missing_statistics: tuple[str, ...]
    artifact_hash_mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.missing_applicable_cells,
                self.nonfinite_cells,
                self.paper_sources_in_computed,
                self.missing_baselines,
                self.missing_ablations,
                self.missing_statistics,
                self.artifact_hash_mismatches,
            )
        )


_NUMERIC_COLUMNS = {
    "per_case_metrics": (
        "tsr",
        "pcs",
        "residual_consistency",
        "node_loss",
        "structure_loss",
        "constraint_loss",
    ),
    "tuning_metrics": (
        "tsr",
        "pcs",
        "residual_consistency",
        "node_loss",
        "structure_loss",
        "constraint_loss",
    ),
    "baselines": ("score",),
    "baseline_boundaries": ("rho_star", "rho_max", "bsi_raw", "bsi_stability"),
    "baseline_open_set": ("updq", "structural_calibration", "calibration_ece"),
    "ablations": ("main_value", "metric_value", "delta"),
    "boundaries": ("rho_star", "rho_max", "bsi_raw", "bsi_stability"),
    "open_set": (
        "purity",
        "stability",
        "separation",
        "separation_normalized",
        "updq",
        "structural_calibration",
        "calibration_ece",
    ),
    "bootstrap_results": (
        "observed_difference",
        "ci_lower",
        "ci_upper",
        "p_value",
        "resamples",
        "confidence_level",
        "seed",
    ),
    "significance_results": ("raw_p_value", "adjusted_p_value"),
}
_ARRAY_COLUMNS = {
    "per_case_metrics": ("residuals", "state_vector"),
    "tuning_metrics": ("residuals", "state_vector"),
    "boundaries": ("evaluated_radii",),
    "baseline_boundaries": ("evaluated_radii",),
    "open_set": ("validation_mean_silhouettes", "unknownness"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_run_root(root: Path, *, latest_complete: bool) -> Path:
    candidate = Path(root).resolve()
    if (candidate / "formal_run_receipt.json").is_file():
        return candidate
    if not latest_complete:
        raise ValueError("root is not a finalized formal run")
    completed = [
        path
        for path in candidate.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "formal_run_receipt.json").is_file()
    ]
    if not completed:
        raise ValueError("no finalized formal run exists under root")
    return max(completed, key=lambda path: path.stat().st_mtime_ns).resolve()


def _finite_array(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.size > 0 and bool(np.isfinite(array).all())


def _load_frames(run_root: Path) -> dict[str, pd.DataFrame]:
    computed = run_root / "computed"
    return {
        name: pd.read_parquet(computed / f"{name}.parquet")
        for name in (
            "responses",
            "tuning_responses",
            "per_case_metrics",
            "tuning_metrics",
            "baselines",
            "baseline_boundaries",
            "baseline_open_set",
            "ablations",
            "boundaries",
            "open_set",
            "bootstrap_results",
            "significance_results",
        )
    }


def inspect_formal_run(root: Path, *, latest_complete: bool = False) -> FormalRunAudit:
    run_root = _resolve_run_root(root, latest_complete=latest_complete)
    receipt = json.loads(
        (run_root / "formal_run_receipt.json").read_text(encoding="utf-8")
    )
    frames = _load_frames(run_root)

    hash_mismatches: list[str] = []
    expected_hashes = dict(receipt.get("artifact_hashes", {}))
    required = set(REQUIRED_FORMAL_ARTIFACTS)
    for relative in sorted(required | set(expected_hashes)):
        path = run_root.joinpath(*PurePosixPath(relative).parts)
        if relative not in required:
            hash_mismatches.append(f"unexpected:{relative}")
        elif relative not in expected_hashes:
            hash_mismatches.append(f"unreceipted:{relative}")
        elif not path.is_file():
            hash_mismatches.append(f"missing:{relative}")
        elif _sha256(path) != expected_hashes[relative]:
            hash_mismatches.append(f"sha256:{relative}")

    missing_cells: list[str] = []
    nonfinite: list[str] = []
    for name, columns in _NUMERIC_COLUMNS.items():
        frame = frames[name]
        for column in columns:
            if column not in frame:
                missing_cells.append(f"{name}.{column}:missing_column")
                continue
            missing_rows = frame.index[frame[column].isna()].tolist()
            missing_cells.extend(f"{name}.{column}:{row}" for row in missing_rows)
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            bad_rows = frame.index[~np.isfinite(values)].tolist()
            nonfinite.extend(f"{name}.{column}:{row}" for row in bad_rows)
    for name, columns in _ARRAY_COLUMNS.items():
        frame = frames[name]
        for column in columns:
            if column not in frame:
                missing_cells.append(f"{name}.{column}:missing_column")
                continue
            nonfinite.extend(
                f"{name}.{column}:{row}"
                for row, value in frame[column].items()
                if not _finite_array(value)
            )

    expected_response_count = int(receipt.get("computed_response_count", -1))
    if len(frames["responses"]) != expected_response_count:
        missing_cells.append(
            f"responses.rows:{len(frames['responses'])}!={expected_response_count}"
        )
    if len(frames["per_case_metrics"]) != expected_response_count:
        missing_cells.append(
            "per_case_metrics.rows:"
            f"{len(frames['per_case_metrics'])}!={expected_response_count}"
        )
    if len(frames["tuning_responses"]) != len(frames["tuning_metrics"]):
        missing_cells.append(
            "tuning.rows:"
            f"{len(frames['tuning_responses'])}!={len(frames['tuning_metrics'])}"
        )
    if set(frames["tuning_metrics"].get("split", ())) != {"train", "validation"}:
        missing_cells.append("tuning_metrics.split:expected_train_and_validation")
    tuning_ids = set(frames["tuning_metrics"].get("original_task_id", ()))
    final_ids = set(frames["per_case_metrics"].get("original_task_id", ()))
    if tuning_ids & final_ids:
        missing_cells.append("tuning_metrics.original_task_id:overlaps_final_test")

    paper_sources: list[str] = []
    for name, frame in frames.items():
        if "source" not in frame:
            continue
        paper_sources.extend(
            f"{name}:{row}:{source}"
            for row, source in frame["source"].items()
            if source != "computed"
        )
    aggregate = json.loads(
        (run_root / "computed" / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    if aggregate.get("source") != "computed":
        paper_sources.append(f"aggregate_metrics:{aggregate.get('source')}")
    for key, value in aggregate.items():
        if key != "source" and (
            not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            nonfinite.append(f"aggregate_metrics.{key}")

    baseline_selection = json.loads(
        (run_root / "computed" / "baseline_selection.json").read_text(
            encoding="utf-8"
        )
    )
    expected_tuned = {
        "uncertainty",
        "knn_support",
        "centroid_support",
        "graph_edit",
        "structure_aware",
        "sinkhorn",
        "gromov",
    }
    if set(baseline_selection.get("trials", {})) != expected_tuned:
        missing_cells.append("baseline_selection.trials:incomplete_config_audit")
    if baseline_selection.get("trial_count_per_tunable_baseline") != 50:
        missing_cells.append("baseline_selection.trial_count:expected_50")
    expected_objectives = {
        "uncertainty": "negative_structural_validity_brier",
        "knn_support": "negative_open_set_novelty_brier",
        "centroid_support": "negative_open_set_novelty_brier",
        "graph_edit": "negative_structural_validity_brier",
        "structure_aware": "negative_structural_validity_brier",
        "sinkhorn": "mean_frozen_alignment_similarity",
        "gromov": "mean_frozen_alignment_similarity",
    }
    if baseline_selection.get("objectives") != expected_objectives:
        missing_cells.append("baseline_selection.objectives:contract_mismatch")
    judge_contract = baseline_selection.get("llm_judge", {})
    expected_judge_fields = {
        "answer_quality",
        "evidence_support",
        "reasoning_consistency",
        "uncertainty_handling",
        "final_score",
        "brief_error_type",
    }
    if set(judge_contract.get("required_fields", ())) != expected_judge_fields:
        missing_cells.append("baseline_selection.llm_judge:field_contract_mismatch")
    judge_route = judge_contract.get("route")
    judge_source = judge_contract.get("source_sha256")
    if judge_route == "exact_gpt4o_response_replay":
        if not isinstance(judge_source, str) or len(judge_source) != 64:
            missing_cells.append("baseline_selection.llm_judge:missing_replay_hash")
    elif judge_route != "local_inferred_when_exact_responses_are_not_supplied":
        missing_cells.append("baseline_selection.llm_judge:unknown_route")
    for kind, trial in baseline_selection.get("trials", {}).items():
        if trial.get("trial_count") != 50:
            missing_cells.append(f"baseline_selection.{kind}.trial_count:expected_50")
        score = trial.get("validation_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            nonfinite.append(f"baseline_selection.{kind}.validation_score")

    implementation_contract = json.loads(
        (run_root / "computed" / "implementation_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if implementation_contract.get("source") != "computed":
        paper_sources.append(
            f"implementation_contract:{implementation_contract.get('source')}"
        )
    parameter_records = implementation_contract.get("resolved_parameters", [])
    parameter_paths = {
        row.get("path")
        for row in parameter_records
        if isinstance(row, dict)
    }
    required_parameter_paths = {
        "baseline_reconstruction.endpoint_case_sensitive",
        "baseline_reconstruction.endpoint_strip_whitespace",
        "baseline_reconstruction.endpoint_collapse_whitespace",
        "baseline_reconstruction.attribution_lowercase",
        "baseline_reconstruction.attribution_token_pattern",
        "baseline_reconstruction.attribution_empty_overlap_score",
        "baseline_reconstruction.taxonomy_patterns",
        "baseline_reconstruction.taxonomy_no_failure_score",
        "baseline_reconstruction.taxonomy_failure_score",
        "baseline_reconstruction.taxonomy_lowercase",
        "baseline_reconstruction.uncertainty_maximum_entropy",
        "baseline_reconstruction.support_clip_to_unit_interval",
        "baseline_reconstruction.transport_score_transform",
        "baseline_reconstruction.relation_multiset",
        "baseline_reconstruction.ranking_descending",
        "baseline_reconstruction.representation_dimensions",
        "baseline_reconstruction.task_success_threshold",
        "baseline_reconstruction.matched_context_rotations",
        "baseline_reconstruction.local_judge_component_weights",
        "baseline_reconstruction.dbp_validation_transport_epsilon",
        "baseline_reconstruction.support_neighbors_grid",
        "baseline_reconstruction.support_distance_scale_grid",
        "baseline_reconstruction.uncertainty_mode_codes",
        "baseline_reconstruction.graph_edit_cost_grid",
        "baseline_reconstruction.graph_edit_timeout_seconds",
        "dbp_reconstruction.signature_neighborhood_hops",
        "dbp_reconstruction.local_transport_hops",
        "dbp_reconstruction.signature_cosine_weight",
        "dbp_reconstruction.relation_overlap_weight",
        "validators.tfidf_proxy.token_pattern",
        "validators.nli_proxy_feature_weights",
        "state_field.initialization",
        "transport.greedy_mass_tolerance",
        "transport.greedy_lp_method",
    }
    if not required_parameter_paths.issubset(parameter_paths):
        missing_cells.append(
            "implementation_contract.resolved_parameters:missing_settings"
        )
    if any(
        row.get("provenance") not in {"PAPER_EXPLICIT", "IMPLEMENTATION_SETTING"}
        for row in parameter_records
        if isinstance(row, dict)
    ):
        missing_cells.append(
            "implementation_contract.resolved_parameters:invalid_provenance"
        )
    formal_parameters = implementation_contract.get("formal_execution_parameters", {})
    if (
        formal_parameters.get("provenance") != "IMPLEMENTATION_SETTING"
        or not isinstance(formal_parameters.get("values"), dict)
    ):
        missing_cells.append(
            "implementation_contract.formal_execution_parameters:missing"
        )
    scicu_allocations = implementation_contract.get("final_allocations", {}).get(
        "scicueval", []
    )
    if any(not row.get("valid", False) for row in scicu_allocations):
        missing_cells.append("implementation_contract.scicueval:invalid_perturbation")
    dbp_allocations = implementation_contract.get("final_allocations", {}).get(
        "dbp15k", []
    )
    dbp_perturbations = implementation_contract.get("final_allocations", {}).get(
        "dbp15k_perturbation_audit", []
    )
    if any(not row.get("valid", False) for row in dbp_perturbations):
        missing_cells.append("implementation_contract.dbp15k:invalid_perturbation")
    expected_dbp_boundary = sum(
        row.get("regime") == "boundary_stress" for row in dbp_allocations
    )
    if len(dbp_perturbations) != expected_dbp_boundary:
        missing_cells.append("implementation_contract.dbp15k:boundary_audit_count")
    dbp_selection = implementation_contract.get("baseline_selection_artifact")
    if dbp_selection != "computed/baseline_selection.json":
        missing_cells.append("implementation_contract.baseline_selection_artifact")
    external_assets = implementation_contract.get("external_assets", {}).get(
        "external_inputs", {}
    )
    expected_external_inputs = {
        "dbp15k_original_split",
        "scicueval_train_validation_split",
        "supplied_model_responses",
        "baseline_configuration",
        "dbp15k_prior_system_outputs",
        "llm_judge_responses",
    }
    if set(external_assets) != expected_external_inputs:
        missing_cells.append(
            "implementation_contract.external_inputs:incomplete_contract"
        )
    baseline_config_contract = external_assets.get("baseline_configuration", {})
    baseline_reconstruction = baseline_config_contract.get(
        "baseline_reconstruction", {}
    )
    if baseline_selection.get("reconstruction_configuration") != baseline_reconstruction:
        missing_cells.append(
            "baseline_selection.reconstruction_configuration:contract_mismatch"
        )
    if (
        baseline_selection.get("dbp_reconstruction_configuration")
        != baseline_config_contract.get("dbp_reconstruction", {})
    ):
        missing_cells.append(
            "baseline_selection.dbp_reconstruction_configuration:contract_mismatch"
        )
    required_baseline_parameters = {
        "endpoint_case_sensitive",
        "endpoint_strip_whitespace",
        "endpoint_collapse_whitespace",
        "attribution_lowercase",
        "attribution_token_pattern",
        "attribution_empty_overlap_score",
        "taxonomy_patterns",
        "taxonomy_no_failure_score",
        "taxonomy_failure_score",
        "taxonomy_lowercase",
        "uncertainty_maximum_entropy",
        "support_clip_to_unit_interval",
        "transport_score_transform",
        "relation_multiset",
        "ranking_descending",
        "representation_dimensions",
        "task_success_threshold",
        "matched_context_rotations",
        "local_judge_component_weights",
        "dbp_validation_transport_epsilon",
        "support_neighbors_grid",
        "support_distance_scale_grid",
        "uncertainty_mode_codes",
        "graph_edit_cost_grid",
        "graph_edit_timeout_seconds",
    }
    if not required_baseline_parameters.issubset(baseline_reconstruction):
        missing_cells.append(
            "implementation_contract.baseline_configuration:missing_parameters"
        )
    dbp_prior_outputs = external_assets.get("dbp15k_prior_system_outputs", {})
    if (
        dbp_prior_outputs.get("status")
        != "not_supplied_use_configured_adapter"
        or dbp_prior_outputs.get("replacement")
        != "concept_compatible_deterministic_adapter"
        or dbp_prior_outputs.get("scope")
        != "auxiliary_ranking_and_structural_matching_comparators"
    ):
        missing_cells.append(
            "implementation_contract.dbp15k_prior_system_outputs:invalid"
        )
    judge_weights = baseline_reconstruction.get("local_judge_component_weights", ())
    if (
        not isinstance(judge_weights, list)
        or len(judge_weights) != 5
        or any(not isinstance(value, (int, float)) or value < 0.0 for value in judge_weights)
        or not math.isclose(
            sum(judge_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        missing_cells.append(
            "implementation_contract.local_judge_component_weights:invalid"
        )
    dbp_reconstruction = baseline_config_contract.get("dbp_reconstruction", {})
    if not {
        "signature_neighborhood_hops",
        "local_transport_hops",
        "signature_cosine_weight",
        "relation_overlap_weight",
    }.issubset(dbp_reconstruction):
        missing_cells.append(
            "implementation_contract.dbp_reconstruction:missing_parameters"
        )
    dbp_weights = (
        dbp_reconstruction.get("signature_cosine_weight"),
        dbp_reconstruction.get("relation_overlap_weight"),
    )
    if (
        any(not isinstance(value, (int, float)) or value < 0.0 for value in dbp_weights)
        or not math.isclose(
            sum(dbp_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        missing_cells.append("implementation_contract.dbp_reconstruction:invalid")
    judge_asset = external_assets.get("llm_judge_responses", {})
    if judge_route == "exact_gpt4o_response_replay":
        if (
            judge_asset.get("status") != "frozen_response_replay_supplied"
            or judge_asset.get("source_sha256") != judge_source
        ):
            missing_cells.append(
                "implementation_contract.llm_judge_responses:replay_mismatch"
            )
    elif (
        judge_asset.get("status") != "external_response_dump_not_supplied"
        or judge_asset.get("source_sha256") is not None
    ):
        missing_cells.append(
            "implementation_contract.llm_judge_responses:local_route_mismatch"
        )

    baseline_frame = frames["baselines"]
    missing_baselines: list[str] = []
    baseline_key = ["case_id", "dataset", "baseline", "metric"]
    duplicates = baseline_frame.duplicated(baseline_key, keep=False)
    missing_baselines.extend(
        f"duplicate:{row}" for row in baseline_frame.index[duplicates].tolist()
    )
    missing_baselines.extend(
        f"unavailable:{row}:{status}"
        for row, status in baseline_frame["status"].items()
        if status != "available"
    )
    expected_kinds_by_dataset = {
        dataset: {
            definition.kind.value
            for definition in PAPER_BASELINES
            if dataset in definition.benchmarks
        }
        for dataset in set(baseline_frame["dataset"])
    }
    for dataset, expected_kinds in expected_kinds_by_dataset.items():
        observed_kinds = set(
            baseline_frame.loc[baseline_frame["dataset"] == dataset, "baseline"]
        )
        missing_baselines.extend(
            f"{dataset}:all_cases:{baseline}"
            for baseline in sorted(expected_kinds - observed_kinds)
        )
    expected_by_dataset = {
        dataset: set(zip(group["baseline"], group["metric"], strict=True))
        for dataset, group in baseline_frame.groupby("dataset")
    }
    for (case_id, dataset), group in baseline_frame.groupby(["case_id", "dataset"]):
        observed = set(zip(group["baseline"], group["metric"], strict=True))
        for baseline, metric in sorted(expected_by_dataset[dataset] - observed):
            missing_baselines.append(f"{dataset}:{case_id}:{baseline}:{metric}")

    ablation_frame = frames["ablations"]
    expected_ablations = {kind.value for kind in AblationKind}
    missing_ablations: list[str] = []
    if set(ablation_frame["ablation"]) != expected_ablations:
        missing_ablations.extend(
            f"missing_kind:{kind}"
            for kind in sorted(expected_ablations - set(ablation_frame["ablation"]))
        )
    for key, group in ablation_frame.groupby(["case_id", "dataset", "metric"]):
        observed = set(group["ablation"])
        if observed != expected_ablations or len(group) != len(expected_ablations):
            missing_ablations.append(f"paired:{key}")

    bootstrap = frames["bootstrap_results"]
    significance = frames["significance_results"]
    missing_statistics: list[str] = []
    bootstrap_ids = set(bootstrap["comparison_id"])
    significance_ids = set(significance["comparison_id"])
    if not bootstrap_ids:
        missing_statistics.append("bootstrap:empty")
    if bootstrap_ids != significance_ids:
        missing_statistics.append("comparison_ids:unpaired")
    if bootstrap["comparison_id"].duplicated().any():
        missing_statistics.append("bootstrap:duplicate_comparison_id")
    if significance["comparison_id"].duplicated().any():
        missing_statistics.append("significance:duplicate_comparison_id")
    if set(bootstrap["resamples"]) != {1000}:
        missing_statistics.append("bootstrap:resamples_not_1000")

    return FormalRunAudit(
        run_root=run_root,
        scale=str(receipt.get("scale")),
        scicu_responses=len(frames["responses"]),
        missing_applicable_cells=tuple(sorted(set(missing_cells))),
        nonfinite_cells=tuple(sorted(set(nonfinite))),
        paper_sources_in_computed=tuple(sorted(set(paper_sources))),
        missing_baselines=tuple(sorted(set(missing_baselines))),
        missing_ablations=tuple(sorted(set(missing_ablations))),
        missing_statistics=tuple(sorted(set(missing_statistics))),
        artifact_hash_mismatches=tuple(sorted(set(hash_mismatches))),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a finalized formal run")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--latest-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = inspect_formal_run(args.root, latest_complete=args.latest_complete)
    print(f"run_root={audit.run_root}")
    print(f"scale={audit.scale}")
    print(f"scicu_responses={audit.scicu_responses}")
    print(f"missing_applicable_cells={len(audit.missing_applicable_cells)}")
    print(f"nonfinite_cells={len(audit.nonfinite_cells)}")
    print(f"paper_sources_in_computed={len(audit.paper_sources_in_computed)}")
    print(f"missing_baselines={len(audit.missing_baselines)}")
    print(f"missing_ablations={len(audit.missing_ablations)}")
    print(f"missing_statistics={len(audit.missing_statistics)}")
    print(f"artifact_hash_mismatches={len(audit.artifact_hash_mismatches)}")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
