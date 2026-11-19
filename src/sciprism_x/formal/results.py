from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Literal

from sciprism_x.results.schema import FormalRunReceipt, StageReceipt

from .stages import canonical_sha256, verify_stage_receipt


REQUIRED_FORMAL_ARTIFACTS = (
    "computed/responses.parquet",
    "computed/tuning_responses.parquet",
    "computed/method_selection.json",
    "computed/per_case_metrics.parquet",
    "computed/tuning_metrics.parquet",
    "computed/baseline_selection.json",
    "computed/baselines.parquet",
    "computed/baseline_boundaries.parquet",
    "computed/baseline_open_set.parquet",
    "computed/ablations.parquet",
    "computed/boundaries.parquet",
    "computed/open_set.parquet",
    "computed/bootstrap_results.parquet",
    "computed/significance_results.parquet",
    "computed/aggregate_metrics.json",
    "computed/implementation_contract.json",
    "computed/sample_manifest.parquet",
    "computed/task_graphs.jsonl",
    "computed/response_graphs.jsonl",
    "computed/validator_outputs.parquet",
    "computed/transport_plans.npz",
    "computed/transport_diagnostics.parquet",
    "computed/residual_components.parquet",
    "computed/per_instance_outputs.parquet",
    "computed/state_assignments.parquet",
    "computed/state_training_manifest.json",
    "computed/state_models.pt",
    "computed/effective_hyperparameters.json",
    "computed/learned_modules.json",
    "computed/solver_environment.json",
    "computed/protocol_manifest.json",
    "computed/plotting_input_manifest.json",
    "release/motif_dictionary.yaml",
    "release/validator_registry.yaml",
    "release/annotation_guidelines.md",
    "release/availability.json",
    "release/package_contract.json",
    "release/release_package.yaml",
    "release/plotting.yaml",
    "figures/aggregate_metrics.svg",
    "figures/ablation_delta_heatmap.svg",
    "figures/figure_manifest.json",
    "paper_reference/published_metrics.json",
    "comparison/discrepancy_report.csv",
)
_NAMESPACES = {"computed", "release", "figures", "paper_reference", "comparison"}


@dataclass(frozen=True, slots=True)
class FormalArtifactSpec:
    target: str
    stage: str
    source: str


def _relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_specs(
    artifacts: tuple[FormalArtifactSpec, ...],
) -> tuple[tuple[FormalArtifactSpec, PurePosixPath, PurePosixPath], ...]:
    if not artifacts:
        raise ValueError("formal result assembly requires artifact specifications")
    resolved = []
    targets = []
    for spec in artifacts:
        target = _relative(spec.target, label="formal artifact target")
        source = _relative(spec.source, label="stage artifact source")
        if target.parts[0] not in _NAMESPACES:
            raise ValueError("formal artifacts must use a declared namespace")
        if not spec.stage:
            raise ValueError("formal artifact stage must be non-empty")
        targets.append(target.as_posix())
        resolved.append((spec, target, source))
    if len(targets) != len(set(targets)):
        raise ValueError("formal artifact targets must be unique")
    missing = sorted(set(REQUIRED_FORMAL_ARTIFACTS) - set(targets))
    if missing:
        raise ValueError(f"formal artifact manifest is incomplete: {missing}")
    return tuple(resolved)


def _saved_run_receipt(
    final: Path,
    *,
    stages: tuple[StageReceipt, ...],
    run_id: str,
    scale: Literal["smoke", "paper"],
    expected_artifacts: tuple[str, ...],
    counts: tuple[int, int, int, int],
) -> FormalRunReceipt:
    payload = json.loads((final / "formal_run_receipt.json").read_text(encoding="utf-8"))
    artifact_hashes = tuple(sorted(dict(payload["artifact_hashes"]).items()))
    expected_stage_hashes = {stage.name: stage.content_hash for stage in stages}
    saved_counts = (
        int(payload["computed_response_count"]),
        int(payload["applicable_metric_cells"]),
        int(payload["nonfinite_metric_cells"]),
        int(payload["paper_sources_in_computed_rows"]),
    )
    if (
        payload["run_id"] != run_id
        or payload["scale"] != scale
        or tuple(payload["completed_stages"]) != tuple(stage.name for stage in stages)
        or dict(payload["stage_content_hashes"]) != expected_stage_hashes
        or set(dict(artifact_hashes)) != set(expected_artifacts)
        or saved_counts != counts
    ):
        raise ValueError("formal run receipt metadata does not match its cache key")
    actual = tuple(
        (relative, _sha256(final.joinpath(*PurePosixPath(relative).parts)))
        for relative, _digest in artifact_hashes
    )
    if actual != artifact_hashes:
        raise ValueError("formal run artifact hashes do not match its receipt")
    return FormalRunReceipt(
        run_id=run_id,
        scale=scale,
        final_path=final,
        stage_receipts=stages,
        artifact_hashes=artifact_hashes,
        computed_response_count=counts[0],
        applicable_metric_cells=counts[1],
        nonfinite_metric_cells=counts[2],
        paper_sources_in_computed_rows=counts[3],
    )


def assemble_formal_results(
    output_root: Path,
    *,
    scale: Literal["smoke", "paper"],
    stages: tuple[StageReceipt, ...],
    artifacts: tuple[FormalArtifactSpec, ...],
    computed_response_count: int,
    applicable_metric_cells: int,
    nonfinite_metric_cells: int,
    paper_sources_in_computed_rows: int,
) -> FormalRunReceipt:
    if scale not in ("smoke", "paper"):
        raise ValueError("formal scale must be smoke or paper")
    if not stages or len({stage.name for stage in stages}) != len(stages):
        raise ValueError("formal assembly requires unique completed stages")
    counts = (
        computed_response_count,
        applicable_metric_cells,
        nonfinite_metric_cells,
        paper_sources_in_computed_rows,
    )
    if any(value < 0 for value in counts):
        raise ValueError("formal result counts must be nonnegative")
    if nonfinite_metric_cells != 0:
        raise ValueError("formal computed metrics contain non-finite cells")
    if paper_sources_in_computed_rows != 0:
        raise ValueError("paper-reference sources entered computed rows")

    resolved_specs = _validate_specs(artifacts)
    stage_by_name = {stage.name: stage for stage in stages}
    for stage in stages:
        verify_stage_receipt(stage)
    for spec, _target, source in resolved_specs:
        if spec.stage not in stage_by_name:
            raise ValueError(f"artifact references an absent stage: {spec.stage}")
        if source.as_posix() not in dict(stage_by_name[spec.stage].artifact_hashes):
            raise ValueError(f"artifact is not declared by its stage: {spec.source}")

    cache_key = canonical_sha256(
        {
            "scale": scale,
            "stages": [(stage.name, stage.content_hash) for stage in stages],
            "artifacts": [
                (spec.target, spec.stage, spec.source) for spec in artifacts
            ],
            "counts": counts,
        }
    )
    run_id = f"formal-{scale}-{cache_key[:16]}"
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root / run_id
    if final.exists():
        return _saved_run_receipt(
            final,
            stages=stages,
            run_id=run_id,
            scale=scale,
            expected_artifacts=tuple(
                target.as_posix() for _spec, target, _source in resolved_specs
            ),
            counts=counts,
        )
    partial = root / f".{run_id}.partial"
    if partial.exists():
        if partial.parent != root or not partial.name.endswith(".partial"):
            raise ValueError("refusing to remove an unsafe formal partial path")
        shutil.rmtree(partial)
    partial.mkdir(parents=False, exist_ok=False)

    artifact_hashes: list[tuple[str, str]] = []
    for spec, target_relative, source_relative in resolved_specs:
        stage = stage_by_name[spec.stage]
        source = stage.path.joinpath(*source_relative.parts)
        expected_hash = dict(stage.artifact_hashes)[source_relative.as_posix()]
        if not source.is_file() or _sha256(source) != expected_hash:
            raise ValueError(f"verified stage artifact changed: {spec.source}")
        target = partial.joinpath(*target_relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        with target.open("r+b") as stream:
            os.fsync(stream.fileno())
        artifact_hashes.append((target_relative.as_posix(), _sha256(target)))

    saved_hashes = tuple(sorted(artifact_hashes))
    payload = {
        "run_id": run_id,
        "scale": scale,
        "completed_stages": [stage.name for stage in stages],
        "stage_content_hashes": {stage.name: stage.content_hash for stage in stages},
        "artifact_hashes": dict(saved_hashes),
        "computed_response_count": computed_response_count,
        "applicable_metric_cells": applicable_metric_cells,
        "nonfinite_metric_cells": nonfinite_metric_cells,
        "paper_sources_in_computed_rows": paper_sources_in_computed_rows,
    }
    receipt_path = partial / "formal_run_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, final)
    return FormalRunReceipt(
        run_id=run_id,
        scale=scale,
        final_path=final,
        stage_receipts=stages,
        artifact_hashes=saved_hashes,
        computed_response_count=computed_response_count,
        applicable_metric_cells=applicable_metric_cells,
        nonfinite_metric_cells=nonfinite_metric_cells,
        paper_sources_in_computed_rows=paper_sources_in_computed_rows,
    )
