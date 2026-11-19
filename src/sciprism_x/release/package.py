from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
import yaml

from sciprism_x.formal.results import REQUIRED_FORMAL_ARTIFACTS


class ReleasePackageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    package_name: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    review_availability: str = Field(pattern=r"^immediate_on_request$")
    public_availability: str = Field(pattern=r"^upon_publication$")
    project_include: tuple[str, ...] = Field(min_length=1)
    run_include: tuple[str, ...] = Field(min_length=1)
    excluded_names: tuple[str, ...] = Field(min_length=1)
    forbidden_identity_tokens: tuple[str, ...] = ()
    allowed_absolute_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageAudit:
    errors: tuple[str, ...]
    file_count: int


@dataclass(frozen=True, slots=True)
class FormalReleaseAudit:
    errors: tuple[str, ...]
    checked_artifacts: int
    checked_instances: int


_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".rst",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_TEXT_NAMES = {".gitignore"}
_SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"qyapi\.weixin\.qq\.com/cgi-bin/webhook/send\?key=[A-Za-z0-9-]+"),
)
_WINDOWS_ABSOLUTE = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*"
    r"[^\\/:*?\"<>|\r\n]*"
)
_POSIX_IDENTITY_PATH = re.compile(
    r"(?:/" + r"home/[^/\s]+|/" + r"Users/[^/\s]+)(?:/[^\s]*)?"
)


def load_release_config(path: Path) -> ReleasePackageConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("release-package configuration must be a mapping")
    return ReleasePackageConfig.model_validate(payload)


def _config(value: ReleasePackageConfig | Mapping[str, object]) -> ReleasePackageConfig:
    return value if isinstance(value, ReleasePackageConfig) else ReleasePackageConfig.model_validate(value)


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"package include path is unsafe: {value}")
    return path


def _excluded(path: Path, config: ReleasePackageConfig) -> bool:
    return any(
        fnmatch.fnmatch(part, pattern)
        for part in path.parts
        for pattern in config.excluded_names
    )


def _copy_allowlisted(
    *,
    source_root: Path,
    relative: str,
    destination_root: Path,
    config: ReleasePackageConfig,
) -> None:
    safe = _safe_relative(relative)
    source = source_root.joinpath(*safe.parts)
    if not source.exists():
        raise FileNotFoundError(f"allowlisted package source is missing: {relative}")
    if source.is_symlink():
        raise ValueError(f"allowlisted package source must not be a symlink: {relative}")
    destination = destination_root.joinpath(*safe.parts)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    for path in sorted(source.rglob("*")):
        child_relative = path.relative_to(source)
        if _excluded(child_relative, config):
            continue
        if path.is_symlink():
            raise ValueError(
                f"allowlisted package tree must not contain symlinks: "
                f"{relative}/{child_relative.as_posix()}"
            )
        target = destination / child_relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowed_absolute(value: str, config: ReleasePackageConfig) -> bool:
    normalized = value.rstrip(" .,:;)'\"`").casefold()
    for prefix in config.allowed_absolute_paths:
        allowed = prefix.rstrip("\\").casefold()
        if normalized == allowed or normalized.startswith(allowed + "\\"):
            return True
    return False


def audit_package_tree(
    root: Path,
    *,
    config: ReleasePackageConfig | Mapping[str, object],
) -> PackageAudit:
    package = Path(root)
    resolved = _config(config)
    errors: list[str] = []
    files = tuple(sorted(path for path in package.rglob("*") if path.is_file()))
    for path in package.rglob("*"):
        relative = path.relative_to(package)
        if path.is_symlink():
            errors.append(f"symlink_present:{relative.as_posix()}")
        if _excluded(relative, resolved):
            errors.append(f"excluded_path_present:{relative.as_posix()}")
    for path in files:
        relative = path.relative_to(package).as_posix()
        if path.suffix.lower() not in _TEXT_SUFFIXES and path.name not in _TEXT_NAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"declared_text_file_is_not_utf8:{relative}")
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret_pattern:{relative}")
                break
        for token in resolved.forbidden_identity_tokens:
            if token and token.casefold() in text.casefold():
                errors.append(f"identity_token:{relative}")
        for match in _WINDOWS_ABSOLUTE.finditer(text):
            if not _allowed_absolute(match.group(0), resolved):
                errors.append(f"local_windows_path:{relative}")
                break
        if _POSIX_IDENTITY_PATH.search(text):
            errors.append(f"local_posix_identity_path:{relative}")
    manifest_path = package / "package_manifest.json"
    availability_path = package / "availability.json"
    if not manifest_path.is_file():
        errors.append("missing_package_manifest")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = list(manifest.get("files", []))
            declared = {
                str(record["path"]): (
                    int(record["bytes"]),
                    str(record["sha256"]),
                )
                for record in records
            }
            actual = {
                path.relative_to(package).as_posix(): (path.stat().st_size, _sha256(path))
                for path in files
                if path != manifest_path
            }
            if len(declared) != len(records):
                errors.append("duplicate_package_manifest_path")
            if declared != actual:
                errors.append("package_manifest_file_set_or_hash_mismatch")
            if int(manifest.get("file_count_excluding_manifest", -1)) != len(actual):
                errors.append("package_manifest_file_count_mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("invalid_package_manifest")
    if not availability_path.is_file():
        errors.append("missing_availability_manifest")
    else:
        try:
            availability = json.loads(availability_path.read_text(encoding="utf-8"))
            if (
                availability.get("package_name") != resolved.package_name
                or availability.get("package_version") != resolved.package_version
                or availability.get("review_availability")
                != resolved.review_availability
                or availability.get("public_availability")
                != resolved.public_availability
            ):
                errors.append("availability_manifest_mismatch")
        except json.JSONDecodeError:
            errors.append("invalid_availability_manifest")
    formal_receipt_path = package / "results" / "formal_run_receipt.json"
    contract_path = package / "results" / "release" / "package_contract.json"
    if formal_receipt_path.is_file() and not contract_path.is_file():
        errors.append("missing_release_package_contract")
    elif contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            requirements = contract.get("requirements")
            if not isinstance(requirements, dict) or set(requirements) != {
                "D1",
                "D2",
                "D3",
                "D4",
                "D5",
                "D6",
                "D7",
                "D8",
            }:
                errors.append("release_package_contract_keys_incomplete")
            else:
                for requirement, paths in requirements.items():
                    if not isinstance(paths, list) or not paths:
                        errors.append(f"empty_release_requirement:{requirement}")
                        continue
                    for value in paths:
                        relative = _safe_relative(str(value))
                        if not package.joinpath(*relative.parts).exists():
                            errors.append(
                                f"release_requirement_path_missing:"
                                f"{requirement}:{relative.as_posix()}"
                            )
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append("invalid_release_package_contract")
    return PackageAudit(tuple(sorted(set(errors))), len(files))


def build_release_package(
    *,
    project_root: Path,
    run_root: Path,
    target: Path,
    config: ReleasePackageConfig | Mapping[str, object],
) -> Path:
    resolved = _config(config)
    project = Path(project_root).resolve()
    run = Path(run_root).resolve()
    destination = Path(target).resolve()
    if destination.exists():
        raise FileExistsError(f"release package target already exists: {destination}")
    if destination == project or project in destination.parents:
        raise ValueError("release package target must be outside the project tree")
    if destination == run or run in destination.parents:
        raise ValueError("release package target must be outside the run tree")
    partial = destination.parent / f".{destination.name}.partial"
    if partial.exists():
        if partial.parent != destination.parent or partial.name != f".{destination.name}.partial":
            raise ValueError("refusing to remove an unsafe partial package path")
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=False)
    try:
        for relative in resolved.project_include:
            _copy_allowlisted(
                source_root=project,
                relative=relative,
                destination_root=partial / "project",
                config=resolved,
            )
        for relative in resolved.run_include:
            _copy_allowlisted(
                source_root=run,
                relative=relative,
                destination_root=partial / "results",
                config=resolved,
            )
        availability = {
            "schema_version": 1,
            "package_name": resolved.package_name,
            "package_version": resolved.package_version,
            "review_availability": resolved.review_availability,
            "public_availability": resolved.public_availability,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (partial / "availability.json").write_text(
            json.dumps(availability, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        file_records = [
            {
                "path": path.relative_to(partial).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(partial.rglob("*"))
            if path.is_file() and path.name != "package_manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "package_name": resolved.package_name,
            "package_version": resolved.package_version,
            "file_count_excluding_manifest": len(file_records),
            "files": file_records,
            "benchmark_instance_identifiers_preserved": True,
            "personal_and_machine_identity_fields_included": False,
            "identity_scan": {
                "allowlist_copy": True,
                "configured_forbidden_token_count": len(
                    resolved.forbidden_identity_tokens
                ),
                "secret_pattern_scan": True,
                "local_absolute_path_scan": True,
                "symlink_rejection": True,
            },
        }
        (partial / "package_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        audit = audit_package_tree(partial, config=resolved)
        if audit.errors:
            raise ValueError(f"release package audit failed: {audit.errors}")
        os.replace(partial, destination)
    except BaseException:
        if partial.exists() and partial.parent == destination.parent:
            shutil.rmtree(partial)
        raise
    return destination


def audit_formal_release_run(
    run_root: Path,
    *,
    scale: str,
) -> FormalReleaseAudit:
    root = Path(run_root)
    errors: list[str] = []
    receipt_path = root / "formal_run_receipt.json"
    if not receipt_path.is_file():
        return FormalReleaseAudit(("missing_formal_run_receipt",), 0, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    hashes = dict(receipt.get("artifact_hashes", {}))
    for relative in REQUIRED_FORMAL_ARTIFACTS:
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            errors.append(f"missing_artifact:{relative}")
        elif hashes.get(relative) != _sha256(path):
            errors.append(f"artifact_hash_mismatch:{relative}")

    computed = root / "computed"
    sample_path = computed / "sample_manifest.parquet"
    link_path = computed / "per_instance_outputs.parquet"
    checked_instances = 0
    if sample_path.is_file():
        samples = pd.read_parquet(sample_path)
        required_sample_columns = {
            "dataset",
            "sample_id",
            "original_sample_id",
            "parent_sample_id",
            "allocation_id",
            "split",
            "regime",
            "valid",
        }
        if not required_sample_columns.issubset(samples.columns):
            errors.append("sample_manifest_schema_incomplete")
            samples = pd.DataFrame(columns=sorted(required_sample_columns))
        if samples.duplicated(["dataset", "sample_id", "split"]).any():
            errors.append("duplicate_sample_manifest_key")
        final = samples[samples["split"] == "formal_final_test"]
        if not final.empty and not final["valid"].astype(bool).all():
            errors.append("invalid_case_present_in_formal_final_test")
        if scale == "paper":
            for dataset in ("scicueval", "dbp15k"):
                group = final[final["dataset"] == dataset]
                for regime in ("iid", "motif_or_structure", "domain_transfer"):
                    count = int((group["regime"] == regime).sum())
                    if count != 600:
                        errors.append(f"sample_count:{dataset}:{regime}:{count}:expected_600")
                boundary_count = int((group["regime"] == "boundary_stress").sum())
                if boundary_count != 2400:
                    errors.append(
                        f"sample_count:{dataset}:boundary_stress:{boundary_count}:expected_2400"
                    )
            boundary = final[final["regime"] == "boundary_stress"]
            if (boundary["parent_sample_id"] == "").any():
                errors.append("boundary_parent_link_missing")
    component_paths = {
        "task_graphs": computed / "task_graphs.jsonl",
        "response_graphs": computed / "response_graphs.jsonl",
        "validators": computed / "validator_outputs.parquet",
        "transport": computed / "transport_diagnostics.parquet",
        "residual": computed / "residual_components.parquet",
        "plans": computed / "transport_plans.npz",
        "states": computed / "state_assignments.parquet",
    }
    if link_path.is_file() and all(path.is_file() for path in component_paths.values()):
        links = pd.read_parquet(link_path)
        checked_instances = len(links)
        required_link_columns = {
            "instance_key",
            "task_id",
            "model_id",
            "replicate",
            "task_graph_id",
            "response_graph_id",
            "plan_key",
            "validator_pair_count",
            "residual_channel_count",
            "baseline_artifact",
            "ablation_artifact",
        }
        if not required_link_columns.issubset(links.columns):
            errors.append("per_instance_output_schema_incomplete")
            return FormalReleaseAudit(
                tuple(sorted(set(errors))),
                len(REQUIRED_FORMAL_ARTIFACTS),
                checked_instances,
            )
        if links["instance_key"].duplicated().any():
            errors.append("duplicate_per_instance_key")
        if links[["task_id", "model_id", "replicate"]].duplicated().any():
            errors.append("duplicate_per_instance_response_key")
        task_rows = [
            json.loads(line)
            for line in component_paths["task_graphs"].read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        graph_rows = [
            json.loads(line)
            for line in component_paths["response_graphs"].read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        task_graph_keys = {row["graph_id"] for row in task_rows}
        if task_graph_keys != set(links["task_graph_id"]):
            errors.append("task_graph_ids_do_not_match")
        graph_keys = {row["instance_key"] for row in graph_rows}
        link_keys = set(links["instance_key"])
        if graph_keys != link_keys:
            errors.append("response_graph_instance_keys_do_not_match")
        validators = pd.read_parquet(component_paths["validators"])
        validator_counts = validators.groupby("instance_key").size().to_dict()
        if any(
            int(row.validator_pair_count)
            != int(validator_counts.get(str(row.instance_key), 0))
            for row in links.itertuples(index=False)
        ):
            errors.append("validator_pair_counts_do_not_match")
        if not set(validators["instance_key"]).issubset(link_keys):
            errors.append("validator_instance_key_is_unknown")
        transport = pd.read_parquet(component_paths["transport"])
        residual = pd.read_parquet(component_paths["residual"])
        if set(transport["instance_key"]) != link_keys:
            errors.append("transport_instance_keys_do_not_match")
        if transport["instance_key"].duplicated().any():
            errors.append("duplicate_transport_diagnostic_key")
        residual_counts = residual.groupby("instance_key")["channel"].nunique()
        if set(residual_counts.index) != link_keys or not (residual_counts == 6).all():
            errors.append("residual_channels_do_not_cover_every_instance")
        if not (links["residual_channel_count"].astype(int) == 6).all():
            errors.append("per_instance_residual_channel_count_is_not_six")
        residual_numeric = [
            column
            for column in ("matched", "missing", "unsupported", "raw", "normalizer", "normalized")
            if column in residual.columns
        ]
        if len(residual_numeric) != 6 or not np.isfinite(
            residual[residual_numeric].to_numpy(dtype=np.float64)
        ).all():
            errors.append("residual_components_are_incomplete_or_nonfinite")
        with np.load(component_paths["plans"]) as plans:
            if set(plans.files) != set(links["plan_key"]):
                errors.append("transport_plan_keys_do_not_match")
        states = pd.read_parquet(component_paths["states"])
        link_response_keys = set(
            zip(links["task_id"], links["model_id"], links["replicate"], strict=True)
        )
        state_response_keys = set(
            zip(states["task_id"], states["model_id"], states["replicate"], strict=True)
        )
        if state_response_keys != link_response_keys:
            errors.append("state_assignment_keys_do_not_match")

        expected_artifact_columns = {
            "baseline_artifact": "computed/baselines.parquet",
            "ablation_artifact": "computed/ablations.parquet",
            "validator_artifact": "computed/validator_outputs.parquet",
            "transport_artifact": "computed/transport_plans.npz",
            "residual_artifact": "computed/residual_components.parquet",
        }
        for column, expected in expected_artifact_columns.items():
            if column not in links.columns or set(links[column]) != {expected}:
                errors.append(f"per_instance_artifact_pointer:{column}")

    effective_path = computed / "effective_hyperparameters.json"
    if effective_path.is_file():
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
        required_method_keys = {
            "lambda_str",
            "lambda_con",
            "epsilon",
            "eta_1_node",
            "eta_2_structure",
            "eta_3_constraint",
            "gamma",
            "omega_miss",
            "omega_unsup",
        }
        if not required_method_keys.issubset(dict(effective.get("method", {}))):
            errors.append("effective_method_parameters_incomplete")
        if not effective.get("validator_thresholds") or not effective.get(
            "regime_thresholds"
        ):
            errors.append("effective_thresholds_incomplete")
    learned_path = computed / "learned_modules.json"
    if learned_path.is_file():
        learned = json.loads(learned_path.read_text(encoding="utf-8"))
        required_modules = {
            "g_theta",
            "e_phi",
            "implicit_motif_recognizer",
            "entailment_validator",
            "state_field_training",
        }
        if not required_modules.issubset(learned):
            errors.append("learned_module_manifest_incomplete")
    return FormalReleaseAudit(
        tuple(sorted(set(errors))),
        len(REQUIRED_FORMAL_ARTIFACTS),
        checked_instances,
    )
