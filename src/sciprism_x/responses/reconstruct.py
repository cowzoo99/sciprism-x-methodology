from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections.abc import Callable
from pathlib import Path

from sciprism_x.data.schema import SciCUCase

from .schema import (
    ReconstructionAudit,
    ReconstructionProfile,
    ReconstructionResult,
    ResponseRecord,
)

SemanticUnits = tuple[str, ...]
Transform = Callable[[SemanticUnits], SemanticUnits]

_UNIT_PATTERN = re.compile(
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?P<unit>m/s|km/h|V/m|m\^2|m|s|kg|K|Pa|N|J|W|T)\b",
    flags=re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])([+-]?(?:\d+(?:\.\d+)?|\.\d+))")


def semantic_units(text: str) -> SemanticUnits:
    normalized = " ".join(text.split())
    if not normalized:
        return ("No reference response was available.",)
    units = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if part.strip()
    )
    return units or (normalized,)


def corrupt_physical_condition(units: SemanticUnits) -> SemanticUnits:
    pattern = re.compile(r"\bunder\s+[^,.;]+(?:condition|conditions)\b", re.IGNORECASE)
    changed = tuple(pattern.sub("regardless of boundary conditions", unit) for unit in units)
    if changed == units:
        changed += ("The same result is asserted regardless of the physical boundary conditions.",)
    return changed


def inject_unsupported_inference(units: SemanticUnits) -> SemanticUnits:
    return units + ("Therefore the conclusion is taken to follow without the stated premises.",)


def alter_dimensional_statement(units: SemanticUnits) -> SemanticUnits:
    changed = tuple(_UNIT_PATTERN.sub(lambda match: f"{match.group('value')} kg", unit) for unit in units)
    if changed == units:
        changed += ("The reported quantity is assigned the incompatible unit kg.",)
    return changed


def reverse_causal_relation(units: SemanticUnits) -> SemanticUnits:
    return units + ("The stated effect is treated as the cause of its preceding condition.",)


def detach_evidence_anchor(units: SemanticUnits) -> SemanticUnits:
    anchor = re.compile(
        r"\b(?:according to\s+)?(?:table|figure|reference|source)\s*[A-Za-z0-9.-]+[,]?\s*",
        re.IGNORECASE,
    )
    changed = tuple(anchor.sub("", unit).strip() for unit in units)
    changed = tuple(unit for unit in changed if unit)
    return changed + ("The values are used without an explicit evidence anchor.",)


def remove_uncertainty_qualifier(units: SemanticUnits) -> SemanticUnits:
    qualifier = re.compile(
        r"\b(?:approximately|about|may|might|could|assuming|assumed|ideal|ideally)\b\s*",
        re.IGNORECASE,
    )
    changed = tuple(qualifier.sub("", unit).replace("  ", " ").strip() for unit in units)
    return tuple(unit for unit in changed if unit) + (
        "This conclusion is asserted as exact and unconditional.",
    )


TRANSFORMS: dict[str, Transform] = {
    "physical": corrupt_physical_condition,
    "logical": inject_unsupported_inference,
    "unit": alter_dimensional_statement,
    "causal": reverse_causal_relation,
    "modal_evidence": detach_evidence_anchor,
    "uncertainty": remove_uncertainty_qualifier,
}


def _alter_endpoint(units: SemanticUnits) -> SemanticUnits:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        if changed:
            return match.group(0)
        value = float(match.group(1))
        replacement = value * 1.1 + 1.0
        changed = True
        return f"{replacement:.6g}"

    output = tuple(_NUMBER_PATTERN.sub(replace, unit) for unit in units)
    if not changed:
        output += ("The final endpoint is changed to an incompatible alternative.",)
    return output


def _omit_motif(units: SemanticUnits, rng: random.Random) -> SemanticUnits:
    if len(units) > 1:
        index = rng.randrange(len(units))
        return units[:index] + units[index + 1 :]
    clauses = tuple(part.strip() for part in re.split(r"[,;]", units[0]) if part.strip())
    if len(clauses) > 1:
        index = rng.randrange(len(clauses))
        kept = clauses[:index] + clauses[index + 1 :]
        return (", ".join(kept),)
    return ("The required reasoning step is omitted.",)


def _add_unsupported_content(units: SemanticUnits) -> SemanticUnits:
    return units + ("An additional unsupported mechanism is claimed to determine the result.",)


def _degrade_evidence_synthesis(units: SemanticUnits) -> SemanticUnits:
    return units + ("The evidence items are listed independently without synthesizing them.",)


def _apply_length_ratio(
    units: SemanticUnits,
    source_units: SemanticUnits,
    ratio: float,
) -> SemanticUnits:
    words = " ".join(units).split()
    source_words = " ".join(source_units).split()
    target = max(1, round(len(source_words) * ratio))
    if len(words) >= target:
        resized = words[:target]
    else:
        filler = source_words or words or ["response"]
        resized = list(words)
        while len(resized) < target:
            resized.extend(filler[: target - len(resized)])
    return (" ".join(resized),)


def reconstruct_response_with_audit(
    case: SciCUCase,
    profile: ReconstructionProfile,
    *,
    seed: int,
    model_id: str = "unspecified",
    replicate: int = 0,
) -> ReconstructionResult:
    rng = random.Random(seed)
    source_units = semantic_units(case.reference_answer)
    units = source_units
    applied: list[str] = []

    error_rates = (
        ("physical", profile.physical_error),
        ("logical", profile.logical_error),
        ("unit", profile.unit_error),
        ("causal", profile.causal_error),
        ("modal_evidence", profile.modal_evidence_error),
        ("uncertainty", profile.uncertainty_error),
    )
    for name, probability in error_rates:
        if rng.random() < probability:
            units = TRANSFORMS[name](units)
            applied.append(name)

    if rng.random() < profile.endpoint_error:
        units = _alter_endpoint(units)
        applied.append("endpoint")
    if rng.random() < profile.motif_omission:
        units = _omit_motif(units, rng)
        applied.append("motif_omission")
    if rng.random() < profile.unsupported_content:
        units = _add_unsupported_content(units)
        applied.append("unsupported_content")
    if rng.random() > profile.evidence_synthesis:
        units = _degrade_evidence_synthesis(units)
        applied.append("evidence_synthesis")
    if not math.isclose(profile.length_ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        units = _apply_length_ratio(units, source_units, profile.length_ratio)
        applied.append("length_ratio")

    text = " ".join(units).strip()
    if not text:
        text = "The response does not state the required result."
        units = (text,)
    record = ResponseRecord(
        task_id=case.task_id,
        model_id=model_id,
        replicate=replicate,
        response_text=text,
    )
    audit = ReconstructionAudit(
        seed=seed,
        source_units=source_units,
        output_units=units,
        applied_transforms=tuple(applied),
    )
    return ReconstructionResult(record=record, audit=audit)


def reconstruct_response(
    case: SciCUCase,
    profile: ReconstructionProfile,
    *,
    seed: int,
    model_id: str = "unspecified",
    replicate: int = 0,
) -> ResponseRecord:
    return reconstruct_response_with_audit(
        case,
        profile,
        seed=seed,
        model_id=model_id,
        replicate=replicate,
    ).record


def fixture_cases(count: int) -> tuple[SciCUCase, ...]:
    motif_families = (
        "symbolic transformation",
        "unit/dimensional consistency",
        "causal-mechanistic relation",
        "evidence binding",
        "modality grounding",
        "uncertainty qualification",
    )
    cases: list[SciCUCase] = []
    for index in range(count):
        task_id = f"FIXTURE-{index + 1:04d}"
        cases.append(
            SciCUCase(
                task_id=task_id,
                prompt="Use Table 1 to derive v = E/B and state the applicable condition.",
                reference_context="Table 1 provides compatible values for E and B.",
                reference_answer=(
                    "Under ideal conditions, the fields balance and the result is "
                    f"{index + 1} m/s according to Table 1."
                ),
                domain="physics",
                competency="fixture verification",
                motif_family=motif_families[index % len(motif_families)],
                source_hash=hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(cases)


def _write_parquet(rows: tuple[ResponseRecord, ...], target: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    schema = pa.schema(
        (
            ("task_id", pa.string()),
            ("model_id", pa.string()),
            ("replicate", pa.int64()),
            ("response_text", pa.string()),
        )
    )
    table = pa.Table.from_pylist(
        [row.model_dump(mode="json") for row in rows],
        schema=schema,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    parquet.write_table(
        table,
        partial,
        compression="zstd",
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
    )
    partial.replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _write_json(payload: dict[str, object], target: Path) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_bytes(encoded)
    partial.replace(target)


def main(argv: list[str] | None = None) -> int:
    from sciprism_x.config import load_run_config
    from sciprism_x.responses.adapters import RecoveryRequest, recover_responses

    parser = argparse.ArgumentParser(description="Build deterministic response fixtures")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture-count", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/SciPRISM-X-runs/gate1_response_fixture"),
    )
    arguments = parser.parse_args(argv)
    if arguments.fixture_count < 1:
        parser.error("--fixture-count must be positive")

    config = load_run_config(arguments.config)
    profile = ReconstructionProfile(
        **config.response_reconstruction.fixture_profile.model_dump()
    )
    recovered = recover_responses(
        RecoveryRequest(
            cases=fixture_cases(arguments.fixture_count),
            model_id=config.generators.model_ids[0],
            responses_per_task=1,
            seed=config.paper.seeds[0],
            reconstruction_profile=profile,
        )
    )
    output_dir: Path = arguments.output_dir
    responses_path = output_dir / "responses.parquet"
    responses_sha256 = _write_parquet(recovered.rows, responses_path)
    manifest = {
        "schema_version": 1,
        "fixture_only": True,
        "row_count": len(recovered.rows),
        "responses_sha256": responses_sha256,
        "acquisition": recovered.acquisition.model_dump(mode="json"),
    }
    manifest_path = output_dir / "input_manifest.json"
    _write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "row_count": len(recovered.rows),
                "responses_sha256": responses_sha256,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
