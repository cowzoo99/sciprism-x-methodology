from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from sciprism_x.config import PaperConfig, RunConfig
from sciprism_x.statistics.bootstrap import paired_bootstrap
from sciprism_x.statistics.multiple_testing import holm_bonferroni


Direction = Literal["higher", "lower"]


@dataclass(frozen=True, slots=True)
class ComputedPairRow:
    pair_id: str
    dataset: str
    regime: str
    metric: str
    proposed_method: str
    comparison_method: str
    proposed_value: float
    comparison_value: float
    direction: Direction
    family: str
    reference_path: tuple[str, ...] = ()
    source: Literal["computed"] = "computed"


@dataclass(frozen=True, slots=True)
class ComputedComparisonRow:
    comparison_id: str
    family: str
    dataset: str
    regime: str
    metric: str
    proposed_method: str
    comparison_method: str
    observed_difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    resamples: int
    confidence_level: float
    seed: int
    direction: Direction
    pair_ids: tuple[str, ...]
    reference_path: tuple[str, ...]
    source: Literal["computed"] = "computed"


@dataclass(frozen=True, slots=True)
class SignificanceRow:
    comparison_id: str
    family: str
    raw_p_value: float
    adjusted_p_value: float
    reject: bool
    family_members: tuple[str, ...]
    source: Literal["computed"] = "computed"


@dataclass(frozen=True, slots=True)
class FormalStatisticsResult:
    bootstrap_rows: tuple[ComputedComparisonRow, ...]
    significance_rows: tuple[SignificanceRow, ...]


@dataclass(frozen=True, slots=True)
class DiscrepancyRow:
    comparison_id: str
    dataset: str
    regime: str
    metric: str
    comparison_method: str
    reference_path: tuple[str, ...]
    computed_value: float
    published_value: float
    absolute_difference: float
    relative_difference: float
    source: Literal["paper_reference_comparison"] = "paper_reference_comparison"


def _paper_config(config: RunConfig | PaperConfig) -> PaperConfig:
    return config.paper if isinstance(config, RunConfig) else config


def _validate_pair(row: ComputedPairRow) -> None:
    text_fields = (
        row.pair_id,
        row.dataset,
        row.regime,
        row.metric,
        row.proposed_method,
        row.comparison_method,
        row.family,
    )
    if any(not value for value in text_fields):
        raise ValueError("computed pair identifiers must be non-empty")
    if row.source != "computed":
        raise ValueError("paper reference rows cannot enter computed statistics")
    if not math.isfinite(row.proposed_value) or not math.isfinite(row.comparison_value):
        raise ValueError("computed paired values must be finite")
    if row.direction not in ("higher", "lower"):
        raise ValueError("computed pair direction must be higher or lower")
    if any(not part for part in row.reference_path):
        raise ValueError("reference path components must be non-empty")


def _comparison_id(key: tuple[object, ...]) -> str:
    family, dataset, regime, metric, proposed, comparison, _direction, _reference = key
    return (
        f"{family}::{dataset}::{regime}::{metric}::"
        f"{proposed}::vs::{comparison}"
    )


def compute_formal_statistics(
    pairs: Sequence[ComputedPairRow],
    config: RunConfig | PaperConfig,
) -> FormalStatisticsResult:
    if not pairs:
        raise ValueError("formal statistics require computed paired rows")
    paper = _paper_config(config)
    if not paper.two_sided_tests:
        raise ValueError("method contract requires two-sided tests")
    if paper.correction != "holm_bonferroni":
        raise ValueError("method contract requires Holm-Bonferroni correction")
    if not paper.seeds:
        raise ValueError("formal statistics require a fixed resampling seed")

    grouped: dict[tuple[object, ...], list[ComputedPairRow]] = defaultdict(list)
    for row in pairs:
        _validate_pair(row)
        key = (
            row.family,
            row.dataset,
            row.regime,
            row.metric,
            row.proposed_method,
            row.comparison_method,
            row.direction,
            row.reference_path,
        )
        grouped[key].append(row)

    comparisons: list[ComputedComparisonRow] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        rows = grouped[key]
        result = paired_bootstrap(
            pair_ids=tuple(row.pair_id for row in rows),
            proposed=np.asarray([row.proposed_value for row in rows], dtype=np.float64),
            baseline=np.asarray([row.comparison_value for row in rows], dtype=np.float64),
            resamples=paper.bootstrap_resamples,
            confidence_level=paper.confidence_level,
            seed=paper.seeds[0],
            direction=rows[0].direction,
        )
        comparisons.append(
            ComputedComparisonRow(
                comparison_id=_comparison_id(key),
                family=rows[0].family,
                dataset=rows[0].dataset,
                regime=rows[0].regime,
                metric=rows[0].metric,
                proposed_method=rows[0].proposed_method,
                comparison_method=rows[0].comparison_method,
                observed_difference=result.observed_difference,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                p_value=result.p_value,
                resamples=result.resamples,
                confidence_level=result.confidence_level,
                seed=result.seed,
                direction=result.direction,
                pair_ids=result.pair_ids,
                reference_path=rows[0].reference_path,
            )
        )
    comparison_ids = [row.comparison_id for row in comparisons]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("formal comparison IDs must be unique")

    by_family: dict[str, list[ComputedComparisonRow]] = defaultdict(list)
    for row in comparisons:
        by_family[row.family].append(row)
    significance: list[SignificanceRow] = []
    for family in sorted(by_family):
        members = tuple(sorted(row.comparison_id for row in by_family[family]))
        adjusted = holm_bonferroni(
            comparisons=tuple(
                (row.comparison_id, row.p_value) for row in by_family[family]
            ),
            alpha=1.0 - paper.confidence_level,
        )
        significance.extend(
            SignificanceRow(
                comparison_id=row.comparison_id,
                family=family,
                raw_p_value=row.raw_p,
                adjusted_p_value=row.adjusted_p,
                reject=row.reject,
                family_members=members,
            )
            for row in adjusted
        )
    return FormalStatisticsResult(tuple(comparisons), tuple(significance))


def write_computed_rows(directory: Path, rows: Sequence[object]) -> Path:
    target_directory = Path(directory)
    if target_directory.name != "computed":
        raise ValueError("computed rows must be written to the computed namespace")
    if not rows:
        raise ValueError("computed output cannot be empty")
    if any(getattr(row, "source", None) != "computed" for row in rows):
        raise ValueError("paper reference rows cannot enter the computed namespace")
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / "formal_statistics.jsonl"
    payload = "".join(
        json.dumps(asdict(row), sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
        for row in rows
    )
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return target


def _published_value(payload: Mapping[str, object], path: tuple[str, ...]) -> float:
    value: object = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"paper reference path is missing: {'/'.join(path)}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"paper reference is not numeric: {'/'.join(path)}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"paper reference is not finite: {'/'.join(path)}")
    return result


def build_discrepancy_report(
    statistics: FormalStatisticsResult,
    published_targets_path: Path,
) -> tuple[DiscrepancyRow, ...]:
    if not statistics.bootstrap_rows or not statistics.significance_rows:
        raise ValueError("computed statistics must be finalized before discrepancy reporting")
    if any(row.source != "computed" for row in statistics.bootstrap_rows):
        raise ValueError("computed statistics are contaminated by paper reference rows")
    payload = json.loads(Path(published_targets_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("published targets must be a JSON object")

    output: list[DiscrepancyRow] = []
    for row in statistics.bootstrap_rows:
        if not row.reference_path:
            continue
        published = _published_value(payload, row.reference_path)
        absolute = row.observed_difference - published
        relative = absolute / max(abs(published), 1.0e-12)
        output.append(
            DiscrepancyRow(
                comparison_id=row.comparison_id,
                dataset=row.dataset,
                regime=row.regime,
                metric=row.metric,
                comparison_method=row.comparison_method,
                reference_path=row.reference_path,
                computed_value=row.observed_difference,
                published_value=published,
                absolute_difference=absolute,
                relative_difference=relative,
            )
        )
    if not output:
        raise ValueError("no computed comparisons declare paper reference paths")
    return tuple(output)
