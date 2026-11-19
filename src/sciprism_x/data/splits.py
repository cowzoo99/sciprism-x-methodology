from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np

from sciprism_x.reproducibility import derive_seed

from .schema import DBPCase, RegimeSplits, SciCUCase


T = TypeVar("T")


def _allocate(group_sizes: dict[str, int], count: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if count > total:
        raise ValueError(f"cannot select {count} items from {total}")
    raw = {key: count * size / total for key, size in group_sizes.items()}
    allocation = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = count - sum(allocation.values())
    ranked = sorted(group_sizes, key=lambda key: (-(raw[key] - allocation[key]), key))
    for key in ranked[:remainder]:
        allocation[key] += 1
    return allocation


def _sample_grouped(
    items: Sequence[T],
    count: int,
    rng: np.random.Generator,
    primary: Callable[[T], str],
    secondary: Callable[[T], str],
) -> tuple[T, ...]:
    primary_groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        primary_groups[primary(item)].append(item)
    primary_quota = _allocate(
        {key: len(values) for key, values in primary_groups.items()},
        count,
    )

    selected: list[T] = []
    for primary_key in sorted(primary_groups):
        secondary_groups: dict[str, list[T]] = defaultdict(list)
        for item in primary_groups[primary_key]:
            secondary_groups[secondary(item)].append(item)
        secondary_quota = _allocate(
            {key: len(values) for key, values in secondary_groups.items()},
            primary_quota[primary_key],
        )
        for secondary_key in sorted(secondary_groups):
            group = secondary_groups[secondary_key]
            order = rng.permutation(len(group))
            selected.extend(group[index] for index in order[: secondary_quota[secondary_key]])
    return tuple(selected)


def _select_regime(
    pool: Sequence[SciCUCase],
    count: int,
    seed: int,
    namespace: str,
) -> tuple[SciCUCase, ...]:
    rng = np.random.default_rng(derive_seed(seed, namespace))
    return _sample_grouped(
        pool,
        count,
        rng,
        primary=lambda case: case.domain,
        secondary=lambda case: case.competency,
    )


def build_regime_splits(
    cases: Iterable[SciCUCase],
    seed: int,
    nonstress_count: int,
    stress_count: int,
) -> RegimeSplits:
    ordered = tuple(sorted(cases, key=lambda case: case.task_id))
    ids = tuple(case.task_id for case in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs must be unique")
    required_nonstress = nonstress_count * 3
    if len(ordered) < required_nonstress:
        raise ValueError(
            f"non-stress regimes require {required_nonstress} unique cases, got {len(ordered)}"
        )
    if len(ordered) < stress_count:
        raise ValueError(
            f"boundary stress requires {stress_count} base cases, got {len(ordered)}"
        )

    remaining = list(ordered)
    iid = _select_regime(remaining, nonstress_count, seed, "iid")
    iid_ids = {case.task_id for case in iid}
    remaining = [case for case in remaining if case.task_id not in iid_ids]

    motif = _select_regime(remaining, nonstress_count, seed, "motif")
    motif_ids = {case.task_id for case in motif}
    remaining = [case for case in remaining if case.task_id not in motif_ids]

    domain = _select_regime(remaining, nonstress_count, seed, "domain")
    boundary = _select_regime(ordered, stress_count, seed, "boundary")

    return RegimeSplits(
        seed=seed,
        iid=tuple(sorted(case.task_id for case in iid)),
        motif=tuple(sorted(case.task_id for case in motif)),
        domain=tuple(sorted(case.task_id for case in domain)),
        boundary=tuple(sorted(case.task_id for case in boundary)),
    )


def _select_dbp_regime(
    pool: Sequence[DBPCase],
    count: int,
    seed: int,
    namespace: str,
) -> tuple[DBPCase, ...]:
    groups: dict[str, list[DBPCase]] = defaultdict(list)
    for case in pool:
        groups[case.language_pair].append(case)
    quota = _allocate({key: len(values) for key, values in groups.items()}, count)
    rng = np.random.default_rng(derive_seed(seed, namespace))
    selected: list[DBPCase] = []
    for language_pair in sorted(groups):
        group = groups[language_pair]
        order = rng.permutation(len(group))
        selected.extend(group[index] for index in order[: quota[language_pair]])
    return tuple(selected)


def build_dbp_regime_splits(
    cases: Iterable[DBPCase],
    seed: int,
    nonstress_count: int,
    stress_count: int,
) -> RegimeSplits:
    ordered = tuple(sorted(cases, key=lambda case: case.pair_id))
    ids = tuple(case.pair_id for case in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("pair IDs must be unique")
    required_nonstress = nonstress_count * 3
    if len(ordered) < required_nonstress:
        raise ValueError(
            f"non-stress regimes require {required_nonstress} unique pairs, got {len(ordered)}"
        )
    if len(ordered) < stress_count:
        raise ValueError(
            f"boundary stress requires {stress_count} base pairs, got {len(ordered)}"
        )

    remaining = list(ordered)
    iid = _select_dbp_regime(remaining, nonstress_count, seed, "dbp-iid")
    iid_ids = {case.pair_id for case in iid}
    remaining = [case for case in remaining if case.pair_id not in iid_ids]

    motif = _select_dbp_regime(remaining, nonstress_count, seed, "dbp-structure")
    motif_ids = {case.pair_id for case in motif}
    remaining = [case for case in remaining if case.pair_id not in motif_ids]

    domain = _select_dbp_regime(remaining, nonstress_count, seed, "dbp-domain")
    boundary = _select_dbp_regime(ordered, stress_count, seed, "dbp-boundary")

    return RegimeSplits(
        seed=seed,
        iid=tuple(sorted(case.pair_id for case in iid)),
        motif=tuple(sorted(case.pair_id for case in motif)),
        domain=tuple(sorted(case.pair_id for case in domain)),
        boundary=tuple(sorted(case.pair_id for case in boundary)),
    )


def write_split_manifest(splits: RegimeSplits, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    payload = json.dumps(splits.model_dump(mode="json"), indent=2) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
