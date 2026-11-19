from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np


def build_trials(
    search_space: Mapping[str, Sequence[float]],
    count: int,
    seed: int,
) -> tuple[dict[str, float], ...]:
    if count != 50:
        raise ValueError("method contract requires exactly 50 trials")
    if not search_space or any(len(values) == 0 for values in search_space.values()):
        raise ValueError("search space dimensions must be non-empty")
    names = tuple(sorted(search_space))
    combinations = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(search_space[name] for name in names))
    ]
    rng = np.random.default_rng(seed)
    trials: list[dict[str, float]] = []
    while len(trials) < count:
        order = rng.permutation(len(combinations))
        trials.extend(combinations[index] for index in order)
    return tuple(trials[:count])
