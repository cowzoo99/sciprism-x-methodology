from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sciprism_x.tuning import build_trials

from .schema import (
    BaselineAdapter,
    BaselineKind,
    BaselineResult,
    EvaluatedCase,
    project_baseline_input,
)


@dataclass(frozen=True, slots=True)
class BaselineDefinition:
    kind: BaselineKind
    paper_family: str
    benchmarks: tuple[str, ...]
    family_only: bool
    exact_identifier: str | None
    official_revision: str | None


PAPER_BASELINES = (
    BaselineDefinition(BaselineKind.ENDPOINT, "answer-centric", ("scicueval",), True, None, None),
    BaselineDefinition(BaselineKind.ATTRIBUTION, "attribution", ("scicueval",), True, None, None),
    BaselineDefinition(BaselineKind.FIXED_TAXONOMY, "fixed taxonomy", ("scicueval",), True, None, None),
    BaselineDefinition(BaselineKind.UNCERTAINTY, "uncertainty-only", ("scicueval",), True, None, None),
    BaselineDefinition(BaselineKind.KNN_SUPPORT, "generic open-set kNN", ("scicueval",), True, None, None),
    BaselineDefinition(BaselineKind.CENTROID_SUPPORT, "generic open-set centroid", ("scicueval",), True, None, None),
    BaselineDefinition(BaselineKind.GRAPH_EDIT, "graph edit distance", ("scicueval", "dbp15k"), True, None, None),
    BaselineDefinition(BaselineKind.SINKHORN, "Sinkhorn OT", ("dbp15k",), True, None, None),
    BaselineDefinition(BaselineKind.GROMOV, "Gromov transport", ("dbp15k",), True, None, None),
    BaselineDefinition(BaselineKind.STRUCTURE_AWARE, "structure-aware matching", ("scicueval", "dbp15k"), True, None, None),
    BaselineDefinition(BaselineKind.RELATION_AWARE, "relation-aware matching", ("dbp15k",), True, None, None),
    BaselineDefinition(BaselineKind.LLM_JUDGE, "LLM judge", ("scicueval",), False, "gpt-4o-2024-08-06", None),
    BaselineDefinition(BaselineKind.HITS_AT_1, "Hits@1 diagnostic", ("dbp15k",), True, None, None),
    BaselineDefinition(BaselineKind.MRR, "MRR diagnostic", ("dbp15k",), True, None, None),
)


@dataclass(frozen=True, slots=True)
class ResolvedBaselineTrial:
    parameters: tuple[tuple[str, float], ...]
    validation_score: float
    trial_count: int
    seed: int


def freeze_strongest_validation_trial(
    *,
    search_space: Mapping[str, Sequence[float]],
    count: int,
    seed: int,
    validation_score: Callable[[dict[str, float]], float],
) -> ResolvedBaselineTrial:
    trials = build_trials(search_space, count=count, seed=seed)
    scored = tuple(float(validation_score(trial)) for trial in trials)
    best_index = max(range(len(trials)), key=lambda index: (scored[index], -index))
    return ResolvedBaselineTrial(
        parameters=tuple(sorted(trials[best_index].items())),
        validation_score=scored[best_index],
        trial_count=len(trials),
        seed=seed,
    )


class BaselineRegistry:
    def __init__(self, adapters: Mapping[BaselineKind, BaselineAdapter]) -> None:
        self._adapters = dict(adapters)
        for kind, adapter in self._adapters.items():
            if adapter.kind is not kind:
                raise ValueError(f"adapter kind mismatch: {kind} != {adapter.kind}")

    @property
    def kinds(self) -> tuple[BaselineKind, ...]:
        return tuple(sorted(self._adapters, key=lambda kind: kind.value))

    def evaluate(self, kind: BaselineKind, case: EvaluatedCase) -> BaselineResult:
        try:
            adapter = self._adapters[kind]
        except KeyError as error:
            raise KeyError(f"baseline adapter is not registered: {kind.value}") from error
        projected = project_baseline_input(case, kind)
        if not isinstance(projected, adapter.input_type):
            raise TypeError(f"projection mismatch for {kind.value}")
        return adapter.evaluate(projected)
