from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from sciprism_x.metrics.structural import compute_tp
from sciprism_x.config import JudgeConfig, RunConfig
from sciprism_x.baselines.graph import (
    GraphEditAdapter,
    GromovTransportAdapter,
    HitsAt1Adapter,
    MRRAdapter,
    RelationAwareAdapter,
    StructureAwareAdapter,
    UnconstrainedSinkhornAdapter,
)
from sciprism_x.baselines.llm_judge import (
    LLMJudgeAdapter,
    LocalInferredJudgeAdapter,
    Requester,
)
from sciprism_x.baselines.registry import (
    PAPER_BASELINES,
    BaselineRegistry,
    ResolvedBaselineTrial,
    freeze_strongest_validation_trial,
)
from sciprism_x.baselines.schema import (
    BaselineAdapter,
    BaselineKind,
    EvaluatedCase,
)
from sciprism_x.baselines.scientific import (
    AttributionAdapter,
    EndpointAdapter,
    FixedTaxonomyAdapter,
    SupportDistanceAdapter,
    UncertaintyAdapter,
)


@dataclass(frozen=True, slots=True)
class FormalBaselineRow:
    case_id: str
    dataset: str
    baseline: str
    metric: str
    score: float
    status: str
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FormalBaselineSelection:
    parameters: Mapping[str, Mapping[str, float]]
    trials: Mapping[str, ResolvedBaselineTrial]
    objectives: Mapping[str, str]


def _parameter_block(
    selected: Mapping[str, Mapping[str, float]],
    kind: BaselineKind,
) -> Mapping[str, float]:
    return selected.get(kind.value, {})


def build_formal_baseline_registry(
    *,
    api_key: str | None,
    requester: Requester | None,
    pass_threshold: float,
    fail_threshold: float,
    judge_config: JudgeConfig,
    run_config: RunConfig,
    selected_parameters: Mapping[str, Mapping[str, float]] | None = None,
) -> BaselineRegistry:
    selected = selected_parameters or {}
    resolved_judge = judge_config
    if resolved_judge.dataset != "scicueval":
        raise ValueError("the paper applies the LLM judge only to SciCUEval")
    judge: BaselineAdapter
    if api_key and requester is not None:
        judge = LLMJudgeAdapter(
            model=resolved_judge.model_id,
            temperature=resolved_judge.temperature,
            top_p=resolved_judge.top_p,
            max_output_tokens=resolved_judge.max_output_tokens,
            api_key=api_key,
            requester=requester,
        )
    else:
        judge = LocalInferredJudgeAdapter(
            pass_threshold=pass_threshold,
            fail_threshold=fail_threshold,
            component_weights=(
                run_config.baseline_reconstruction.local_judge_component_weights
            ),
        )

    reconstruction = run_config.baseline_reconstruction
    default_epsilon = float(run_config.transport.epsilon_grid[2])
    default_structure = float(run_config.transport.lambda_structure_grid[2])
    default_graph_cost = float(reconstruction.graph_edit_cost_grid[1])
    default_graph_timeout = float(reconstruction.graph_edit_timeout_seconds)
    default_neighbors = int(reconstruction.support_neighbors_grid[0])
    default_distance_scale = float(reconstruction.support_distance_scale_grid[2])
    uncertainty_code = int(
        round(_parameter_block(selected, BaselineKind.UNCERTAINTY).get("mode_code", 2.0))
    )
    uncertainty_modes = {0: "confidence", 1: "one_minus_entropy", 2: "product"}
    if uncertainty_code not in uncertainty_modes:
        raise ValueError("selected uncertainty mode is invalid")

    transport = {
        "max_iterations": run_config.transport.max_iterations,
        "marginal_tolerance": run_config.transport.marginal_tolerance,
        "objective_tolerance": run_config.transport.objective_tolerance,
        "stable_iterations": run_config.transport.stable_iterations,
        "score_transform": reconstruction.transport_score_transform,
    }
    sinkhorn_transport = {
        **transport,
        "epsilon": _parameter_block(selected, BaselineKind.SINKHORN).get(
            "epsilon", default_epsilon
        ),
    }
    gromov_transport = {
        **transport,
        "epsilon": _parameter_block(selected, BaselineKind.GROMOV).get(
            "epsilon", default_epsilon
        ),
        "lambda_structure": _parameter_block(selected, BaselineKind.GROMOV).get(
            "lambda_structure", default_structure
        ),
    }
    structure_transport = {
        **transport,
        "epsilon": _parameter_block(selected, BaselineKind.STRUCTURE_AWARE).get(
            "epsilon", default_epsilon
        ),
        "lambda_structure": _parameter_block(
            selected, BaselineKind.STRUCTURE_AWARE
        ).get("lambda_structure", default_structure),
    }
    graph_edit = _parameter_block(selected, BaselineKind.GRAPH_EDIT)
    knn = _parameter_block(selected, BaselineKind.KNN_SUPPORT)
    centroid = _parameter_block(selected, BaselineKind.CENTROID_SUPPORT)
    adapters: dict[BaselineKind, BaselineAdapter] = {
        BaselineKind.ENDPOINT: EndpointAdapter(
            case_sensitive=reconstruction.endpoint_case_sensitive,
            strip_whitespace=reconstruction.endpoint_strip_whitespace,
            collapse_whitespace=reconstruction.endpoint_collapse_whitespace,
        ),
        BaselineKind.ATTRIBUTION: AttributionAdapter(
            lowercase=reconstruction.attribution_lowercase,
            token_pattern=reconstruction.attribution_token_pattern,
            empty_overlap_score=reconstruction.attribution_empty_overlap_score,
        ),
        BaselineKind.FIXED_TAXONOMY: FixedTaxonomyAdapter(
            category_patterns=reconstruction.taxonomy_patterns,
            no_failure_score=reconstruction.taxonomy_no_failure_score,
            failure_score=reconstruction.taxonomy_failure_score,
            lowercase=reconstruction.taxonomy_lowercase,
        ),
        BaselineKind.UNCERTAINTY: UncertaintyAdapter(
            mode=uncertainty_modes[uncertainty_code],
            maximum_entropy=reconstruction.uncertainty_maximum_entropy,
        ),
        BaselineKind.KNN_SUPPORT: SupportDistanceAdapter(
            mode="knn",
            neighbors=int(round(knn.get("neighbors", float(default_neighbors)))),
            distance_scale=knn.get("distance_scale", default_distance_scale),
            clip_to_unit_interval=reconstruction.support_clip_to_unit_interval,
        ),
        BaselineKind.CENTROID_SUPPORT: SupportDistanceAdapter(
            mode="centroid",
            neighbors=default_neighbors,
            distance_scale=centroid.get("distance_scale", default_distance_scale),
            clip_to_unit_interval=reconstruction.support_clip_to_unit_interval,
        ),
        BaselineKind.GRAPH_EDIT: GraphEditAdapter(
            node_insertion_deletion_cost=graph_edit.get(
                "node_cost", default_graph_cost
            ),
            relation_insertion_deletion_cost=graph_edit.get(
                "relation_cost", default_graph_cost
            ),
            relation_substitution_cost=graph_edit.get(
                "substitution_cost", default_graph_cost
            ),
            timeout_seconds=default_graph_timeout,
        ),
        BaselineKind.SINKHORN: UnconstrainedSinkhornAdapter(**sinkhorn_transport),
        BaselineKind.GROMOV: GromovTransportAdapter(**gromov_transport),
        BaselineKind.STRUCTURE_AWARE: StructureAwareAdapter(**structure_transport),
        BaselineKind.RELATION_AWARE: RelationAwareAdapter(
            multiset=reconstruction.relation_multiset
        ),
        BaselineKind.LLM_JUDGE: judge,
        BaselineKind.HITS_AT_1: HitsAt1Adapter(
            descending=reconstruction.ranking_descending
        ),
        BaselineKind.MRR: MRRAdapter(descending=reconstruction.ranking_descending),
    }
    if set(adapters) != set(BaselineKind):
        raise AssertionError("formal baseline registry is incomplete")
    return BaselineRegistry(adapters)


def select_formal_baseline_parameters(
    *,
    scicu_validation_cases: Sequence[EvaluatedCase],
    scicu_structural_validity: Mapping[str, bool],
    dbp_validation_cases: Sequence[EvaluatedCase],
    config: RunConfig,
    seed: int,
    pass_threshold: float,
    fail_threshold: float,
) -> FormalBaselineSelection:
    """Run Appendix D.3 validation-only selection for configured grids."""

    scicu = tuple(scicu_validation_cases)
    dbp = tuple(dbp_validation_cases)
    if not scicu or not dbp:
        raise ValueError("baseline selection requires independent SciCU and DBP validation cases")
    if set(case.case_id for case in scicu) != set(scicu_structural_validity):
        raise ValueError("SciCUEval baseline validation labels are incomplete")

    spaces: dict[BaselineKind, Mapping[str, Sequence[float]]] = {
        BaselineKind.UNCERTAINTY: {
            "mode_code": tuple(
                float(value)
                for value in config.baseline_reconstruction.uncertainty_mode_codes
            )
        },
        BaselineKind.KNN_SUPPORT: {
            "neighbors": tuple(
                float(value)
                for value in config.baseline_reconstruction.support_neighbors_grid
            ),
            "distance_scale": config.baseline_reconstruction.support_distance_scale_grid,
        },
        BaselineKind.CENTROID_SUPPORT: {
            "distance_scale": config.baseline_reconstruction.support_distance_scale_grid,
        },
        BaselineKind.GRAPH_EDIT: {
            "node_cost": config.baseline_reconstruction.graph_edit_cost_grid,
            "relation_cost": config.baseline_reconstruction.graph_edit_cost_grid,
            "substitution_cost": config.baseline_reconstruction.graph_edit_cost_grid,
        },
        BaselineKind.STRUCTURE_AWARE: {
            "epsilon": config.transport.epsilon_grid,
            "lambda_structure": config.transport.lambda_structure_grid,
        },
        BaselineKind.SINKHORN: {"epsilon": config.transport.epsilon_grid},
        BaselineKind.GROMOV: {
            "epsilon": config.transport.epsilon_grid,
            "lambda_structure": config.transport.lambda_structure_grid,
        },
    }
    scicu_kinds = {
        BaselineKind.UNCERTAINTY,
        BaselineKind.KNN_SUPPORT,
        BaselineKind.CENTROID_SUPPORT,
        BaselineKind.GRAPH_EDIT,
        BaselineKind.STRUCTURE_AWARE,
    }
    selected: dict[str, Mapping[str, float]] = {}
    audits: dict[str, ResolvedBaselineTrial] = {}
    objectives: dict[str, str] = {}

    for kind, space in spaces.items():
        def validation_score(
            trial: dict[str, float],
            *,
            baseline_kind: BaselineKind = kind,
        ) -> float:
            registry = build_formal_baseline_registry(
                api_key=None,
                requester=None,
                pass_threshold=pass_threshold,
                fail_threshold=fail_threshold,
                judge_config=config.judge,
                run_config=config,
                selected_parameters={baseline_kind.value: trial},
            )
            if baseline_kind in scicu_kinds:
                predictions = []
                targets = []
                novelty = baseline_kind in {
                    BaselineKind.KNN_SUPPORT,
                    BaselineKind.CENTROID_SUPPORT,
                }
                for case in scicu:
                    result = registry.evaluate(baseline_kind, case)
                    if result.score is None or not math.isfinite(result.score):
                        return -1.0e12
                    predictions.append(float(result.score))
                    validity = float(scicu_structural_validity[case.case_id])
                    targets.append(1.0 - validity if novelty else validity)
                return -float(
                    np.mean(
                        (np.asarray(predictions) - np.asarray(targets)) ** 2
                    )
                )
            scores = []
            for case in dbp:
                result = registry.evaluate(baseline_kind, case)
                if result.score is None or not math.isfinite(result.score):
                    return -1.0e12
                scores.append(float(result.score))
            return float(np.mean(scores))

        trial = freeze_strongest_validation_trial(
            search_space=space,
            count=config.paper.hyperparameter_trials,
            seed=seed,
            validation_score=validation_score,
        )
        selected[kind.value] = dict(trial.parameters)
        audits[kind.value] = trial
        if kind in {BaselineKind.KNN_SUPPORT, BaselineKind.CENTROID_SUPPORT}:
            objectives[kind.value] = "negative_open_set_novelty_brier"
        elif kind in scicu_kinds:
            objectives[kind.value] = "negative_structural_validity_brier"
        else:
            objectives[kind.value] = "mean_frozen_alignment_similarity"
    return FormalBaselineSelection(selected, audits, objectives)


def run_formal_baselines(
    cases: Sequence[EvaluatedCase],
    registry: BaselineRegistry,
) -> tuple[FormalBaselineRow, ...]:
    rows: list[FormalBaselineRow] = []
    registered = set(registry.kinds)
    for case in cases:
        for definition in PAPER_BASELINES:
            if case.dataset not in definition.benchmarks:
                continue
            if definition.kind not in registered:
                raise ValueError(
                    f"compatible baseline is not registered: {definition.kind.value}"
                )
            result = registry.evaluate(definition.kind, case)
            if result.score is None or not math.isfinite(result.score):
                raise RuntimeError(
                    f"baseline {definition.kind.value} produced no finite score "
                    f"for {case.case_id}: {result.status.value} {dict(result.details)}"
                )
            rows.append(
                FormalBaselineRow(
                    case_id=case.case_id,
                    dataset=case.dataset,
                    baseline=definition.kind.value,
                    metric=result.metric,
                    score=float(result.score),
                    status=result.status.value,
                    details=result.details,
                )
            )
            if case.dataset == "dbp15k" and definition.kind is BaselineKind.SINKHORN:
                rows.append(
                    FormalBaselineRow(
                        case_id=case.case_id,
                        dataset=case.dataset,
                        baseline=definition.kind.value,
                        metric="TP",
                        score=compute_tp(
                            case.transport_plan,
                            case.scientific_constraint_mask,
                        ),
                        status=result.status.value,
                        details=result.details + (("evaluation", "frozen_reference_correspondence"),),
                    )
                )
    return tuple(rows)
