import numpy as np
import pytest


def test_paper_baseline_registry_covers_every_compatible_family() -> None:
    from sciprism_x.baselines.registry import PAPER_BASELINES
    from sciprism_x.baselines.schema import BaselineKind

    assert {definition.kind for definition in PAPER_BASELINES} == set(BaselineKind)
    family_only = [definition for definition in PAPER_BASELINES if definition.family_only]
    assert family_only
    assert all(definition.official_revision is None for definition in family_only)


def test_endpoint_and_generic_support_adapters(evaluated_case) -> None:
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input
    from sciprism_x.baselines.scientific import EndpointAdapter, SupportDistanceAdapter

    endpoint = EndpointAdapter(
        case_sensitive=False,
        strip_whitespace=True,
        collapse_whitespace=True,
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.ENDPOINT))
    support = SupportDistanceAdapter(
        mode="knn",
        neighbors=1,
        distance_scale=1.0,
        clip_to_unit_interval=True,
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.KNN_SUPPORT))

    assert endpoint.score == 1.0
    assert support.score == pytest.approx(1.0)


def test_remaining_scientific_baseline_families(evaluated_case) -> None:
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input
    from sciprism_x.baselines.scientific import (
        AttributionAdapter,
        FixedTaxonomyAdapter,
        SupportDistanceAdapter,
        UncertaintyAdapter,
    )

    attribution = AttributionAdapter(
        lowercase=True,
        token_pattern=r"[A-Za-z]+",
        empty_overlap_score=0.0,
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.ATTRIBUTION))
    taxonomy = FixedTaxonomyAdapter(
        category_patterns=(("unsupported", ("unsupported",)),),
        no_failure_score=1.0,
        failure_score=0.0,
        lowercase=True,
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.FIXED_TAXONOMY))
    uncertainty = UncertaintyAdapter(
        mode="product",
        maximum_entropy=1.0,
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.UNCERTAINTY))
    centroid = SupportDistanceAdapter(
        mode="centroid",
        neighbors=1,
        distance_scale=2.0,
        clip_to_unit_interval=True,
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.CENTROID_SUPPORT))

    assert attribution.score == pytest.approx(0.8)
    assert taxonomy.score == 1.0
    assert uncertainty.score == pytest.approx(0.64)
    assert centroid.score == pytest.approx(np.sqrt(2.5) / 2.0)


def test_graph_and_ranking_adapters(evaluated_case) -> None:
    from sciprism_x.baselines.graph import (
        GraphEditAdapter,
        HitsAt1Adapter,
        MRRAdapter,
        RelationAwareAdapter,
    )
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    graph_input = project_baseline_input(evaluated_case, BaselineKind.GRAPH_EDIT)
    ranking_input = project_baseline_input(evaluated_case, BaselineKind.HITS_AT_1)

    assert GraphEditAdapter(
        node_insertion_deletion_cost=1.0,
        relation_insertion_deletion_cost=1.0,
        relation_substitution_cost=1.0,
        timeout_seconds=2.0,
    ).evaluate(graph_input).score == pytest.approx(1.0)
    assert RelationAwareAdapter(multiset=True).evaluate(graph_input).score == 1.0
    assert HitsAt1Adapter(descending=True).evaluate(ranking_input).score == 1.0
    assert MRRAdapter(descending=True).evaluate(ranking_input).score == 1.0


def test_unconstrained_sinkhorn_uses_no_scientific_mask(evaluated_case) -> None:
    from sciprism_x.baselines.graph import UnconstrainedSinkhornAdapter
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    result = UnconstrainedSinkhornAdapter(
        epsilon=0.05,
        max_iterations=1000,
        marginal_tolerance=1.0e-9,
        objective_tolerance=1.0e-12,
        stable_iterations=5,
        score_transform="exp_negative_cost",
    ).evaluate(project_baseline_input(evaluated_case, BaselineKind.SINKHORN))

    assert result.score is not None
    assert 0.0 < result.score <= 1.0
    assert result.status.value == "available"


def test_relational_transport_baselines(evaluated_case) -> None:
    from sciprism_x.baselines.graph import GromovTransportAdapter, StructureAwareAdapter
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    controls = {
        "epsilon": 0.05,
        "lambda_structure": 1.0,
        "max_iterations": 1000,
        "marginal_tolerance": 1.0e-9,
        "objective_tolerance": 1.0e-12,
        "stable_iterations": 5,
        "score_transform": "exp_negative_cost",
    }
    gromov = GromovTransportAdapter(**controls).evaluate(
        project_baseline_input(evaluated_case, BaselineKind.GROMOV)
    )
    structure = StructureAwareAdapter(**controls).evaluate(
        project_baseline_input(evaluated_case, BaselineKind.STRUCTURE_AWARE)
    )

    assert gromov.score is not None
    assert 0.0 < gromov.score <= 1.0
    assert structure.score is not None
    assert 0.0 < structure.score <= 1.0


def test_llm_judge_request_and_unavailable_external(evaluated_case) -> None:
    from sciprism_x.baselines.llm_judge import LLMJudgeAdapter
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    adapter = LLMJudgeAdapter(
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=512,
        api_key=None,
        requester=None,
    )
    projected = project_baseline_input(evaluated_case, BaselineKind.LLM_JUDGE)
    request = adapter.build_request(projected)
    result = adapter.evaluate(projected)

    assert request.input == {
        "task": evaluated_case.task,
        "response": evaluated_case.response,
        "reference_context": evaluated_case.reference_context,
    }
    assert set(request.response_schema["properties"]) == {
        "answer_quality",
        "evidence_support",
        "reasoning_consistency",
        "uncertainty_handling",
        "final_score",
        "brief_error_type",
    }
    assert request.case_id == evaluated_case.case_id
    assert "Do not provide chain-of-thought" in request.instructions
    assert result.status.value == "unavailable_external"
    assert result.score is None


def test_family_search_freezes_best_of_exactly_fifty_trials() -> None:
    from sciprism_x.baselines.registry import freeze_strongest_validation_trial

    resolved = freeze_strongest_validation_trial(
        search_space={"alpha": [0.1, 0.5, 1.0], "beta": [1.0, 2.0]},
        count=50,
        seed=42,
        validation_score=lambda trial: -abs(trial["alpha"] - 0.5) - trial["beta"] / 100,
    )

    assert resolved.trial_count == 50
    assert resolved.parameters == (("alpha", 0.5), ("beta", 1.0))
    assert resolved.validation_score == pytest.approx(-0.01)


def test_registry_projects_before_adapter_evaluation(evaluated_case) -> None:
    from sciprism_x.baselines.registry import BaselineRegistry
    from sciprism_x.baselines.schema import BaselineKind
    from sciprism_x.baselines.scientific import EndpointAdapter

    registry = BaselineRegistry(
        adapters={
            BaselineKind.ENDPOINT: EndpointAdapter(
                case_sensitive=False,
                strip_whitespace=True,
                collapse_whitespace=True,
            )
        }
    )

    assert registry.evaluate(BaselineKind.ENDPOINT, evaluated_case).score == 1.0
