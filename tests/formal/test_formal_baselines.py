import math

import numpy as np

from sciprism_x.baselines.llm_judge import LocalInferredJudgeAdapter
from sciprism_x.baselines.registry import PAPER_BASELINES
from sciprism_x.baselines.schema import (
    BaselineStatus,
    EvaluatedCase,
    GraphEdge,
    GraphView,
    LLMJudgeInput,
)
from sciprism_x.formal.baselines import (
    build_formal_baseline_registry,
    run_formal_baselines,
)

from .helpers import run_config


def _registry(*, api_key, requester):
    config = run_config()
    return build_formal_baseline_registry(
        api_key=api_key,
        requester=requester,
        pass_threshold=0.8,
        fail_threshold=0.8,
        judge_config=config.judge,
        run_config=config,
    )


def _evaluated_case(case_id: str, dataset: str) -> EvaluatedCase:
    source = GraphView(
        nodes=("a", "b"),
        edges=(GraphEdge("a", "supports", "b"),),
    )
    target = GraphView(
        nodes=("x", "y"),
        edges=(GraphEdge("x", "supports", "y"),),
    )
    adjacency = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float64)
    return EvaluatedCase(
        case_id=case_id,
        dataset=dataset,
        task="What conclusion follows from the evidence?",
        response="The evidence supports conclusion B with high confidence.",
        reference_context="Evidence A supports conclusion B.",
        predicted_target="B",
        reference_target="B",
        response_segments=("Evidence A", "supports conclusion B"),
        evidence_spans=("Evidence A", "supports conclusion B"),
        task_metadata=(("domain", "physics"),),
        confidence=0.8,
        entropy=0.2,
        representation=np.array([1.0, 2.0], dtype=np.float64),
        validation_support=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        source_graph=source,
        target_graph=target,
        pairwise_cost=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        source_mass=np.array([0.5, 0.5], dtype=np.float64),
        target_mass=np.array([0.5, 0.5], dtype=np.float64),
        source_adjacency=adjacency,
        target_adjacency=adjacency,
        candidate_scores=np.array([0.9, 0.2, 0.1], dtype=np.float64),
        correct_candidate_index=0,
        residuals=np.ones(6, dtype=np.float64),
        transport_plan=np.eye(2, dtype=np.float64),
        generator_identity="withheld-generator",
        motif_graph="withheld-motif-graph",
        scientific_constraint_mask=np.eye(2, dtype=np.bool_),
        boundary_label="withheld-boundary",
        open_set_category="withheld-open-set",
    )


def test_local_judge_returns_numeric_score_without_credentials() -> None:
    adapter = LocalInferredJudgeAdapter(
        pass_threshold=0.8,
        fail_threshold=0.8,
        component_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
    )
    result = adapter.evaluate(
        LLMJudgeInput(
            case_id="case-1",
            task="What conclusion follows?",
            response="Evidence A supports conclusion B.",
            reference_context="Evidence A supports conclusion B.",
        )
    )

    assert result.status is BaselineStatus.AVAILABLE
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert dict(result.details)["route"] == "local_inferred"


def test_exact_judge_is_selected_only_with_both_external_inputs() -> None:
    calls: list[tuple[str, str]] = []

    def requester(request, api_key):
        calls.append((request.model, api_key))
        return {
            "answer_quality": 0.8,
            "evidence_support": 0.7,
            "reasoning_consistency": 0.8,
            "uncertainty_handling": 0.7,
            "final_score": 0.75,
            "brief_error_type": "minor_evidence_gap",
        }

    for registry in (
        _registry(api_key=None, requester=None),
        _registry(api_key="secret", requester=None),
    ):
        rows = run_formal_baselines((_evaluated_case("scicu-1", "scicueval"),), registry)
        judge = next(row for row in rows if row.baseline == "llm_judge")
        assert judge.status == "available"
        assert dict(judge.details)["route"] == "local_inferred"

    exact = _registry(api_key="secret", requester=requester)
    rows = run_formal_baselines((_evaluated_case("scicu-1", "scicueval"),), exact)
    judge = next(row for row in rows if row.baseline == "llm_judge")
    assert judge.score == 0.75
    assert calls == [("gpt-4o-2024-08-06", "secret")]


def test_every_compatible_baseline_produces_finite_smoke_output() -> None:
    cases = (
        _evaluated_case("scicu-1", "scicueval"),
        _evaluated_case("dbp-1", "dbp15k"),
    )
    rows = run_formal_baselines(
        cases,
        _registry(api_key=None, requester=None),
    )

    assert len(rows) == 17
    assert any(
        row.dataset == "dbp15k"
        and row.baseline == "sinkhorn"
        and row.metric == "TP"
        for row in rows
    )
    assert all(math.isfinite(row.score) for row in rows)
    assert not any(row.status == "unavailable_external" for row in rows)
    assert {
        (row.case_id, row.baseline)
        for row in rows
    } == {
        (case.case_id, definition.kind.value)
        for case in cases
        for definition in PAPER_BASELINES
        if case.dataset in definition.benchmarks
    }
