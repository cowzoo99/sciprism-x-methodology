import dataclasses

import numpy as np
import pytest


FORBIDDEN_FIELDS = {
    "residuals",
    "transport_plan",
    "generator_identity",
    "motif_graph",
    "scientific_constraint_mask",
    "boundary_label",
    "open_set_category",
}


def test_endpoint_baseline_cannot_receive_residuals(evaluated_case) -> None:
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    projected = project_baseline_input(evaluated_case, BaselineKind.ENDPOINT)

    assert dataclasses.asdict(projected) == {
        "case_id": "case-1",
        "predicted_target": "B",
        "reference_target": "B",
    }
    assert all(not hasattr(projected, field) for field in FORBIDDEN_FIELDS)


@pytest.mark.parametrize(
    "kind_name",
    [
        "ATTRIBUTION",
        "FIXED_TAXONOMY",
        "UNCERTAINTY",
        "KNN_SUPPORT",
        "CENTROID_SUPPORT",
        "GRAPH_EDIT",
        "SINKHORN",
        "GROMOV",
        "STRUCTURE_AWARE",
        "RELATION_AWARE",
        "LLM_JUDGE",
        "HITS_AT_1",
        "MRR",
    ],
)
def test_every_projection_withholds_sciprism_internal_outputs(
    evaluated_case, kind_name: str
) -> None:
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    projected = project_baseline_input(evaluated_case, BaselineKind[kind_name])

    assert all(not hasattr(projected, field) for field in FORBIDDEN_FIELDS)


def test_llm_judge_projection_contains_only_allowed_content(evaluated_case) -> None:
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    projected = project_baseline_input(evaluated_case, BaselineKind.LLM_JUDGE)

    assert set(dataclasses.asdict(projected)) == {
        "case_id",
        "task",
        "response",
        "reference_context",
    }


def test_numeric_projections_are_independent_read_only_copies(evaluated_case) -> None:
    from sciprism_x.baselines.schema import BaselineKind, project_baseline_input

    support = project_baseline_input(evaluated_case, BaselineKind.KNN_SUPPORT)
    graph = project_baseline_input(evaluated_case, BaselineKind.SINKHORN)
    ranking = project_baseline_input(evaluated_case, BaselineKind.MRR)

    assert not np.shares_memory(support.representation, evaluated_case.representation)
    assert not np.shares_memory(graph.pairwise_cost, evaluated_case.pairwise_cost)
    assert not np.shares_memory(ranking.candidate_scores, evaluated_case.candidate_scores)
    with pytest.raises(ValueError, match="read-only"):
        support.representation[0] = 99.0
