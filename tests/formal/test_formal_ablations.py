from dataclasses import dataclass, replace
import math

import numpy as np

from sciprism_x.formal.ablations import (
    AblationControls,
    AblationKind,
    PerCaseMetric,
    resolve_ablation_policy,
    run_formal_ablations,
    typed_greedy_minimum_cost_matching,
)


@dataclass(frozen=True, slots=True)
class _SmokeBundle:
    case_ids: tuple[str, ...]


class _SmokeEvaluator:
    def __init__(self, controls: AblationControls) -> None:
        self.resolved_controls = controls

    def evaluate(self, bundle: object) -> tuple[PerCaseMetric, ...]:
        assert isinstance(bundle, _SmokeBundle)
        disabled = sum(
            (
                not self.resolved_controls.motif_structure,
                not self.resolved_controls.hard_constraints,
                not self.resolved_controls.soft_penalties,
                not self.resolved_controls.residual_channels,
                not self.resolved_controls.state_field,
                not self.resolved_controls.boundary_estimation,
                not self.resolved_controls.open_set,
                self.resolved_controls.transport_mode == "greedy",
            )
        )
        return tuple(
            PerCaseMetric(
                case_id=case_id,
                dataset="scicueval",
                metric=metric,
                value=1.0 - disabled / 10.0,
            )
            for case_id in bundle.case_ids
            for metric in ("PCS", "CCS", "BSI", "UPDQ", "SC", "SCS", "TP")
        )


def _evaluator_factory(
    kind: AblationKind | None = None,
) -> _SmokeEvaluator:
    controls = (
        AblationControls()
        if kind is None
        else resolve_ablation_policy(kind).controls
    )
    return _SmokeEvaluator(controls)


def test_all_paper_ablations_are_recomputed_and_paired() -> None:
    rows = run_formal_ablations(
        _SmokeBundle(("case-1", "case-2")),
        _evaluator_factory,
    )
    expected = {
        "without_motif_structure",
        "without_hard_constraints",
        "without_soft_penalties",
        "without_residual_channels",
        "without_state_field",
        "without_boundary_estimation",
        "without_open_set",
        "without_transport",
    }

    assert {kind.value for kind in AblationKind} == expected
    assert {row.ablation for row in rows} == expected
    assert len(rows) == 2 * 7 * 8
    assert all(math.isfinite(row.metric_value) for row in rows)
    assert all(math.isfinite(row.main_value) for row in rows)
    assert all(row.delta == row.main_value - row.metric_value for row in rows)
    assert all(row.source == "computed" for row in rows)


def test_ablation_changes_only_its_declared_control() -> None:
    main = _evaluator_factory().resolved_controls
    expected = {
        AblationKind.WITHOUT_MOTIF_STRUCTURE: replace(main, motif_structure=False),
        AblationKind.WITHOUT_HARD_CONSTRAINTS: replace(main, hard_constraints=False),
        AblationKind.WITHOUT_SOFT_PENALTIES: replace(main, soft_penalties=False),
        AblationKind.WITHOUT_RESIDUAL_CHANNELS: replace(main, residual_channels=False),
        AblationKind.WITHOUT_STATE_FIELD: replace(main, state_field=False),
        AblationKind.WITHOUT_BOUNDARY_ESTIMATION: replace(
            main, boundary_estimation=False
        ),
        AblationKind.WITHOUT_OPEN_SET: replace(main, open_set=False),
        AblationKind.WITHOUT_TRANSPORT: replace(main, transport_mode="greedy"),
    }

    assert {
        kind: _evaluator_factory(kind).resolved_controls
        for kind in AblationKind
    } == expected


def test_without_transport_uses_typed_greedy_matching() -> None:
    plan = typed_greedy_minimum_cost_matching(
        node_cost=np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64),
        source_mass=np.array([0.5, 0.5], dtype=np.float64),
        target_mass=np.array([0.5, 0.5], dtype=np.float64),
        compatible=np.array([[True, False], [False, True]], dtype=np.bool_),
        tolerance=1.0e-12,
        numerical_tolerance=1.0e-9,
        lp_method="highs-ds",
        lp_presolve=True,
        lp_primal_feasibility_tolerance=1.0e-9,
        lp_dual_feasibility_tolerance=1.0e-9,
        lp_edge_weight_strategy="dantzig",
    )

    np.testing.assert_allclose(plan, np.diag([0.5, 0.5]))
    np.testing.assert_allclose(plan.sum(axis=1), [0.5, 0.5])
    np.testing.assert_allclose(plan.sum(axis=0), [0.5, 0.5])


def test_typed_greedy_preserves_a_feasible_path_for_constrained_nodes() -> None:
    plan = typed_greedy_minimum_cost_matching(
        node_cost=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        source_mass=np.array([0.5, 0.5], dtype=np.float64),
        target_mass=np.array([0.5, 0.5], dtype=np.float64),
        compatible=np.array([[True, True], [True, False]], dtype=np.bool_),
        tolerance=1.0e-12,
        numerical_tolerance=1.0e-9,
        lp_method="highs-ds",
        lp_presolve=True,
        lp_primal_feasibility_tolerance=1.0e-9,
        lp_dual_feasibility_tolerance=1.0e-9,
        lp_edge_weight_strategy="dantzig",
    )

    np.testing.assert_allclose(plan, np.array([[0.0, 0.5], [0.5, 0.0]]))
