import numpy as np
import pytest


def test_boundary_score_and_censoring() -> None:
    from sciprism_x.perturbations.boundary import find_boundary

    result = find_boundary(
        radii=[0.25, 0.5, 1.0],
        is_valid=lambda radius: radius < 0.5,
        rho_max=1.0,
    )

    assert result.rho_star == 0.5
    assert result.bsi == pytest.approx(0.5)
    assert result.right_censored is False
    assert result.evaluated_radii == (0.25, 0.5)


def test_boundary_is_right_censored_when_no_regime_change_occurs() -> None:
    from sciprism_x.perturbations.boundary import find_boundary

    result = find_boundary(
        radii=[0.25, 0.5, 1.0],
        is_valid=lambda _radius: True,
        rho_max=1.0,
    )

    assert result.rho_star == 1.0
    assert result.bsi == pytest.approx(0.0)
    assert result.right_censored is True
    assert result.evaluated_radii == (0.25, 0.5, 1.0)


def test_scientifically_scaled_norm_uses_validation_mad() -> None:
    from sciprism_x.perturbations.boundary import (
        fit_mad_scale,
        scientifically_scaled_norm,
    )

    validation = np.array(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
        dtype=np.float64,
    )
    scale = fit_mad_scale(validation, floor=1.0e-12)

    np.testing.assert_allclose(scale, [1.0, 2.0])
    assert scientifically_scaled_norm(
        delta=np.array([1.0, 2.0]),
        mad_scale=scale,
        scientific_weights=np.array([1.0, 2.0]),
    ) == pytest.approx(3.0)


def test_scicu_perturbations_audit_every_changed_unit() -> None:
    from sciprism_x.perturbations.scicu import generate_scicu_perturbations

    records = generate_scicu_perturbations(
        task_id="SCICU-1",
        evidence_unit_ids=("e1", "e2", "e3", "e4"),
        context_unit_ids=("c1", "c2", "c3"),
        modality_unit_ids=("m1",),
        symbolic_unit_ids=("s1",),
        boundary_unit_ids=("b1",),
        radii=(0.25, 0.5, 1.0),
        depth_steps=(1, 2, 3),
        context_steps=(1, 2, 3),
        seed=42,
    )

    assert {record.family for record in records} == {
        "evidence_removal",
        "context_composition_shift",
        "modality_conflict",
        "symbolic_depth_increase",
        "boundary_condition_change",
    }
    assert all(record.changed_units for record in records)
    evidence = [record for record in records if record.family == "evidence_removal"]
    assert [len(record.changed_units) for record in evidence] == [1, 2, 3]
    assert all(len(record.changed_edges) == 0 for record in records)


def test_dbp15k_perturbations_audit_removed_edges_and_seeds() -> None:
    from sciprism_x.perturbations.dbp15k import generate_dbp15k_perturbations

    edges = tuple((f"s{i}", "r", f"o{i}") for i in range(10))
    seeds = tuple((f"left{i}", f"right{i}") for i in range(10))
    records = generate_dbp15k_perturbations(
        pair_id="ZH-EN-1",
        source_edges=edges,
        target_edges=edges,
        seed_alignments=seeds,
        removal_ratios=(0.05, 0.10, 0.20),
        seed_fractions=(0.25, 0.20, 0.10),
        original_seed_fraction=0.30,
        seed=42,
    )

    assert {record.family for record in records} == {
        "neighborhood_degradation",
        "relation_ambiguity_injection",
        "seed_alignment_reduction",
        "cross_graph_evidence_weakening",
    }
    assert all(record.changed_edges or record.changed_seed_alignments for record in records)
    seed_records = [
        record for record in records if record.family == "seed_alignment_reduction"
    ]
    assert [record.strength for record in seed_records] == [0.25, 0.20, 0.10]
    assert [len(record.changed_seed_alignments) for record in seed_records] == [2, 3, 7]
