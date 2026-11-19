def _known_profile():
    from sciprism_x.responses.schema import ReconstructionProfile

    return ReconstructionProfile(
        physical_error=0.07,
        logical_error=0.11,
        unit_error=0.13,
        causal_error=0.17,
        modal_evidence_error=0.19,
        uncertainty_error=0.23,
        endpoint_error=0.29,
        motif_omission=0.31,
        unsupported_content=0.37,
        length_ratio=1.21,
        evidence_synthesis=0.83,
    )


def test_fit_recovers_known_profile_repeatably() -> None:
    from sciprism_x.responses.fit import (
        ProfileFitRequest,
        ProfileGroup,
        fit_profiles,
        profile_observables,
    )

    known = _known_profile()
    request = ProfileFitRequest(
        group=ProfileGroup(
            model_id="gpt-4o-2024-08-06",
            regime="iid",
            domain="physics",
            competency="multi-source integration",
        ),
        targets=profile_observables(known),
        weights={name: 1.0 for name in profile_observables(known)},
    )

    first = fit_profiles((request,), seed=42)
    second = fit_profiles((request,), seed=42)

    assert first == second
    fitted = first[0].profile.model_dump()
    expected = known.model_dump()
    assert max(abs(fitted[name] - expected[name]) for name in expected) < 0.05
    assert first[0].optimizer.strategy == "best1bin"
    assert first[0].optimizer.maxiter == 500
    assert first[0].optimizer.popsize == 15
    assert first[0].optimizer.tol == 1.0e-9
    assert first[0].optimizer.mutation == (0.5, 1.0)
    assert first[0].optimizer.recombination == 0.7
    assert first[0].optimizer.polish is False
    assert first[0].convergence_trace


def test_fit_records_targets_weights_bounds_and_group() -> None:
    from sciprism_x.responses.fit import (
        ProfileFitRequest,
        ProfileGroup,
        fit_profiles,
        profile_observables,
    )

    known = _known_profile()
    targets = profile_observables(known)
    group = ProfileGroup(
        model_id="claude-3.5-sonnet-20241022",
        regime="boundary_stress",
        domain="physics",
        competency="multi-source integration",
    )
    result = fit_profiles(
        (
            ProfileFitRequest(
                group=group,
                targets=targets,
                weights={name: 2.0 for name in targets},
            ),
        ),
        seed=123,
    )[0]

    assert result.group == group
    assert result.target_vector == targets
    assert result.weights == {name: 2.0 for name in targets}
    assert set(result.bounds) == set(targets)


def test_published_targets_match_reported_tables() -> None:
    import json
    from pathlib import Path

    payload = json.loads(
        Path("data/manifests/published_targets.json").read_text(encoding="utf-8")
    )

    assert payload["scicueval"]["TSR"]["iid"]["method_mean"] == 0.919
    assert payload["scicueval"]["PCS"]["boundary_stress"]["method_mean"] == 0.754
    assert payload["scicueval"]["BSI"]["boundary_stress"]["method_mean"] == 0.115
    assert payload["dbp15k"]["SCS"]["boundary_stress"]["method_mean"] == 0.716
    assert payload["dbp15k"]["TP"]["boundary_stress"]["baseline_mean"] == 0.527
    assert payload["worked_case"]["case_id"] == "SCICU-0417"
    assert payload["worked_case"]["pcs"] == 0.61
    assert payload["worked_case"]["bsi"] == 0.28
    assert payload["fit_availability"]["published_strata"] == ["aggregate_by_regime"]
