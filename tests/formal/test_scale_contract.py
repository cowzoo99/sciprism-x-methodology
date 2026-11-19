from pathlib import Path

import pytest

from sciprism_x.config import load_run_config
from sciprism_x.formal.config import load_formal_config, resolve_scale_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configs():
    run = load_run_config(PROJECT_ROOT / "configs" / "track_b.yaml")
    formal = load_formal_config(PROJECT_ROOT / "configs" / "formal_execution.yaml")
    return run, formal


def test_paper_scale_contract_has_exact_counts() -> None:
    run, formal = _configs()
    contract = resolve_scale_contract(run, formal, scale="paper")

    assert contract.scicu_nonstress_per_regime == 600
    assert contract.scicu_boundary == 2400
    assert contract.dbp_nonstress_per_regime == 600
    assert contract.dbp_boundary == 2400
    assert contract.generator_count == 6
    assert contract.responses_per_task == 3
    assert contract.seeds == (42, 0, 1, 123, 1234)
    assert contract.scicu_task_count == 4200
    assert contract.scicu_response_count == 75_600


def test_smoke_contract_uses_same_matrix_with_small_counts() -> None:
    run, formal = _configs()
    contract = resolve_scale_contract(run, formal, scale="smoke")

    assert contract.scicu_nonstress_per_regime == 6
    assert contract.scicu_boundary == 15
    assert contract.dbp_nonstress_per_regime == 6
    assert contract.dbp_boundary == 36
    assert contract.generator_count == 6
    assert contract.responses_per_task == 3
    assert contract.scicu_task_count == 33
    assert contract.scicu_response_count == 594


def test_formal_configuration_has_no_extra_or_implicit_fields(tmp_path: Path) -> None:
    config = PROJECT_ROOT / "configs" / "formal_execution.yaml"
    text = config.read_text(encoding="utf-8") + "\nunexpected: true\n"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_formal_config(invalid)


def test_split_fractions_must_sum_to_one(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """schema_version: 1
split_fractions: [0.7, 0.2, 0.2]
smoke_nonstress_cases: 6
smoke_scicu_boundary_cases: 15
smoke_dbp_boundary_cases: 36
response_profile_entropy_weight: 0.01
stable_pcs_quantile: 0.67
stable_residual_quantile: 0.33
invalid_pcs_quantile: 0.33
invalid_residual_quantile: 0.67
proxy_pass_threshold: 0.8
proxy_fail_threshold: 0.8
resume: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to one"):
        load_formal_config(invalid)
