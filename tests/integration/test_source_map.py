import json
from pathlib import Path

from sciprism_x.config import load_run_config
from sciprism_x.provenance import ProvenanceClass


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_every_core_module_is_mapped_to_method_source() -> None:
    method_map = (PROJECT_ROOT / "docs" / "method_map.md").read_text(encoding="utf-8")
    modules = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src" / "sciprism_x").rglob("*.py")
    )

    assert modules
    assert not [module for module in modules if module not in method_map]


def test_every_implementation_setting_key_is_documented() -> None:
    parameters = (PROJECT_ROOT / "docs" / "implementation_parameters.md").read_text(
        encoding="utf-8"
    )
    config = load_run_config(PROJECT_ROOT / "configs" / "track_b.yaml")
    keys = {
        record.path
        for record in config.parameter_records()
        if record.provenance is ProvenanceClass.IMPLEMENTATION_SETTING
    }
    keys.update(
        {
            "workflow.fixture_grid_selection",
            "workflow.fixture_scope",
            "workflow.infeasible_transport_mapping",
        }
    )

    assert keys
    assert not [key for key in sorted(keys) if f"`{key}`" not in parameters]


def test_source_hashes_and_official_revisions_are_crosswalked() -> None:
    method_map = (PROJECT_ROOT / "docs" / "method_map.md").read_text(encoding="utf-8")
    registry = json.loads(
        (PROJECT_ROOT / "data" / "manifests" / "source_registry.json").read_text(
            encoding="utf-8"
        )
    )

    required_hashes = {
        "9611FA08BB2DAAF52A90DC9CC2262DC38EE799E26F1938E5FC5BC95E6E066934",
        "822DDCC81F767FE5148786127C09750D50484540BB322B5CC164D9620943AEC2",
        registry["repositories"]["scicueval"]["commit"],
        registry["repositories"]["jape"]["commit"],
        registry["sources"][-1]["sha256"],
    }
    assert not [value for value in required_hashes if value not in method_map]
