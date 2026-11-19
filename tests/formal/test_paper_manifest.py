from pathlib import Path

from sciprism_x.formal.datasets import load_formal_datasets
from sciprism_x.formal.manifest import build_paper_manifest

from .helpers import formal_config, run_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_paper_manifest_declares_exact_experiment_matrix(tmp_path: Path) -> None:
    datasets = load_formal_datasets(PROJECT_ROOT / "data", extraction_root=tmp_path)
    manifest = build_paper_manifest(run_config(), formal_config(), datasets)

    assert manifest.scicu_response_count == 75_600
    assert manifest.scicu_nonstress_counts == {
        "iid": 600,
        "motif_or_structure": 600,
        "domain_transfer": 600,
    }
    assert manifest.scicu_boundary_count == 2_400
    assert manifest.dbp_nonstress_counts == {
        "iid": 600,
        "motif_or_structure": 600,
        "domain_transfer": 600,
    }
    assert manifest.dbp_boundary_count == 2_400
    assert manifest.seeds == (42, 0, 1, 123, 1234)
