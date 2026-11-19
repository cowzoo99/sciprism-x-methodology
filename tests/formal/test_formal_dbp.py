import math
from pathlib import Path

import pytest

from sciprism_x.formal.allocation import AllocationRow, AllocationTable
from sciprism_x.formal.dbp import (
    build_dbp_context,
    build_dbp_manifest,
    evaluate_dbp_case,
    run_dbp_experiment,
)
from sciprism_x.formal.datasets import (
    DBP15KDataset,
    DBPLanguageDataset,
    FormalDatasets,
    load_formal_datasets,
)

from .helpers import paper_contract, run_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tiny_dataset() -> DBPLanguageDataset:
    return DBPLanguageDataset(
        language_pair="ZH-EN",
        entity_ids_1=((0, "s0"), (1, "s1"), (2, "s2")),
        entity_ids_2=((10, "t10"), (11, "t11"), (12, "t12"), (13, "t13")),
        triples_1=((0, 1, 1), (1, 2, 2), (2, 3, 0)),
        triples_2=((10, 1, 11), (11, 2, 12), (12, 3, 10), (13, 9, 13)),
        seed_pairs=((1, 11), (2, 12)),
        reference_pairs=((0, 10),),
        target_candidate_ids=(10, 11, 12, 13),
    )


def test_dbp_evaluation_ranks_against_full_candidate_pool() -> None:
    dataset = _tiny_dataset()
    context = build_dbp_context(dataset, run_config())
    result = evaluate_dbp_case(0, 10, context)

    assert result.candidate_count == len(dataset.target_candidate_ids)
    assert result.candidate_pruning is False
    assert math.isfinite(result.scs)
    assert math.isfinite(result.transport_purity)
    assert result.hits_at_1 in {0.0, 1.0}
    assert 0.0 < result.mrr <= 1.0
    assert result.correct_rank >= 1


def test_dbp_paper_manifest_has_exact_allocations(tmp_path: Path) -> None:
    datasets = load_formal_datasets(PROJECT_ROOT / "data", extraction_root=tmp_path)
    manifest = build_dbp_manifest(datasets, paper_contract(), seed=42)

    assert manifest.nonstress_counts == {
        "iid": 600,
        "motif_or_structure": 600,
        "domain_transfer": 600,
    }
    assert manifest.boundary_count == 2_400
    assert set(manifest.language_pairs) == {"ZH-EN", "JA-EN", "FR-EN"}


def test_boundary_allocation_cannot_silently_reuse_unperturbed_graph() -> None:
    dataset = _tiny_dataset()
    datasets = FormalDatasets(
        scicueval=(),
        dbp15k=DBP15KDataset(language_pairs={"ZH-EN": dataset}),
    )
    allocations = AllocationTable(
        rows=(
            AllocationRow(
                dataset="dbp15k",
                allocation_id="boundary-1",
                case_id="ZH-EN::0::10",
                regime="boundary_stress",
                family="neighborhood_degradation",
                strength=0.25,
                language_pair="ZH-EN",
            ),
        )
    )

    with pytest.raises(ValueError, match="materialized perturbation"):
        run_dbp_experiment(datasets, allocations, run_config())


def test_dbp_results_preserve_allocation_identity_for_repeated_entity_pairs() -> None:
    dataset = _tiny_dataset()
    datasets = FormalDatasets(
        scicueval=(),
        dbp15k=DBP15KDataset(language_pairs={"ZH-EN": dataset}),
    )
    allocations = AllocationTable(
        rows=tuple(
            AllocationRow(
                dataset="dbp15k",
                allocation_id=allocation_id,
                case_id="ZH-EN::0::10",
                regime=regime,
                family=None,
                strength=None,
                language_pair="ZH-EN",
            )
            for allocation_id, regime in (
                ("iid-1", "iid"),
                ("domain-1", "domain_transfer"),
            )
        )
    )

    results = run_dbp_experiment(datasets, allocations, run_config())

    assert [result.allocation_id for result in results] == ["iid-1", "domain-1"]
    assert [result.regime for result in results] == ["iid", "domain_transfer"]
