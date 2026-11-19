from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "SciPRISM_X_Methodology.ipynb"
REQUIRED_HEADINGS = [
    "SciPRISM-X Methodology",
    "1. Environment and Unified Entry Point",
    "2. Pinned Environment and Execution Scale",
    "3. Method Parameter Specification",
    "4. Run the Complete Formal Workflow",
    "5. Integrity and Provenance Checks",
    "6. Load All Computed Results",
    "7. Numeric Finiteness and Paired-Result Checks",
    "8. Reference and Computed Values",
    "9. Save Results and Hash Manifest",
    "10. Deterministic Plotting Entry Point",
]


def test_notebook_has_ordered_sections() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    headings = [
        cell.source.splitlines()[0].lstrip("# ").strip()
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("#")
    ]
    positions = [headings.index(name) for name in REQUIRED_HEADINGS]
    assert positions == sorted(positions)


def test_notebook_is_thin_formal_orchestration_with_one_plot_loader() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "run_formal_experiment" in code
    assert 'os.environ.get("SCIPRISM_SCALE", "paper")' in code
    assert "FORMAL_STAGE_ORDER" in code
    assert "run_track_b" not in code
    assert "fixture_count" not in code
    assert code.count("def load_plot_inputs") == 1
    for artifact in (
        "per_case_metrics.parquet",
        "baselines.parquet",
        "ablations.parquet",
        "boundaries.parquet",
        "open_set.parquet",
        "aggregate_metrics.json",
        "bootstrap_results.parquet",
        "significance_results.parquet",
        "discrepancy_report.csv",
    ):
        assert artifact in code
