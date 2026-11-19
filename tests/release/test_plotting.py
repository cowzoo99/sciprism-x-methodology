from __future__ import annotations

from pathlib import Path

import pandas as pd

from sciprism_x.release.plotting import (
    load_plot_style,
    render_ablation_heatmap_svg,
    render_aggregate_metrics_svg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_plotting_uses_fixed_style_and_computed_inputs(tmp_path: Path) -> None:
    style = load_plot_style(PROJECT_ROOT / "configs" / "plotting.yaml")
    aggregate = {
        "tsr": 0.7,
        "pcs": 0.8,
        "ccs": 0.75,
        "BSI": 0.2,
        "UPDQ": 0.6,
        "SC": 0.78,
        "SCS": 0.81,
        "TP": 0.83,
        "source": "computed",
    }
    aggregate_path = render_aggregate_metrics_svg(
        aggregate,
        style=style,
        target=tmp_path / "aggregate.svg",
    )
    heatmap_path = render_ablation_heatmap_svg(
        pd.DataFrame(
            [
                {
                    "ablation": "without_transport",
                    "metric": "PCS",
                    "delta": 0.2,
                    "source": "computed",
                },
                {
                    "ablation": "without_transport",
                    "metric": "CCS",
                    "delta": 0.1,
                    "source": "computed",
                },
            ]
        ),
        style=style,
        target=tmp_path / "ablation.svg",
    )

    assert aggregate_path.read_text(encoding="utf-8").startswith("<svg")
    assert heatmap_path.read_text(encoding="utf-8").startswith("<svg")
    assert style.palette[:3] == ("#0072B2", "#E69F00", "#009E73")
