from __future__ import annotations

from dataclasses import dataclass
from html import escape
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlotStyle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    style_id: str = Field(min_length=1)
    canvas_width: int = Field(gt=0)
    canvas_height: int = Field(gt=0)
    margin_left: int = Field(ge=0)
    margin_right: int = Field(ge=0)
    margin_top: int = Field(ge=0)
    margin_bottom: int = Field(ge=0)
    font_family: str = Field(min_length=1)
    title_font_size: int = Field(gt=0)
    axis_font_size: int = Field(gt=0)
    label_font_size: int = Field(gt=0)
    annotation_font_size: int = Field(gt=0)
    line_width: float = Field(gt=0.0)
    grid_width: float = Field(gt=0.0)
    background: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    text_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    axis_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    grid_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    palette: tuple[str, ...] = Field(min_length=1)
    heatmap_negative: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    heatmap_neutral: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    heatmap_positive: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    decimal_places: int = Field(ge=0, le=6)
    aggregate_metric_order: tuple[str, ...] = Field(min_length=1)
    ablation_order: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_canvas(self) -> "PlotStyle":
        if self.margin_left + self.margin_right >= self.canvas_width:
            raise ValueError("horizontal plot margins consume the canvas")
        if self.margin_top + self.margin_bottom >= self.canvas_height:
            raise ValueError("vertical plot margins consume the canvas")
        if any(
            not isinstance(color, str)
            or re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None
            for color in self.palette
        ):
            raise ValueError("plot palette entries must be six-digit hex colors")
        if len(self.aggregate_metric_order) != len(set(self.aggregate_metric_order)):
            raise ValueError("aggregate metric order must not contain duplicates")
        if len(self.ablation_order) != len(set(self.ablation_order)):
            raise ValueError("ablation order must not contain duplicates")
        return self


def load_plot_style(path: Path) -> PlotStyle:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plotting configuration must be a mapping")
    return PlotStyle.model_validate(payload)


def _write_svg(target: Path, fragments: list[str], style: PlotStyle) -> Path:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{style.canvas_width}" height="{style.canvas_height}" '
            f'viewBox="0 0 {style.canvas_width} {style.canvas_height}">'
        ),
        f'<rect width="100%" height="100%" fill="{style.background}"/>',
        *fragments,
        "</svg>",
        "",
    ]
    path.write_text("\n".join(svg), encoding="utf-8", newline="\n")
    return path


def _text(
    x: float,
    y: float,
    value: str,
    *,
    style: PlotStyle,
    size: int,
    anchor: str = "middle",
    weight: str = "normal",
    transform: str = "",
) -> str:
    transform_attribute = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="{escape(style.font_family)}" font-size="{size}" '
        f'font-weight="{weight}" fill="{style.text_color}"{transform_attribute}>'
        f"{escape(value)}</text>"
    )


def render_aggregate_metrics_svg(
    aggregate: Mapping[str, object],
    *,
    style: PlotStyle,
    target: Path,
) -> Path:
    if aggregate.get("source") != "computed":
        raise ValueError("aggregate plot accepts only computed-namespace values")
    lookup = {str(key).upper(): value for key, value in aggregate.items()}
    metrics = style.aggregate_metric_order
    values = []
    for metric in metrics:
        if metric not in lookup:
            raise ValueError(f"aggregate plot is missing metric: {metric}")
        value = float(lookup[metric])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"aggregate metric must be in [0, 1]: {metric}")
        values.append(value)

    left = style.margin_left
    top = style.margin_top
    right = style.canvas_width - style.margin_right
    bottom = style.canvas_height - style.margin_bottom
    width = right - left
    height = bottom - top
    fragments = [
        _text(
            style.canvas_width / 2,
            30,
            "SciPRISM-X aggregate metrics",
            style=style,
            size=style.title_font_size,
            weight="bold",
        )
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = bottom - value * height
        fragments.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            f'stroke="{style.grid_color}" stroke-width="{style.grid_width}"/>'
        )
        fragments.append(
            _text(
                left - 12,
                y + 5,
                f"{value:.1f}",
                style=style,
                size=style.axis_font_size,
                anchor="end",
            )
        )
    fragments.extend(
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
            f'stroke="{style.axis_color}" stroke-width="{style.line_width}"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
            f'stroke="{style.axis_color}" stroke-width="{style.line_width}"/>',
        )
    )
    slot = width / len(metrics)
    bar_width = slot * 0.64
    for index, (metric, value) in enumerate(zip(metrics, values, strict=True)):
        x = left + index * slot + (slot - bar_width) / 2
        bar_height = value * height
        y = bottom - bar_height
        color = style.palette[index % len(style.palette)]
        fragments.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="{color}"/>'
        )
        fragments.append(
            _text(
                x + bar_width / 2,
                max(top + style.annotation_font_size, y - 8),
                f"{value:.{style.decimal_places}f}",
                style=style,
                size=style.annotation_font_size,
            )
        )
        fragments.append(
            _text(
                x + bar_width / 2,
                bottom + 28,
                metric,
                style=style,
                size=style.label_font_size,
                weight="bold",
            )
        )
    fragments.append(
        _text(
            24,
            top + height / 2,
            "Score",
            style=style,
            size=style.axis_font_size,
            weight="bold",
            transform=f"rotate(-90 24 {top + height / 2:.2f})",
        )
    )
    return _write_svg(Path(target), fragments, style)


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _blend(left: str, right: str, fraction: float) -> str:
    clipped = min(max(fraction, 0.0), 1.0)
    first = _rgb(left)
    second = _rgb(right)
    values = tuple(
        round(a + (b - a) * clipped) for a, b in zip(first, second, strict=True)
    )
    return "#" + "".join(f"{value:02X}" for value in values)


def render_ablation_heatmap_svg(
    frame: pd.DataFrame,
    *,
    style: PlotStyle,
    target: Path,
) -> Path:
    required = {"ablation", "metric", "delta", "source"}
    if not required.issubset(frame.columns):
        raise ValueError("ablation plotting frame is missing required columns")
    if frame.empty or set(frame["source"]) != {"computed"}:
        raise ValueError("ablation plot accepts non-empty computed rows only")
    means = frame.groupby(["ablation", "metric"], sort=True)["delta"].mean()
    observed_ablations = set(means.index.get_level_values("ablation"))
    undeclared_ablations = observed_ablations - set(style.ablation_order)
    if undeclared_ablations:
        raise ValueError(
            f"ablation plot contains undeclared rows: {sorted(undeclared_ablations)}"
        )
    ablations = tuple(
        value for value in style.ablation_order if value in observed_ablations
    )
    metrics = tuple(sorted(set(means.index.get_level_values("metric"))))
    if not ablations or not metrics:
        raise ValueError("ablation plot has no declared rows or metrics")
    expected_cells = {(ablation, metric) for ablation in ablations for metric in metrics}
    observed_cells = set(means.index)
    missing_cells = sorted(expected_cells - observed_cells)
    if missing_cells:
        raise ValueError(f"ablation plot is missing paired cells: {missing_cells}")
    maximum = max(abs(float(value)) for value in means.values)
    scale = maximum if maximum > 0.0 else 1.0
    left = style.margin_left + 190
    top = style.margin_top + 20
    right = style.canvas_width - style.margin_right
    bottom = style.canvas_height - style.margin_bottom
    cell_width = (right - left) / len(metrics)
    cell_height = (bottom - top) / len(ablations)
    fragments = [
        _text(
            style.canvas_width / 2,
            30,
            "Mean paired ablation delta",
            style=style,
            size=style.title_font_size,
            weight="bold",
        )
    ]
    for column, metric in enumerate(metrics):
        fragments.append(
            _text(
                left + (column + 0.5) * cell_width,
                top - 12,
                metric,
                style=style,
                size=style.label_font_size,
                weight="bold",
            )
        )
    for row, ablation in enumerate(ablations):
        y = top + row * cell_height
        fragments.append(
            _text(
                left - 12,
                y + cell_height / 2 + 5,
                ablation.replace("without_", "− ").replace("_", " "),
                style=style,
                size=style.label_font_size,
                anchor="end",
            )
        )
        for column, metric in enumerate(metrics):
            value = float(means.loc[(ablation, metric)])
            fraction = min(abs(value) / scale, 1.0)
            endpoint = style.heatmap_positive if value >= 0.0 else style.heatmap_negative
            color = _blend(style.heatmap_neutral, endpoint, fraction)
            x = left + column * cell_width
            fragments.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" '
                f'height="{cell_height:.2f}" fill="{color}" '
                f'stroke="{style.background}" stroke-width="1"/>'
            )
            fragments.append(
                _text(
                    x + cell_width / 2,
                    y + cell_height / 2 + 5,
                    f"{value:.{style.decimal_places}f}",
                    style=style,
                    size=style.annotation_font_size,
                )
            )
    return _write_svg(Path(target), fragments, style)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_figure_manifest(
    *,
    target: Path,
    style_path: Path,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> Path:
    payload = {
        "schema_version": 1,
        "style": {
            "path": Path(style_path).name,
            "sha256": _sha256(Path(style_path)),
        },
        "inputs": {
            name: {"path": path.name, "sha256": _sha256(path)}
            for name, path in sorted(inputs.items())
        },
        "outputs": {
            name: {"path": path.name, "sha256": _sha256(path)}
            for name, path in sorted(outputs.items())
        },
        "paper_reference_used_as_plot_input": False,
    }
    path = Path(target)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path
