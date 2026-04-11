"""Backtest visualization utilities.

This module is decoupled from backtest execution: it only reads persisted
parquet files and writes local Plotly artifacts.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.backtest.metrics import calculate_metrics
from src.logger import logger

_PALETTE = [
    "#0f766e",
    "#2563eb",
    "#f97316",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
    "#be123c",
    "#a16207",
    "#4338ca",
]
_UP_COLOR = "#dc2626"
_DOWN_COLOR = "#16a34a"
_UP_PALETTE = ["#dc2626", "#ef4444", "#be123c", "#f97316", "#b91c1c", "#fb7185"]
_DOWN_PALETTE = ["#16a34a", "#059669", "#0f766e", "#65a30d", "#15803d", "#22c55e"]


@dataclass(frozen=True)
class BacktestVisualizationResult:
    run_id: str
    output_dir: Path
    summary_path: Path
    report_path: Path
    chart_paths: list[Path]
    image_paths: list[Path]

    @property
    def artifact_paths(self) -> list[Path]:
        return [self.summary_path, self.report_path, *self.chart_paths, *self.image_paths]


def _load_json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _read_latest_metrics(metrics_path: Path | None, results_df: pl.DataFrame) -> dict[str, Any]:
    if metrics_path is not None and metrics_path.exists():
        metrics_df = pl.read_parquet(metrics_path)
        if len(metrics_df) > 0:
            return dict(metrics_df.sort("as_of_week").tail(1).row(0, named=True))

    metrics = calculate_metrics(results_df)
    return metrics.model_dump()


def _prepare_results(results_df: pl.DataFrame, run_id: str | None = None) -> pl.DataFrame:
    df = results_df
    if run_id and "run_id" in df.columns:
        df = df.filter(pl.col("run_id") == run_id)
    if len(df) == 0:
        raise ValueError("No backtest rows available for visualization.")
    for required in ("week_start", "nav", "weekly_return"):
        if required not in df.columns:
            raise ValueError(f"Backtest results must contain {required}.")
    return df.sort("week_start")


def _compute_drawdown(nav_values: list[float]) -> list[float]:
    peak = None
    drawdowns: list[float] = []
    for nav in nav_values:
        peak = nav if peak is None else max(peak, nav)
        drawdowns.append((nav / peak) - 1.0 if peak and peak > 0 else 0.0)
    return drawdowns


def _collect_allocation_rows(results_df: pl.DataFrame) -> tuple[list[str], dict[str, list[float]]]:
    if "holdings" not in results_df.columns:
        return [], {}

    weeks: list[str] = []
    sector_values: dict[str, list[float]] = {}
    rows = results_df.select(["week_start", "holdings"]).to_dicts()
    all_sectors: list[str] = []
    parsed_rows: list[tuple[str, dict[str, float]]] = []
    for row in rows:
        week = str(row.get("week_start", ""))
        holdings = _load_json_cell(row.get("holdings"), {})
        if not isinstance(holdings, dict):
            holdings = {}
        normalized = {str(k): float(v or 0.0) for k, v in holdings.items()}
        parsed_rows.append((week, normalized))
        for sector in normalized:
            if sector not in all_sectors:
                all_sectors.append(sector)

    for week, holdings in parsed_rows:
        weeks.append(week)
        for sector in all_sectors:
            sector_values.setdefault(sector, []).append(float(holdings.get(sector, 0.0)))
    return weeks, sector_values


def _collect_sector_metric_rows(results_df: pl.DataFrame, column: str) -> tuple[list[str], dict[str, list[float]]]:
    if column not in results_df.columns:
        return [], {}

    weeks: list[str] = []
    sector_values: dict[str, list[float]] = {}
    rows = results_df.select(["week_start", column]).to_dicts()
    all_sectors: list[str] = []
    parsed_rows: list[tuple[str, dict[str, float]]] = []
    for row in rows:
        week = str(row.get("week_start", ""))
        payload = _load_json_cell(row.get(column), {})
        if not isinstance(payload, dict):
            payload = {}
        normalized = {str(k): float(v or 0.0) for k, v in payload.items()}
        parsed_rows.append((week, normalized))
        for sector in normalized:
            if sector not in all_sectors:
                all_sectors.append(sector)

    for week, payload in parsed_rows:
        weeks.append(week)
        for sector in all_sectors:
            sector_values.setdefault(sector, []).append(float(payload.get(sector, 0.0)))
    return weeks, sector_values


def _collect_weight_rows(results_df: pl.DataFrame) -> tuple[list[str], list[float], list[float]]:
    weeks = [str(v) for v in results_df["week_start"].to_list()]
    if "invested_weight" in results_df.columns and "cash_weight" in results_df.columns:
        invested = [float(v or 0.0) for v in results_df["invested_weight"].to_list()]
        cash = [float(v or 0.0) for v in results_df["cash_weight"].to_list()]
        return weeks, invested, cash

    allocation_weeks, sector_values = _collect_allocation_rows(results_df)
    if not allocation_weeks or not sector_values:
        return [], [], []
    invested = [sum(values[idx] for values in sector_values.values()) for idx in range(len(allocation_weeks))]
    cash = [1.0 - value for value in invested]
    return allocation_weeks, invested, cash


def _plotly_modules():
    import plotly.graph_objects as go
    import plotly.io as pio

    return go, pio


def _figure_layout(fig, *, title: str, yaxis_title: str = ""):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "color": "#0f172a"},
        },
        template="plotly_white",
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, ui-sans-serif, system-ui, -apple-system", "color": "#111827"},
        margin={"l": 56, "r": 28, "t": 72, "b": 64},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    fig.update_xaxes(showgrid=False, tickangle=-35)
    fig.update_yaxes(title=yaxis_title, gridcolor="#e5e7eb", zerolinecolor="#cbd5e1")
    return fig


def _write_chart(fig, path: Path, *, include_plotlyjs: bool = True) -> Path:
    _, pio = _plotly_modules()
    path.write_text(
        pio.to_html(
            fig,
            full_html=True,
            include_plotlyjs=include_plotlyjs,
            config={"displaylogo": False, "responsive": True},
        ),
        encoding="utf-8",
    )
    return path


def _chart_slug(title: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in title)
    return "_".join(part for part in slug.split("_") if part)


def _matplotlib_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


def _format_image_xaxis(ax, labels: list[str]) -> None:
    if not labels:
        return
    stride = max(1, len(labels) // 8)
    ticks = list(range(0, len(labels), stride))
    if ticks[-1] != len(labels) - 1:
        ticks.append(len(labels) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[idx] for idx in ticks], rotation=35, ha="right")


def _save_image(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    from matplotlib import pyplot as plt

    plt.close(fig)
    return path


def _write_equity_curve_image(results_df: pl.DataFrame, path: Path) -> Path:
    plt = _matplotlib_pyplot()
    from matplotlib.ticker import PercentFormatter

    weeks = [str(v) for v in results_df["week_start"].to_list()]
    nav = [float(v) for v in results_df["nav"].to_list()]
    initial = float(results_df["initial_capital"][0]) if "initial_capital" in results_df.columns else nav[0]
    cumulative_return = [(value / initial) - 1.0 if initial > 0 else 0.0 for value in nav]
    x_values = list(range(len(weeks)))

    fig, ax_nav = plt.subplots(figsize=(12, 6))
    ax_return = ax_nav.twinx()
    ax_nav.plot(
        x_values,
        nav,
        color="#0f766e",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="NAV / Total Value",
    )
    ax_return.plot(
        x_values,
        cumulative_return,
        color="#2563eb",
        linewidth=2.0,
        linestyle="--",
        label="Cumulative Return",
    )
    ax_nav.set_title("NAV / Total Value", loc="left", fontsize=16, fontweight="bold")
    ax_nav.set_ylabel("NAV / Total Value")
    ax_return.set_ylabel("Cumulative Return")
    ax_return.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_nav.grid(True, axis="y", color="#e5e7eb")
    _format_image_xaxis(ax_nav, weeks)
    lines = [*ax_nav.get_lines(), *ax_return.get_lines()]
    ax_nav.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    return _save_image(fig, path)


def _write_weekly_returns_image(results_df: pl.DataFrame, path: Path) -> Path:
    plt = _matplotlib_pyplot()
    from matplotlib.ticker import PercentFormatter

    weeks = [str(v) for v in results_df["week_start"].to_list()]
    returns = [float(v) for v in results_df["weekly_return"].to_list()]
    colors = [_UP_COLOR if value >= 0 else _DOWN_COLOR for value in returns]
    x_values = list(range(len(weeks)))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x_values, returns, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_title("Weekly Return Distribution", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Weekly Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", color="#e5e7eb")
    _format_image_xaxis(ax, weeks)
    fig.tight_layout()
    return _save_image(fig, path)


def _write_drawdown_image(results_df: pl.DataFrame, path: Path) -> Path:
    plt = _matplotlib_pyplot()
    from matplotlib.ticker import PercentFormatter

    weeks = [str(v) for v in results_df["week_start"].to_list()]
    nav = [float(v) for v in results_df["nav"].to_list()]
    drawdowns = _compute_drawdown(nav)
    x_values = list(range(len(weeks)))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.fill_between(x_values, drawdowns, 0, color=_DOWN_COLOR, alpha=0.22)
    ax.plot(x_values, drawdowns, color=_DOWN_COLOR, linewidth=2)
    ax.set_title("Drawdown Profile", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", color="#e5e7eb")
    _format_image_xaxis(ax, weeks)
    fig.tight_layout()
    return _save_image(fig, path)


def _write_allocation_image(results_df: pl.DataFrame, path: Path) -> Path | None:
    weeks, sector_values = _collect_allocation_rows(results_df)
    if not weeks or not sector_values:
        return None

    plt = _matplotlib_pyplot()
    from matplotlib.ticker import PercentFormatter

    x_values = list(range(len(weeks)))
    labels = list(sector_values.keys())
    values = [sector_values[label] for label in labels]
    colors = [_PALETTE[idx % len(_PALETTE)] for idx in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(x_values, values, labels=labels, colors=colors, alpha=0.9)
    ax.set_title("Portfolio Allocation Drift", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", color="#e5e7eb")
    _format_image_xaxis(ax, weeks)
    ax.legend(loc="upper left", ncols=min(3, len(labels)))
    fig.tight_layout()
    return _save_image(fig, path)


def _write_cash_invested_image(results_df: pl.DataFrame, path: Path) -> Path | None:
    weeks, invested, cash = _collect_weight_rows(results_df)
    if not weeks:
        return None

    plt = _matplotlib_pyplot()
    from matplotlib.ticker import PercentFormatter

    x_values = list(range(len(weeks)))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.stackplot(
        x_values,
        [invested, cash],
        labels=["Invested Weight", "Cash Weight"],
        colors=["#0f766e", "#94a3b8"],
        alpha=0.85,
    )
    ax.plot(x_values, invested, color="#064e3b", linewidth=2)
    ax.set_title("Cash vs Invested Weight", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", color="#e5e7eb")
    _format_image_xaxis(ax, weeks)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return _save_image(fig, path)


def _write_sector_contribution_image(results_df: pl.DataFrame, path: Path) -> Path | None:
    weeks, sector_values = _collect_sector_metric_rows(results_df, "meta_sector_contributions")
    if not weeks or not sector_values:
        return None

    plt = _matplotlib_pyplot()
    from matplotlib.ticker import PercentFormatter

    x_values = list(range(len(weeks)))
    positive_bottom = [0.0] * len(weeks)
    negative_bottom = [0.0] * len(weeks)
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, (sector, values) in enumerate(sector_values.items()):
        positives = [value if value >= 0 else 0.0 for value in values]
        negatives = [value if value < 0 else 0.0 for value in values]
        up_color = _UP_PALETTE[idx % len(_UP_PALETTE)]
        down_color = _DOWN_PALETTE[idx % len(_DOWN_PALETTE)]
        ax.bar(x_values, positives, bottom=positive_bottom, color=up_color, label=f"{sector} +", width=0.82)
        ax.bar(x_values, negatives, bottom=negative_bottom, color=down_color, label=f"{sector} -", width=0.82)
        positive_bottom = [base + value for base, value in zip(positive_bottom, positives, strict=True)]
        negative_bottom = [base + value for base, value in zip(negative_bottom, negatives, strict=True)]

    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_title("Sector Contribution", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Weekly Return Contribution")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", color="#e5e7eb")
    _format_image_xaxis(ax, weeks)
    ax.legend(loc="upper left", ncols=min(3, len(sector_values)))
    fig.tight_layout()
    return _save_image(fig, path)


def _write_sector_return_heatmap_image(results_df: pl.DataFrame, path: Path) -> Path | None:
    weeks, sector_values = _collect_sector_metric_rows(results_df, "meta_sector_returns")
    if not weeks or not sector_values:
        return None

    plt = _matplotlib_pyplot()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import PercentFormatter

    sectors = list(sector_values.keys())
    matrix = [sector_values[sector] for sector in sectors]
    max_abs = max(abs(value) for values in matrix for value in values) or 0.01
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(13, max(5, 0.42 * len(sectors) + 2.8)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", norm=norm)
    ax.set_title("Sector Return Heatmap", loc="left", fontsize=16, fontweight="bold")
    _format_image_xaxis(ax, weeks)
    ax.set_yticks(list(range(len(sectors))))
    ax.set_yticklabels(sectors)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    return _save_image(fig, path)


def _write_images(results_df: pl.DataFrame, output_dir: Path) -> list[Path]:
    image_paths = [
        _write_equity_curve_image(results_df, output_dir / "equity_curve.png"),
        _write_weekly_returns_image(results_df, output_dir / "weekly_return_distribution.png"),
        _write_drawdown_image(results_df, output_dir / "drawdown_profile.png"),
    ]
    for optional_path in (
        _write_cash_invested_image(results_df, output_dir / "cash_vs_invested_weight.png"),
        _write_allocation_image(results_df, output_dir / "portfolio_allocation_drift.png"),
        _write_sector_contribution_image(results_df, output_dir / "sector_contribution.png"),
        _write_sector_return_heatmap_image(results_df, output_dir / "sector_return_heatmap.png"),
    ):
        if optional_path is not None:
            image_paths.append(optional_path)
    return image_paths


def _equity_curve_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks = [str(v) for v in results_df["week_start"].to_list()]
    nav = [float(v) for v in results_df["nav"].to_list()]
    initial = float(results_df["initial_capital"][0]) if "initial_capital" in results_df.columns else nav[0]
    cumulative_return = [(value / initial) - 1.0 if initial > 0 else 0.0 for value in nav]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=nav,
            mode="lines+markers",
            name="NAV / Total Value",
            line={"color": "#0f766e", "width": 3},
            marker={"size": 6, "color": "#14b8a6"},
            hovertemplate="Week=%{x}<br>NAV / Total Value=%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=cumulative_return,
            mode="lines",
            name="Cumulative Return",
            yaxis="y2",
            line={"color": "#2563eb", "width": 2, "dash": "dot"},
            hovertemplate="Week=%{x}<br>Cumulative Return=%{y:.2%}<extra></extra>",
        )
    )
    _figure_layout(fig, title="NAV / Total Value", yaxis_title="NAV / Total Value")
    fig.update_layout(
        yaxis2={
            "title": "Cumulative Return",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "showgrid": False,
        }
    )
    return fig


def _weekly_returns_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks = [str(v) for v in results_df["week_start"].to_list()]
    returns = [float(v) for v in results_df["weekly_return"].to_list()]
    colors = [_UP_COLOR if value >= 0 else _DOWN_COLOR for value in returns]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=returns,
            name="Weekly Return",
            marker={"color": colors, "line": {"color": "#ffffff", "width": 0.8}},
            hovertemplate="Week=%{x}<br>Return=%{y:.2%}<extra></extra>",
        )
    )
    _figure_layout(fig, title="Weekly Return Distribution", yaxis_title="Weekly Return")
    fig.update_yaxes(tickformat=".1%")
    fig.add_hline(y=0, line_width=1, line_color="#475569")
    return fig


def _drawdown_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks = [str(v) for v in results_df["week_start"].to_list()]
    nav = [float(v) for v in results_df["nav"].to_list()]
    drawdowns = _compute_drawdown(nav)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=drawdowns,
            name="Drawdown",
            fill="tozeroy",
            mode="lines",
            line={"color": _DOWN_COLOR, "width": 2},
            fillcolor="rgba(22, 163, 74, 0.22)",
            hovertemplate="Week=%{x}<br>Drawdown=%{y:.2%}<extra></extra>",
        )
    )
    _figure_layout(fig, title="Drawdown Profile", yaxis_title="Drawdown")
    fig.update_yaxes(tickformat=".1%")
    return fig


def _allocation_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks, sector_values = _collect_allocation_rows(results_df)
    if not weeks or not sector_values:
        return None

    fig = go.Figure()
    for idx, (sector, values) in enumerate(sector_values.items()):
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=values,
                name=sector,
                mode="lines",
                stackgroup="one",
                line={"width": 0.8, "color": _PALETTE[idx % len(_PALETTE)]},
                hovertemplate=f"{html.escape(sector)}=%{{y:.1%}}<extra></extra>",
            )
        )
    _figure_layout(fig, title="Portfolio Allocation Drift", yaxis_title="Weight")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def _cash_invested_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks, invested, cash = _collect_weight_rows(results_df)
    if not weeks:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=invested,
            name="Invested Weight",
            mode="lines",
            stackgroup="one",
            line={"width": 1.6, "color": "#0f766e"},
            hovertemplate="Week=%{x}<br>Invested=%{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=cash,
            name="Cash Weight",
            mode="lines",
            stackgroup="one",
            line={"width": 1.6, "color": "#94a3b8"},
            hovertemplate="Week=%{x}<br>Cash=%{y:.1%}<extra></extra>",
        )
    )
    _figure_layout(fig, title="Cash vs Invested Weight", yaxis_title="Portfolio Weight")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def _sector_contribution_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks, sector_values = _collect_sector_metric_rows(results_df, "meta_sector_contributions")
    if not weeks or not sector_values:
        return None

    fig = go.Figure()
    for idx, (sector, values) in enumerate(sector_values.items()):
        positives = [value if value >= 0 else 0.0 for value in values]
        negatives = [value if value < 0 else 0.0 for value in values]
        fig.add_trace(
            go.Bar(
                x=weeks,
                y=positives,
                name=f"{sector} +",
                marker={"color": _UP_PALETTE[idx % len(_UP_PALETTE)]},
                hovertemplate=f"Week=%{{x}}<br>{html.escape(sector)} contribution=%{{y:.2%}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=weeks,
                y=negatives,
                name=f"{sector} -",
                marker={"color": _DOWN_PALETTE[idx % len(_DOWN_PALETTE)]},
                hovertemplate=f"Week=%{{x}}<br>{html.escape(sector)} contribution=%{{y:.2%}}<extra></extra>",
            )
        )
    _figure_layout(fig, title="Sector Contribution", yaxis_title="Weekly Return Contribution")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(tickformat=".1%")
    fig.add_hline(y=0, line_width=1, line_color="#475569")
    return fig


def _sector_return_heatmap_figure(results_df: pl.DataFrame):
    go, _ = _plotly_modules()
    weeks, sector_values = _collect_sector_metric_rows(results_df, "meta_sector_returns")
    if not weeks or not sector_values:
        return None

    sectors = list(sector_values.keys())
    z_values = [sector_values[sector] for sector in sectors]
    max_abs = max(abs(value) for values in z_values for value in values) or 0.01
    fig = go.Figure(
        data=go.Heatmap(
            x=weeks,
            y=sectors,
            z=z_values,
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            colorscale="RdYlGn_r",
            colorbar={"title": "Return", "tickformat": ".1%"},
            hovertemplate="Week=%{x}<br>Sector=%{y}<br>Return=%{z:.2%}<extra></extra>",
        )
    )
    _figure_layout(fig, title="Sector Return Heatmap", yaxis_title="")
    return fig


def _summary_payload(
    *,
    run_id: str,
    results_path: Path,
    metrics_path: Path | None,
    results_df: pl.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    latest = results_df.tail(1).row(0, named=True)
    best = results_df.sort("weekly_return", descending=True).head(1).row(0, named=True)
    worst = results_df.sort("weekly_return").head(1).row(0, named=True)
    return {
        "run_id": run_id,
        "results_path": str(results_path),
        "metrics_path": str(metrics_path) if metrics_path is not None else "",
        "rows": len(results_df),
        "start_week": str(results_df["week_start"][0]),
        "end_week": str(latest.get("week_start", "")),
        "final_nav": float(latest.get("nav", 0.0) or 0.0),
        "total_return": float(metrics.get("total_return", 0.0) or 0.0),
        "annual_return": float(metrics.get("annual_return", 0.0) or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0) or 0.0),
        "sortino_ratio": float(metrics.get("sortino_ratio", 0.0) or 0.0),
        "calmar_ratio": float(metrics.get("calmar_ratio", 0.0) or 0.0),
        "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
        "best_week": str(best.get("week_start", "")),
        "best_weekly_return": float(best.get("weekly_return", 0.0) or 0.0),
        "worst_week": str(worst.get("week_start", "")),
        "worst_weekly_return": float(worst.get("weekly_return", 0.0) or 0.0),
    }


def _fmt_percent(value: Any) -> str:
    return f"{float(value or 0.0):.2%}"


def _fmt_number(value: Any) -> str:
    return f"{float(value or 0.0):,.2f}"


def _metric_cards(summary: dict[str, Any]) -> str:
    items = [
        ("Final NAV", _fmt_number(summary.get("final_nav")), "terminal net asset value"),
        ("Total Return", _fmt_percent(summary.get("total_return")), "whole-period performance"),
        ("Annual Return", _fmt_percent(summary.get("annual_return")), "annualized return"),
        ("Max Drawdown", _fmt_percent(summary.get("max_drawdown")), "worst peak-to-trough loss"),
        ("Sharpe", _fmt_number(summary.get("sharpe_ratio")), "risk-adjusted return"),
        ("Win Rate", _fmt_percent(summary.get("win_rate")), "positive weeks"),
    ]
    return "\n".join(
        f"""
        <div class="metric-card">
          <span>{html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
          <small>{html.escape(subtitle)}</small>
        </div>
        """
        for label, value, subtitle in items
    )


def _figure_fragment(fig, *, include_plotlyjs: bool) -> str:
    _, pio = _plotly_modules()
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displaylogo": False, "responsive": True},
    )


def _write_dashboard(summary: dict[str, Any], figures: list[tuple[str, Any]], output_dir: Path) -> Path:
    fragments = []
    for idx, (title, fig) in enumerate(figures):
        fragments.append(
            f"""
            <section class="chart-panel">
              <div class="chart-title">{html.escape(title)}</div>
              {_figure_fragment(fig, include_plotlyjs=(idx == 0))}
            </section>
            """
        )

    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Dashboard - {html.escape(str(summary.get("run_id", "")))}</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #64748b;
      --panel: rgba(255, 255, 255, 0.86);
      --line: rgba(148, 163, 184, 0.25);
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 8% 4%, rgba(20, 184, 166, 0.22), transparent 28rem),
        radial-gradient(circle at 92% 8%, rgba(37, 99, 235, 0.18), transparent 26rem),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }}
    .shell {{ max-width: 1320px; margin: 0 auto; padding: 40px 28px 56px; }}
    .hero {{
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.70);
      box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
      border-radius: 28px;
      padding: 28px;
      backdrop-filter: blur(16px);
    }}
    .eyebrow {{ color: var(--accent); font-size: 13px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 4px; font-size: clamp(32px, 4vw, 56px); line-height: 1.02; letter-spacing: -0.05em; }}
    .meta {{ color: var(--muted); font-size: 15px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-top: 24px; }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      background: var(--panel);
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }}
    .metric-card strong {{ display: block; margin-top: 8px; font-size: 26px; letter-spacing: -0.04em; }}
    .metric-card small {{ display: block; margin-top: 4px; color: var(--muted); }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 28px; }}
    .chart-panel {{
      border: 1px solid var(--line);
      border-radius: 24px;
      overflow: hidden;
      background: var(--panel);
      box-shadow: 0 18px 52px rgba(15, 23, 42, 0.08);
    }}
    .chart-title {{ padding: 18px 22px 0; font-weight: 800; color: #1e293b; }}
    @media (max-width: 760px) {{ .shell {{ padding: 22px 14px 36px; }} .hero {{ padding: 20px; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <div class="eyebrow">News2ETF Agent Backtest</div>
      <h1>Interactive Performance Dashboard</h1>
      <div class="meta">
        run_id={html.escape(str(summary.get("run_id", "")))} ·
        {html.escape(str(summary.get("start_week", "")))} → {html.escape(str(summary.get("end_week", "")))} ·
        rows={html.escape(str(summary.get("rows", "")))}
      </div>
      <div class="metric-grid">{_metric_cards(summary)}</div>
    </header>
    <div class="chart-grid">
      {"".join(fragments)}
    </div>
  </main>
</body>
</html>
"""
    path = output_dir / "report.html"
    path.write_text(content, encoding="utf-8")
    return path


def visualize_backtest(
    *,
    results_path: str | Path,
    metrics_path: str | Path | None = None,
    output_dir: str | Path,
    run_id: str | None = None,
) -> BacktestVisualizationResult:
    results_path = Path(results_path)
    metrics_path = Path(metrics_path) if metrics_path is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"Backtest results not found: {results_path}")
    if metrics_path is not None and not metrics_path.exists():
        logger.warning("Backtest metrics not found, recalculating from results: {}", metrics_path)

    raw_results_df = pl.read_parquet(results_path)
    resolved_run_id = run_id
    if resolved_run_id is None and "run_id" in raw_results_df.columns and len(raw_results_df) > 0:
        run_ids = [str(value) for value in raw_results_df["run_id"].drop_nulls().unique().to_list()]
        resolved_run_id = run_ids[0] if len(run_ids) == 1 else "backtest"
    resolved_run_id = resolved_run_id or "backtest"

    results_df = _prepare_results(raw_results_df, run_id=run_id)
    metrics = _read_latest_metrics(metrics_path, results_df)
    summary = _summary_payload(
        run_id=resolved_run_id,
        results_path=results_path,
        metrics_path=metrics_path,
        results_df=results_df,
        metrics=metrics,
    )

    figures: list[tuple[str, Any]] = [
        ("NAV / Total Value", _equity_curve_figure(results_df)),
        ("Weekly Return Distribution", _weekly_returns_figure(results_df)),
        ("Drawdown Profile", _drawdown_figure(results_df)),
    ]
    for title, fig in (
        ("Cash vs Invested Weight", _cash_invested_figure(results_df)),
        ("Portfolio Allocation Drift", _allocation_figure(results_df)),
        ("Sector Contribution", _sector_contribution_figure(results_df)),
        ("Sector Return Heatmap", _sector_return_heatmap_figure(results_df)),
    ):
        if fig is not None:
            figures.append((title, fig))

    chart_paths = [_write_chart(fig, output_dir / f"{_chart_slug(title)}.html") for title, fig in figures]
    image_paths = _write_images(results_df, output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_dashboard(summary, figures, output_dir)

    logger.info("Backtest visualization saved to {}", output_dir)
    return BacktestVisualizationResult(
        run_id=resolved_run_id,
        output_dir=output_dir,
        summary_path=summary_path,
        report_path=report_path,
        chart_paths=chart_paths,
        image_paths=image_paths,
    )
