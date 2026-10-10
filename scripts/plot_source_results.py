#!/usr/bin/env python3
"""Plot frozen source-quality and paired search-time evidence; no experiment is run."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Stabilize PDF metadata across repeated generations where supported.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", str(Path.home()))) / "litobench-matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
QUALITY_PATH = EVIDENCE / "independent-source-20261010" / "run-29f83e4" / "result.json"
EVENTS_DIR = EVIDENCE / "source-events-20261010"
REPORTS = {
    "Uncached v1": EVENTS_DIR / "event_uncached_50e2d8f_benchmark_report.json",
    "Cache-enabled v2": EVENTS_DIR / "event_cached_690275d_benchmark_report.json",
}
OUT = ROOT / "paper" / "figures"

SCOPE_ORDER = [
    ("pooled", "Pooled"),
    ("StdMetal271", "StdMetal"),
    ("StdContact165", "StdContact"),
]
METRICS = [
    ("pvband_rate", "PV-band"),
    ("nominal_error_fraction_of_target", "Nominal mismatch"),
    ("worst_dose_error_fraction_of_target", "Worst-dose mismatch"),
]
COMPARISONS = [
    ("event_candidate_minus_reference", "Frozen reference", "#2369a8", "o", -0.14),
    ("event_candidate_minus_best_known", "Frozen best-known", "#c26719", "s", 0.14),
]
SOURCE_SERIES = [
    ("reference", "Frozen reference", "#2369a8"),
    ("event_candidate", "Event candidate", "#c26719"),
    ("best_known", "Frozen best-known", "#45824e"),
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52606d",
        "axes.labelcolor": "#263238",
        "text.color": "#263238",
        "xtick.color": "#455a64",
        "ytick.color": "#455a64",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_pair(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", metadata={"Software": "plot_source_results.py"})
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "plot_source_results.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"WROTE {png.relative_to(ROOT)} ({png.stat().st_size} bytes)")
    print(f"WROTE {pdf.relative_to(ROOT)} ({pdf.stat().st_size} bytes)")


def forest_data(result: dict) -> list[dict]:
    rows = []
    for scope_key, scope_label in SCOPE_ORDER:
        scope = result["scope_summaries"][scope_key]
        for metric_key, metric_label in METRICS:
            for comparison_key, comparison_label, color, marker, offset in COMPARISONS:
                stats = scope[comparison_key][metric_key]
                estimate = float(stats["equal_group_mean_delta"]) * 100.0
                ci_low, ci_high = [float(v) * 100.0 for v in stats["bootstrap_95_percentile_ci"]]
                rows.append({
                    "scope": scope_label,
                    "metric": metric_label,
                    "comparison": comparison_label,
                    "color": color,
                    "marker": marker,
                    "offset": offset,
                    "estimate_pp": estimate,
                    "ci_low_pp": ci_low,
                    "ci_high_pp": ci_high,
                    "groups": int(stats["paired_group_count"]),
                })
    return rows


def plot_quality_deltas(result: dict) -> list[dict]:
    rows = forest_data(result)
    labels = [f"{scope} · {metric}" for scope, _ in SCOPE_ORDER for _, metric in METRICS]
    y = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    fig.subplots_adjust(left=0.27, right=0.985, top=0.82, bottom=0.16)
    ax.axvline(0, color="#30343b", linewidth=1.15, linestyle=(0, (3, 2)), zorder=0)
    for boundary in (2.5, 5.5):
        ax.axhline(boundary, color="#d7dee4", linewidth=0.8, zorder=0)
    for comparison_key, comparison_label, color, marker, offset in COMPARISONS:
        chosen = [row for row in rows if row["comparison"] == comparison_label]
        for row_idx, row in enumerate(chosen):
            estimate = row["estimate_pp"]
            low = row["ci_low_pp"]
            high = row["ci_high_pp"]
            ax.errorbar(
                estimate, y[row_idx] + offset,
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt=marker, color=color, ecolor=color,
                markersize=5.0, elinewidth=1.15, capsize=2.1, capthick=1.05,
                markeredgecolor="white", markeredgewidth=0.45,
                label=("vs reference" if comparison_label == "Frozen reference" else "vs best-known") if row_idx == 0 else None,
                zorder=3,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.016, 0.006)
    ax.set_xticks([-0.015, -0.010, -0.005, 0.000, 0.005])
    ax.set_xlabel("Candidate − frozen comparator (percentage points; lower is better)")
    fig.suptitle("Fixed-source transfer: group-mean error deltas (95% bootstrap CIs)", x=0.01, y=0.985, ha="left", fontsize=11)
    ax.grid(axis="x", color="#e7edf1", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.985, 0.945))
    fig.text(
        0.01, 0.015,
        "Rates are converted to percentage points (×100); axis retained on a common scale. "
        "Intervals are descriptive equal-cell-group percentile bootstraps (10,000 draws); pooled groups can link families.",
        ha="left", va="top", fontsize=7.7, color="#455a64",
    )
    save_pair(fig, "transfer-quality-deltas")
    return rows


def plot_absolute_errors(result: dict) -> None:
    metric_defs = [
        ("nominal_error_fraction_of_target", "Nominal XOR mismatch", "Nominal mismatch to target-positive pixels"),
        ("worst_dose_error_fraction_of_target", "Worst-dose XOR mismatch", "Worst-dose mismatch to target-positive pixels"),
    ]
    families = [("StdMetal271", "StdMetal"), ("StdContact165", "StdContact")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.25), sharey=True, constrained_layout=True)
    x = np.arange(len(families), dtype=float)
    bar_w = 0.22
    for ax, (metric_key, metric_label, panel_title) in zip(axes, metric_defs):
        for series_idx, (source_key, source_label, color) in enumerate(SOURCE_SERIES):
            values = [
                float(result["scope_summaries"][scope_key][source_key]["group_mean_metrics"][metric_key]) * 100.0
                for scope_key, _ in families
            ]
            xpos = x + (series_idx - 1) * bar_w
            bars = ax.bar(
                xpos, values, width=bar_w * 0.9, color=color, edgecolor="white", linewidth=0.45,
                label=source_label, zorder=3,
            )
            for rect, value in zip(bars, values):
                ax.annotate(
                    f"{value:.1f}%", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.2,
                )
        ax.set_title(panel_title, loc="left", fontsize=9.5)
        ax.set_xticks(x, [label for _, label in families])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#e7edf1", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Error pixels / target-positive pixels (%)")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#d7dee4")
    fig.suptitle("Absolute transfer errors by family and frozen source", x=0.01, ha="left", fontsize=11)
    fig.text(
        0.01, -0.01,
        "Equal means over pinned cell-group components. StdContact remains high for every source: nominal 86.6% and worst-dose 89.9% for the candidate.",
        ha="left", va="top", fontsize=7.7, color="#455a64",
    )
    save_pair(fig, "transfer-absolute-errors")


def timing_pairs(report: dict) -> list[dict]:
    pairs = []
    for item in report["paired_runs"]:
        event = float(item["arms"]["event"]["timings"]["arm_wall_seconds_including_periodic_report_io"])
        grid = float(item["arms"]["grid"]["timings"]["arm_wall_seconds_including_periodic_report_io"])
        pairs.append({
            "seed": int(item["seed"]),
            "repeat": int(item["paired_repeat"]) + 1,
            "event_seconds": event,
            "grid_seconds": grid,
            "grid_over_event": grid / event,
        })
    return pairs


def plot_runtime_ratios() -> dict[str, list[dict]]:
    datasets = {label: timing_pairs(read_json(path)) for label, path in REPORTS.items()}
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    x = np.arange(15, dtype=float)
    series_style = {
        "Uncached v1": ("#2369a8", "o", -0.11),
        "Cache-enabled v2": ("#c26719", "s", 0.11),
    }
    for label, pairs in datasets.items():
        color, marker, jitter = series_style[label]
        xs = x + jitter
        ratios = np.asarray([pair["grid_over_event"] for pair in pairs], dtype=float)
        ax.scatter(xs, ratios, s=33, marker=marker, color=color, edgecolor="white", linewidth=0.55, label=label, zorder=3)
        median = float(np.median(ratios))
        ax.axhline(median, color=color, alpha=0.75, linewidth=1.0, linestyle=(0, (4, 2)))
    labels = [f"{pair['seed']}/{pair['repeat']}" for pair in datasets["Uncached v1"]]
    ax.axhline(1, color="#697780", linewidth=0.9, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(-0.7, 14.7)
    ax.set_ylim(0, 20)
    ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Matched fixed seed / timing repeat")
    ax.set_ylabel("Uniform-grid / event full-arm elapsed-time ratio")
    ax.set_title("Paired quality-time workload ratios (15 matched observations per protocol)", loc="left", pad=10)
    ax.grid(axis="y", color="#e7edf1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    v1 = np.asarray([pair["grid_over_event"] for pair in datasets["Uncached v1"]])
    v2 = np.asarray([pair["grid_over_event"] for pair in datasets["Cache-enabled v2"]])
    ax.text(0.99, 0.97, f"Paired medians: v1 {np.median(v1):.3f}×   ·   v2 {np.median(v2):.3f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#263238")
    fig.text(
        0.01, 0.015,
        "Same fixed seeds, paths, protected incumbents, and per-arm caps. Ratios describe this CPU search workload, not hardware speedup; a shared GPU job and v2 test overlap limit causal timing interpretations.",
        ha="left", va="top", fontsize=7.7, color="#455a64",
    )
    save_pair(fig, "paired-runtime-ratios")
    return datasets


def main() -> None:
    style()
    result = read_json(QUALITY_PATH)
    if result.get("status") != "complete" or result["scoring"].get("completed_source_layout_records") != 1155:
        raise SystemExit("Frozen independent-source result is incomplete; refusing to plot.")
    rows = plot_quality_deltas(result)
    plot_absolute_errors(result)
    timing = plot_runtime_ratios()
    deltas = [row["estimate_pp"] for row in rows]
    print(f"QUALITY_ROWS={len(rows)} estimate_range_pp=[{min(deltas):.6g},{max(deltas):.6g}]")
    for label, pairs in timing.items():
        ratios = [pair["grid_over_event"] for pair in pairs]
        print(f"{label}: pairs={len(pairs)} median_grid_over_event={np.median(ratios):.4f} range=[{min(ratios):.4f},{max(ratios):.4f}]")
    print("INPUTS=docs/evidence/independent-source-20261010/run-29f83e4/result.json; docs/evidence/source-events-20261010/*benchmark_report.json")


if __name__ == "__main__":
    main()