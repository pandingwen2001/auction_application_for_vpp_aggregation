#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a publication-oriented main figure from Experiment 1 output."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFERRED_ORDER = [
    "constrained_social_opt",
    "learned_peer_model_best.pth:post",
    "vcg_disaggregation",
    "shapley_value_disaggregation",
    "nucleolus_disaggregation",
    "dlmp_settlement",
    "bid_dependent_opf_uniform_da",
    "bid_dependent_opf_pay_as_bid",
]

LABELS = {
    "constrained_social_opt": "Social\nopt.",
    "learned_peer_model_best.pth:post": "Ours",
    "vcg_disaggregation": "VCG",
    "shapley_value_disaggregation": "Shap.",
    "nucleolus_disaggregation": "Nucl.",
    "dlmp_settlement": "DLMP",
    "bid_dependent_opf_uniform_da": "Uni.\nDA",
    "bid_dependent_opf_pay_as_bid": "PAB",
}

COLORS = {
    "operation": "#5B7C99",
    "rent": "#D08C60",
    "shortfall": "#C55353",
    "ours": "#2E7D6B",
    "baseline": "#8F99A8",
    "line": "#1F2933",
    "renewable": "#739E82",
    "grid": "#D9DEE7",
}


def _float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def _is_learned_post(row):
    return (
        row.get("category") == "learned_posted_price"
        and row.get("stage") == "post"
    )


def _row_order(rows):
    by_name = {row["name"]: row for row in rows}
    ordered = []
    used = set()

    for name in PREFERRED_ORDER:
        if name in by_name:
            ordered.append(by_name[name])
            used.add(name)

    if not any(_is_learned_post(row) for row in ordered):
        learned = sorted(
            (row for row in rows if _is_learned_post(row)),
            key=lambda r: r.get("name", ""),
        )
        if learned:
            insert_at = 1 if ordered else 0
            ordered.insert(insert_at, learned[0])
            used.add(learned[0]["name"])

    remaining = [
        row for row in rows
        if row.get("name") not in used
        and row.get("category") != "fixed_price"
    ]
    ordered.extend(sorted(remaining, key=lambda r: r.get("name", "")))
    return ordered


def _label(row):
    name = row["name"]
    if name in LABELS:
        return LABELS[name]
    if _is_learned_post(row):
        return "Ours"
    return name.replace("_", "\n")


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return _row_order(rows)


def _highlight_colors(names):
    return [
        COLORS["ours"] if name.startswith("learned_peer_")
        else COLORS["operation"]
        for name in names
    ]


def make_figure(rows, out_pdf, out_png):
    names = [row["name"] for row in rows]
    labels = [_label(row) for row in rows]
    x = np.arange(len(rows), dtype=float)

    operation = np.array([_float(row, "operation_cost") for row in rows])
    rent = np.array([_float(row, "info_rent_cost") for row in rows])
    total = np.array([_float(row, "total_procurement_cost") for row in rows])
    op_gap = np.array([_float(row, "operation_cost_gap_pct") for row in rows])
    dispatch_gap = np.array([_float(row, "dispatch_l1_gap_mwh") for row in rows])
    utility_shortfall = np.array([
        _float(row, "utility_shortfall_cost") for row in rows
    ])
    renewable_share = np.array([
        _float(row, "renewable_share_pct") for row in rows
    ])

    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )

    ax = axes[0]
    op_colors = _highlight_colors(names)
    rent_pos = np.clip(rent, 0.0, None)
    rent_neg = np.clip(rent, None, 0.0)
    ax.bar(x, operation, color=op_colors, label="Operation cost")
    ax.bar(x, rent_pos, bottom=operation, color=COLORS["rent"],
           label="Information rent")
    if np.any(rent_neg < 0):
        ax.bar(x, rent_neg, bottom=operation, color=COLORS["shortfall"],
               label="Payment shortfall")
    ax.plot(x, total, "o", color=COLORS["line"], markersize=4,
            label="Total procurement")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Procurement Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax2 = axes[1]
    ax2.bar(x, op_gap, color=op_colors, label="Operation gap")
    ax2.set_ylabel("Operation cost gap (%)")
    ax2.set_title("Dispatch Efficiency")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax2.set_axisbelow(True)

    ax2b = ax2.twinx()
    ax2b.plot(x, dispatch_gap, color=COLORS["line"], marker="o",
              linewidth=1.4, markersize=4, label="Dispatch gap")
    ax2b.set_ylabel("Dispatch L1 gap (MWh)")
    lines, line_labels = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines_b, line_labels + labels_b,
               frameon=False, fontsize=8, loc="upper left")

    ax3 = axes[2]
    shortfall_colors = [
        COLORS["shortfall"] if value > 1e-9 else COLORS["baseline"]
        for value in utility_shortfall
    ]
    ax3.bar(x, utility_shortfall, color=shortfall_colors,
            label="Utility shortfall")
    ax3.set_ylabel("Max utility shortfall ($)")
    ax3.set_title("Safety and Clean Use")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=35, ha="right")
    ax3.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax3.set_axisbelow(True)

    ax3b = ax3.twinx()
    ax3b.plot(x, renewable_share, color=COLORS["renewable"], marker="o",
              linewidth=1.4, markersize=4, label="Renewable share")
    ax3b.set_ylabel("Renewable share of DER dispatch (%)")
    lines, line_labels = ax3.get_legend_handles_labels()
    lines_b, labels_b = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines_b, line_labels + labels_b,
               frameon=False, fontsize=8, loc="upper left")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
    for axis in (ax2b, ax3b):
        axis.spines["top"].set_visible(False)

    fig.suptitle("ERCOT 2023 Typical-Day Overall Performance (24 Samples/Scenario)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default=os.path.join(
            os.path.dirname(__file__),
            "results_ercot_paper_v2",
            "overall_performance_table.csv",
        ),
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    table = os.path.abspath(args.table)
    out_dir = args.out_dir or os.path.dirname(table)
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(table)
    out_pdf = os.path.join(out_dir, "overall_performance_main_figure.pdf")
    out_png = os.path.join(out_dir, "overall_performance_main_figure.png")
    make_figure(rows, out_pdf, out_png)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
