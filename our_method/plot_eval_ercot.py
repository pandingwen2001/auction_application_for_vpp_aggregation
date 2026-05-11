#!/usr/bin/env python3
"""Generate diagnostic plots from eval_ercot.csv."""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run dir containing eval_ercot.csv")
    parser.add_argument("--checkpoint", default="model_best.pth",
                        help="Which checkpoint to plot in detail")
    args = parser.parse_args()

    if args.run is None:
        runs_dir = os.path.join(_THIS_DIR, "..", "runs")
        candidates = [os.path.join(runs_dir, d)
                      for d in os.listdir(runs_dir)
                      if os.path.isdir(os.path.join(runs_dir, d))]
        run_dir = max(candidates, key=os.path.getmtime)
    else:
        run_dir = os.path.abspath(args.run)
    csv_path = os.path.join(run_dir, "eval_ercot.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}, rows={len(df)}")

    # Focus on one checkpoint for per-scenario plots
    sub = df[df["checkpoint"] == args.checkpoint].copy()
    sub_pre = sub[sub["name"] == "pre"].sort_values("scenario_idx").reset_index(drop=True)
    sub_post = sub[sub["name"] == "post"].sort_values("scenario_idx").reset_index(drop=True)
    sub_soc = sub[sub["name"] == "social_opt"].sort_values("scenario_idx").reset_index(drop=True)

    idx = sub_pre["scenario_idx"].values
    dates = sub_pre["scenario_date"].values
    bar_w = 0.28

    fig, ax = plt.subplots(3, 2, figsize=(16, 12))
    ax = ax.flatten()

    # Plot 1: Procurement cost per scenario (3 bars: pre / post / social-opt)
    ax[0].bar(idx - bar_w, sub_pre["procurement_cost"], bar_w, label="pre", color="#5B9BD5")
    ax[0].bar(idx,         sub_post["procurement_cost"], bar_w, label="post", color="#ED7D31")
    ax[0].bar(idx + bar_w, sub_soc["social_cost_true"], bar_w, label="social-opt", color="#70AD47")
    ax[0].set_title(f"Procurement cost per scenario  ({args.checkpoint})")
    ax[0].set_xlabel("scenario_idx")
    ax[0].set_ylabel("cost ($)")
    ax[0].legend()
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels([d[5:] for d in dates], rotation=45, fontsize=7)

    # Plot 2: Information rent per scenario (post-stage only)
    ax[1].bar(idx, sub_post["info_rent"], color="#ED7D31")
    ax[1].set_title(f"Info rent (post) per scenario  ({args.checkpoint})")
    ax[1].set_xlabel("scenario_idx")
    ax[1].set_ylabel("info_rent ($)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels([d[5:] for d in dates], rotation=45, fontsize=7)

    # Plot 3: rent ratio (info_rent / procurement_cost)
    rent_ratio = sub_post["info_rent"] / sub_post["procurement_cost"]
    ax[2].bar(idx, rent_ratio * 100, color="#7030A0")
    ax[2].set_title("Rent ratio = info_rent / procurement_cost")
    ax[2].set_xlabel("scenario_idx")
    ax[2].set_ylabel("rent ratio (%)")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xticks(idx)
    ax[2].set_xticklabels([d[5:] for d in dates], rotation=45, fontsize=7)

    # Plot 4: MT floor gap pre vs post
    if "mt_floor_gap_mwh" in sub_pre.columns:
        ax[3].bar(idx - bar_w/2, sub_pre["mt_floor_gap_mwh"], bar_w, label="pre", color="#5B9BD5")
        ax[3].bar(idx + bar_w/2, sub_post["mt_floor_gap_mwh"], bar_w, label="post", color="#ED7D31")
        ax[3].set_title("MT floor gap (MWh) — should drop to 0 after postprocess")
        ax[3].set_xlabel("scenario_idx")
        ax[3].set_ylabel("MT gap (MWh)")
        ax[3].legend()
        ax[3].set_xticks(idx)
        ax[3].set_xticklabels([d[5:] for d in dates], rotation=45, fontsize=7)

    # Plot 5: positive adjustment (MWh) — postprocess uplift volume
    if "positive_adjustment_mwh" in sub_post.columns:
        ax[4].bar(idx, sub_post["positive_adjustment_mwh"], color="#FFC000")
        ax[4].set_title("Post-process positive adjustment (MWh)")
        ax[4].set_xlabel("scenario_idx")
        ax[4].set_ylabel("uplift volume (MWh)")
        ax[4].set_xticks(idx)
        ax[4].set_xticklabels([d[5:] for d in dates], rotation=45, fontsize=7)

    # Plot 6: comparison across checkpoints (aggregate)
    agg = df.groupby(["checkpoint", "name"]).agg(
        proc=("procurement_cost", "mean"),
        social=("social_cost_true", "mean"),
        rent=("info_rent", "mean"),
        mt_gap=("mt_floor_gap_mwh", "mean"),
    ).reset_index()
    post_only = agg[agg["name"] == "post"]
    soc_only = agg[agg["name"] == "social_opt"]
    ckpts = post_only["checkpoint"].values
    x = np.arange(len(ckpts))
    ax[5].bar(x - bar_w, post_only["proc"].values, bar_w, label="proc (post)", color="#ED7D31")
    ax[5].bar(x, soc_only["social"].values, bar_w, label="social-opt", color="#70AD47")
    ax[5].bar(x + bar_w, post_only["rent"].values, bar_w, label="info_rent", color="#7030A0")
    ax[5].set_title("Aggregate (mean over 24 scenarios) per checkpoint")
    ax[5].set_xticks(x)
    ax[5].set_xticklabels(ckpts, rotation=15, fontsize=8)
    ax[5].set_ylabel("$")
    ax[5].legend()

    plt.tight_layout()
    out_path = os.path.join(run_dir, "eval_ercot_diagnostics.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")

    # Per-checkpoint summary table
    print("\n=== Aggregate ranking (lower proc cost / lower rent is better) ===")
    rank = post_only.set_index("checkpoint")[["proc", "social", "rent"]].copy()
    rank["proc_premium_vs_social"] = (rank["proc"] - rank["social"]) / rank["social"] * 100
    rank["rent_ratio_%"] = rank["rent"] / rank["proc"] * 100
    print(rank.sort_values("proc").to_string(float_format=lambda x: f"{x:8.2f}"))


if __name__ == "__main__":
    main()
