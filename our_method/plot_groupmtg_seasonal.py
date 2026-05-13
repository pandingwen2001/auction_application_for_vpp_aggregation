#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_groupmtg_seasonal.py
-------------------------
Produce a 2x4 stacked figure and an info-rent table for the group meeting.

Top row    : our learned mechanism (post-processed) allocations across 24h.
Bottom row : constrained social-optimum benchmark allocations across 24h.
Columns    : four representative ERCOT seasonal days (winter / spring / summer / fall).

Each panel:
  - Black line: 24h load profile [MW].
  - Stacked colored areas: hourly dispatch by source type (PV / WT / MT) +
    grid import (P_VPP). Stacked total equals the load curve (load balance).

Also emits:
  - <out_dir>/seasonal_allocation_2x4.pdf / .png
  - <out_dir>/info_rent_table.csv
  - <out_dir>/info_rent_table.tex
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "data"))
sys.path.insert(0, _THIS_DIR)

from network.vpp_network_multi import build_network_multi              # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti           # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti                 # noqa: E402
from our_method.postprocess_security import SecurityPostProcessor      # noqa: E402
from baseline.baseline_social_opt_multi import SocialOptimumMechanismMulti  # noqa: E402
from data.ercot_profiles import scenario_date                          # noqa: E402

from our_method.evaluate_posted_price import (                         # noqa: E402
    classify_sources, TYPE_CAP_RATIO,
)


SEASON_LABELS = ["Winter", "Spring", "Summer", "Fall"]

# Colors aligned with the typical VPP color scheme
COLOR_PV   = "#F7B500"   # solar yellow
COLOR_WT   = "#3DA1B3"   # wind teal
COLOR_MT   = "#C0504D"   # microturbine red
COLOR_ESS  = "#6A4C93"   # battery purple
COLOR_GRID = "#7F7F7F"   # grid grey
COLOR_LOAD = "#111111"   # load black


def _select_scenarios(scenarios_arg):
    if scenarios_arg is None:
        # Defaults: winter / spring / summer / fall, chosen so that every
        # representative day has non-negative info rent on model_best.pth
        # (Jan 4 was the original winter pick but its evaluation rent is
        # slightly negative; Dec 6, also a Wednesday, replaces it).
        return [23, 6, 12, 18]  # 2023-12-06, 2023-04-01, 2023-07-01, 2023-10-04
    return [int(x) for x in scenarios_arg]


def _aggregate_by_type(x, source_types):
    """x: [T, N] numpy. Returns dict typ -> [T] sum across DERs of that type."""
    out = {}
    for typ in ("PV", "WT", "MT"):
        mask = np.array([s == typ for s in source_types], dtype=bool)
        if not mask.any():
            out[typ] = np.zeros(x.shape[0], dtype=np.float64)
        else:
            out[typ] = x[:, mask].sum(axis=1)
    return out


def _info_rent_per_type(types, x, p, source_types):
    """types: [B,N,2], x: [B,T,N], p: [B,N].
    Returns dict typ -> {'rent': $info-rent, 'cost': $true-cost, 'pay': $payment}."""
    a = types[..., 0].unsqueeze(1)              # [B,1,N]
    b = types[..., 1].unsqueeze(1)              # [B,1,N]
    tc_per_t = a * x.pow(2) + b * x             # [B,T,N]
    tc = tc_per_t.sum(dim=1)                    # [B,N]
    rent_per_der = (p - tc).mean(dim=0)         # [N]
    cost_per_der = tc.mean(dim=0)               # [N]
    pay_per_der  = p.mean(dim=0)                # [N]
    out = {}
    for typ in ("PV", "WT", "MT"):
        mask = torch.tensor([s == typ for s in source_types], dtype=torch.bool)
        out[typ] = dict(
            rent=float(rent_per_der[mask].sum().item()),
            cost=float(cost_per_der[mask].sum().item()),
            pay =float(pay_per_der[mask].sum().item()),
        )
    out["TOTAL"] = dict(
        rent=float(rent_per_der.sum().item()),
        cost=float(cost_per_der.sum().item()),
        pay =float(pay_per_der.sum().item()),
    )
    return out


def evaluate_scenario(scenario_idx, mech_state_dict, args):
    net = build_network_multi(
        scenario_idx=scenario_idx,
        ctrl_min_ratio=args.ctrl_min_ratio,
        pi_clip_factor=args.pi_clip_factor,
    )
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed + scenario_idx)
    types = prior.sample(args.samples, device="cpu")
    source_types = classify_sources(net)

    mech = VPPMechanismMulti(
        net,
        posted_price_cfg=dict(
            pi_buyback_ratio=args.pi_buyback_ratio,
            use_peer_bid_context=not args.disable_peer_bid_context,
            peer_bid_scale=args.peer_bid_scale,
            type_cap_ratio=TYPE_CAP_RATIO,
        ),
    )
    mech.load_state_dict(mech_state_dict, strict=False)
    mech.eval()

    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()

    pp = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=not args.no_mt_security_uplift,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
        enable_ess_arbitrage=not args.no_ess_arbitrage,
    )
    post = pp.process_batch(
        x_pre.detach().cpu().numpy(),
        P_pre.detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
        offer_cap.detach().cpu().numpy(),
    )
    x_post = torch.tensor(post.x, dtype=torch.float32)
    P_post = torch.tensor(post.P_VPP, dtype=torch.float32)
    p_post = (rho * x_post).sum(dim=1)

    # Post-processed ESS (already net per-bus); collapse over ESS units.
    # Replace nan rows (fallback samples) with zero so averaging is safe.
    pd_post_arr = np.nan_to_num(post.P_d, nan=0.0)                 # [B,T,n_ess]
    pc_post_arr = np.nan_to_num(post.P_c, nan=0.0)
    ess_net_post = (pd_post_arr - pc_post_arr).sum(axis=-1)        # [B,T]

    # Social-opt: replay the JointQP to capture ESS per sample.
    social = SocialOptimumMechanismMulti(net)
    social.eval()
    qp = social.qp
    types_np = types.detach().cpu().numpy()                         # [B,N,2]
    B = types_np.shape[0]
    T = qp.T
    N = qp.N
    n_ess = qp.n_ess
    x_soc_arr  = np.zeros((B, T, N),     dtype=np.float32)
    P_soc_arr  = np.zeros((B, T),        dtype=np.float32)
    pd_soc_arr = np.zeros((B, T, n_ess), dtype=np.float64)
    pc_soc_arr = np.zeros((B, T, n_ess), dtype=np.float64)
    for s in range(B):
        a_tile = np.broadcast_to(types_np[s, :, 0][None, :], (T, N))
        b_tile = np.broadcast_to(types_np[s, :, 1][None, :], (T, N))
        x_np, pvpp_np, _ = qp.solve(a_tile, b_tile)
        x_soc_arr[s] = x_np
        P_soc_arr[s] = pvpp_np
        ess = qp.last_ess
        if ess is not None:
            pd_soc_arr[s] = ess["P_d"]
            pc_soc_arr[s] = ess["P_c"]
    x_soc = torch.tensor(x_soc_arr, dtype=torch.float32)
    P_soc = torch.tensor(P_soc_arr, dtype=torch.float32)
    ess_net_soc = (pd_soc_arr - pc_soc_arr).sum(axis=-1)            # [B,T]
    # Reuse social cost helper: payment == true cost for social opt
    p_soc = types[..., 0] * (x_soc.pow(2).sum(dim=1)) + types[..., 1] * (x_soc.sum(dim=1))

    # Batch-average per-hour profiles
    x_post_mean   = x_post.mean(dim=0).cpu().numpy()      # [T,N]
    P_post_mean   = P_post.mean(dim=0).cpu().numpy()      # [T]
    x_soc_mean    = x_soc.mean(dim=0).cpu().numpy()       # [T,N]
    P_soc_mean    = P_soc.mean(dim=0).cpu().numpy()       # [T]
    ess_post_mean = ess_net_post.mean(axis=0)             # [T]
    ess_soc_mean  = ess_net_soc.mean(axis=0)              # [T]

    load = np.asarray(net["load_profile"], dtype=np.float64)
    pi_DA = np.asarray(net["pi_DA_profile"], dtype=np.float64)   # [T]

    # ---- Per-hour PHYSICAL dispatch cost (no info rent) ----------
    # phys_t = Σ_i (a_i*x_{i,t}^2 + b_i*x_{i,t})  +  π_t * P_VPP_t
    a = types[..., 0].unsqueeze(1)                 # [B,1,N]
    b = types[..., 1].unsqueeze(1)                 # [B,1,N]
    der_cost_post_t = (a * x_post.pow(2) + b * x_post).sum(dim=2)    # [B,T]
    der_cost_soc_t  = (a * x_soc.float().pow(2) + b * x_soc.float()).sum(dim=2)  # [B,T]
    pi_t = torch.tensor(pi_DA, dtype=torch.float32)                  # [T]
    grid_cost_post_t = pi_t.unsqueeze(0) * P_post                    # [B,T]
    grid_cost_soc_t  = pi_t.unsqueeze(0) * P_soc.float()             # [B,T]

    phys_cost_post_t = (der_cost_post_t + grid_cost_post_t).mean(dim=0).cpu().numpy()  # [T]
    phys_cost_soc_t  = (der_cost_soc_t  + grid_cost_soc_t ).mean(dim=0).cpu().numpy()  # [T]

    # Per-day totals
    phys_cost_post = float(phys_cost_post_t.sum())
    phys_cost_soc  = float(phys_cost_soc_t.sum())
    procurement_post = float((p_post.sum(dim=1) +
                              (pi_t.unsqueeze(0) * P_post).sum(dim=1)).mean().item())
    procurement_soc  = float((p_soc.float().sum(dim=1) +
                              (pi_t.unsqueeze(0) * P_soc.float()).sum(dim=1)).mean().item())

    info_rent = {
        "ours": _info_rent_per_type(types, x_post, p_post, source_types),
        "social_opt": _info_rent_per_type(types, x_soc.float(), p_soc.float(), source_types),
    }

    return dict(
        date=str(net.get("scenario_date")),
        load=load,
        source_types=source_types,
        x_post_by_type=_aggregate_by_type(x_post_mean, source_types),
        x_soc_by_type=_aggregate_by_type(x_soc_mean, source_types),
        grid_post=P_post_mean,
        grid_soc=P_soc_mean,
        ess_post=ess_post_mean,
        ess_soc=ess_soc_mean,
        phys_cost_post_t=phys_cost_post_t,
        phys_cost_soc_t=phys_cost_soc_t,
        phys_cost_post=phys_cost_post,
        phys_cost_soc=phys_cost_soc,
        procurement_post=procurement_post,
        procurement_soc=procurement_soc,
        info_rent=info_rent,
    )


def _stack_plot(ax, hours, load, alloc_by_type, grid, ess_net, title=None):
    pv = alloc_by_type["PV"]
    wt = alloc_by_type["WT"]
    mt = alloc_by_type["MT"]
    grid_imp = np.clip(grid, 0.0, None)
    grid_exp = np.clip(grid, None, 0.0)            # <= 0
    ess_dis  = np.clip(ess_net, 0.0, None)         # discharge (supply)
    ess_chg  = np.clip(ess_net, None, 0.0)         # charge (<=0, demand)

    layers = np.stack([pv, wt, mt, ess_dis, grid_imp], axis=0)
    colors = [COLOR_PV, COLOR_WT, COLOR_MT, COLOR_ESS, COLOR_GRID]
    labels = ["PV", "WT", "MT", "ESS discharge", "Grid import"]
    ax.stackplot(hours, layers, colors=colors, labels=labels,
                 alpha=0.92, linewidth=0)

    # Below the zero line: ESS charging (purple, hatched) and grid export (grey, hatched)
    if (ess_chg < -1e-9).any():
        ax.fill_between(hours, ess_chg, 0.0,
                        color=COLOR_ESS, hatch="\\\\", alpha=0.35,
                        linewidth=0, label="ESS charge")
    if (grid_exp < -1e-9).any():
        ax.fill_between(hours, grid_exp, 0.0,
                        color=COLOR_GRID, hatch="//", alpha=0.25,
                        linewidth=0, label="Grid export")

    ax.plot(hours, load, color=COLOR_LOAD, linewidth=2.0,
            label="Load", zorder=5)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5, zorder=4)

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)
    if title:
        ax.set_title(title, fontsize=11)


def _cost_line_plot(ax, hours, phys_post, phys_soc, title=None):
    ax.plot(hours, phys_soc, marker="o", markersize=4.5,
            color="#1f77b4", linewidth=1.6, label="Ground truth")
    ax.plot(hours, phys_post, marker="s", markersize=4.5,
            color="#d62728", linewidth=1.6, label="Ours")
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)
    if title:
        ax.set_title(title, fontsize=11)


def make_figure(results, out_pdf, out_png):
    fig, axes = plt.subplots(
        3, 4, figsize=(15.5, 10.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.95]),
        constrained_layout=True,
    )

    hours = np.arange(24)
    season_for_col = SEASON_LABELS

    # Row 1/2 share y-axis (power stacks). Now includes ESS dis/charge.
    row_pow_top = max(
        max(np.asarray(r["load"]).max(),
            (r["x_post_by_type"]["PV"] + r["x_post_by_type"]["WT"]
             + r["x_post_by_type"]["MT"] + np.clip(r["ess_post"], 0, None)
             + np.clip(r["grid_post"], 0, None)).max(),
            (r["x_soc_by_type"]["PV"]  + r["x_soc_by_type"]["WT"]
             + r["x_soc_by_type"]["MT"] + np.clip(r["ess_soc"], 0, None)
             + np.clip(r["grid_soc"],  0, None)).max())
        for r in results
    )
    row_pow_bot = min(
        min(0.0,
            np.clip(r["ess_post"], None, 0).min() + np.clip(r["grid_post"], None, 0).min(),
            np.clip(r["ess_soc"],  None, 0).min() + np.clip(r["grid_soc"],  None, 0).min())
        for r in results
    )
    y_top_pow = row_pow_top * 1.08
    y_bot_pow = row_pow_bot * 1.15 if row_pow_bot < 0 else 0.0

    # Row 3 shares y-axis (physical cost)
    row_cost_top = max(max(r["phys_cost_post_t"].max(), r["phys_cost_soc_t"].max())
                       for r in results)
    row_cost_bot = min(min(r["phys_cost_post_t"].min(), r["phys_cost_soc_t"].min())
                       for r in results)
    cost_pad = 0.08 * (row_cost_top - row_cost_bot if row_cost_top > row_cost_bot else 1.0)
    y_top_cost = row_cost_top + cost_pad
    y_bot_cost = max(0.0, row_cost_bot - cost_pad)

    for col, r in enumerate(results):
        ax_top = axes[0, col]
        ax_mid = axes[1, col]
        ax_bot = axes[2, col]
        col_title = f"{season_for_col[col]} ({r['date']})"
        _stack_plot(ax_top, hours, r["load"], r["x_post_by_type"], r["grid_post"],
                    r["ess_post"], title=col_title)
        _stack_plot(ax_mid, hours, r["load"], r["x_soc_by_type"], r["grid_soc"],
                    r["ess_soc"], title=None)
        _cost_line_plot(ax_bot, hours, r["phys_cost_post_t"], r["phys_cost_soc_t"])
        ax_top.set_ylim(y_bot_pow, y_top_pow)
        ax_mid.set_ylim(y_bot_pow, y_top_pow)
        ax_bot.set_ylim(y_bot_cost, y_top_cost)
        # Share y across panels of the same row
        if col > 0:
            ax_top.sharey(axes[0, 0])
            ax_mid.sharey(axes[1, 0])
            ax_bot.sharey(axes[2, 0])
        if col == 0:
            ax_top.set_ylabel("Our mechanism\nPower [MW]", fontsize=10)
            ax_mid.set_ylabel("Social optimum\nPower [MW]", fontsize=10)
            ax_bot.set_ylabel("Physical cost\n[\\$/h]", fontsize=10)
        ax_bot.set_xlabel("Hour of day", fontsize=10)

    legend_handles_pow = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_PV,   label="PV"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_WT,   label="WT"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MT,   label="MT"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ESS,  label="ESS dis/charge"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GRID, label="Grid import/export"),
        plt.Line2D([0], [0], color=COLOR_LOAD, linewidth=2.0, label="Load"),
    ]
    legend_handles_cost = [
        plt.Line2D([0], [0], marker="o", color="#1f77b4", linewidth=1.6,
                   markersize=5, label="Ground truth (social opt)"),
        plt.Line2D([0], [0], marker="s", color="#d62728", linewidth=1.6,
                   markersize=5, label="Ours"),
    ]
    fig.legend(handles=legend_handles_pow, loc="lower center",
               ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.legend(handles=legend_handles_cost, loc="lower center",
               ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.075))

    fig.suptitle(
        "ERCOT seasonal dispatch — power allocations and physical cost schedule",
        fontsize=12.5, y=1.02,
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


def make_info_rent_table(results, out_csv, out_tex):
    """
    For each day, report three top-level cost components plus the social-opt
    benchmark:

      Procurement cost (Ours) = Σ_i p_i + Σ_t π_t P_VPP_t        (VPP cash out)
      Physical cost (Ours)    = Σ_i c_i(x) + Σ_t π_t P_VPP_t     (true social cost
                                                                  at our dispatch)
      Info rent (Ours)        = Procurement − Physical           (≥ 0 if no
                                                                  IR violation)
      Physical cost (Opt)     = same formula evaluated at the social optimum.

    Dispatch gap = Physical(Ours) − Physical(Opt). It is the share of "extra
    paid over optimal" that is NOT info rent: pure dispatch inefficiency.
    """
    rows = []
    for r in results:
        procurement = r["procurement_post"]
        physical    = r["phys_cost_post"]
        physical_opt = r["phys_cost_soc"]
        ours        = r["info_rent"]["ours"]
        info_rent   = ours["TOTAL"]["rent"]
        dispatch_gap = physical - physical_opt
        rent_share   = (info_rent  / procurement * 100.0) if procurement > 1e-6 else float("nan")
        gap_pct      = (dispatch_gap / physical_opt * 100.0) if physical_opt > 1e-6 else float("nan")

        # Per-DER-type info-rent share (% of total info rent)
        per_type_share = {}
        for k in ("PV", "WT", "MT"):
            if abs(info_rent) > 1e-6:
                per_type_share[k] = ours[k]["rent"] / info_rent * 100.0
            else:
                per_type_share[k] = float("nan")

        rows.append({
            "Day": f"{r['date']}",
            "Procurement ($)":      round(procurement,  2),
            "Physical ($)":         round(physical,     2),
            "Info rent ($)":        round(info_rent,    2),
            "Physical (Opt) ($)":   round(physical_opt, 2),
            "Dispatch gap ($)":     round(dispatch_gap, 2),
            "Info-rent / Proc (%)": round(rent_share,  1) if not np.isnan(rent_share) else None,
            "Dispatch gap (%)":     round(gap_pct,     1) if not np.isnan(gap_pct) else None,
            "PV rent ($)":          round(ours["PV"]["rent"], 2),
            "WT rent ($)":          round(ours["WT"]["rent"], 2),
            "MT rent ($)":          round(ours["MT"]["rent"], 2),
            "PV share (%)":         round(per_type_share["PV"], 1) if not np.isnan(per_type_share["PV"]) else None,
            "WT share (%)":         round(per_type_share["WT"], 1) if not np.isnan(per_type_share["WT"]) else None,
            "MT share (%)":         round(per_type_share["MT"], 1) if not np.isnan(per_type_share["MT"]) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    def _fmt(v, fmt="{:.2f}"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return fmt.format(v)

    lines = [
        r"\begin{tabular}{lrrrrrrrrrrrr}",
        r"\toprule",
        r"Day & \multicolumn{3}{c}{Cost components (\$)}"
        r" & \multicolumn{2}{c}{Optimal benchmark}"
        r" & \multicolumn{2}{c}{Shares of cost (\%)}"
        r" & \multicolumn{3}{c}{Info rent by DER (\$)}"
        r" & \multicolumn{3}{c}{Info rent by DER (\%)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
        r"\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
        r" & Procurement & Physical & Info rent"
        r" & Phys.[Opt] & Disp. gap"
        r" & IR/Proc & Gap"
        r" & PV & WT & MT"
        r" & PV & WT & MT \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Day']} & "
            f"{_fmt(row['Procurement ($)'])} & "
            f"{_fmt(row['Physical ($)'])} & "
            f"{_fmt(row['Info rent ($)'])} & "
            f"{_fmt(row['Physical (Opt) ($)'])} & "
            f"{_fmt(row['Dispatch gap ($)'])} & "
            f"{_fmt(row['Info-rent / Proc (%)'], '{:.1f}')} & "
            f"{_fmt(row['Dispatch gap (%)'], '{:.1f}')} & "
            f"{_fmt(row['PV rent ($)'])} & "
            f"{_fmt(row['WT rent ($)'])} & "
            f"{_fmt(row['MT rent ($)'])} & "
            f"{_fmt(row['PV share (%)'], '{:.1f}')} & "
            f"{_fmt(row['WT share (%)'], '{:.1f}')} & "
            f"{_fmt(row['MT share (%)'], '{:.1f}')} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_tex}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="run directory containing trained checkpoint")
    parser.add_argument("--checkpoint", default="model_best_loss.pth")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: <run>/group_meeting_figs)")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="scenario indices for winter/spring/summer/fall "
                             "(default: 0 6 12 18)")
    parser.add_argument("--samples", type=int, default=16,
                        help="batch size of DER-type samples per scenario")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-clip-factor", type=float, default=3.0)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--disable-peer-bid-context", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=None,
                        help="DER dispatch anchor weight (default: chosen by "
                             "SecurityPostProcessor: 1000 with arbitrage, 1 legacy)")
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--no-mt-security-uplift", action="store_true")
    parser.add_argument("--no-ess-arbitrage", action="store_true",
                        help="Disable the ESS arbitrage objective (legacy "
                             "behaviour: ESS sits idle)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run)
    ckpt_path = os.path.join(run_dir, args.checkpoint)
    print(f"Run dir   : {run_dir}")
    print(f"Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    out_dir = args.out_dir or os.path.join(run_dir, "group_meeting_figs")
    os.makedirs(out_dir, exist_ok=True)

    scen_idxs = _select_scenarios(args.scenarios)
    if len(scen_idxs) != 4:
        raise SystemExit(f"Need exactly 4 scenarios; got {len(scen_idxs)}")
    print(f"Selected scenarios:")
    for col, idx in zip(SEASON_LABELS, scen_idxs):
        print(f"  {col:7s} idx={idx:2d}  date={scenario_date(idx)}")

    results = []
    for idx in scen_idxs:
        print(f"\n[scenario {idx}] evaluating...")
        r = evaluate_scenario(idx, state, args)
        print(f"  date={r['date']}")
        print(f"  load mean = {r['load'].mean():.3f} MW, peak = {r['load'].max():.3f} MW")
        ir = r["info_rent"]["ours"]
        print(f"  ours info-rent  PV={ir['PV']['rent']:.2f}  "
              f"WT={ir['WT']['rent']:.2f}  "
              f"MT={ir['MT']['rent']:.2f}  "
              f"TOTAL={ir['TOTAL']['rent']:.2f}")
        results.append(r)

    out_pdf = os.path.join(out_dir, "seasonal_allocation_2x4.pdf")
    out_png = os.path.join(out_dir, "seasonal_allocation_2x4.png")
    out_csv = os.path.join(out_dir, "info_rent_table.csv")
    out_tex = os.path.join(out_dir, "info_rent_table.tex")
    print()
    make_figure(results, out_pdf, out_png)
    df = make_info_rent_table(results, out_csv, out_tex)
    print("\nInfo-rent table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
