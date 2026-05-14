#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: ERCOT Pre/Post Feasibility
----------------------------------------
Compare the learned posted-price mechanism against the constrained social
optimum on ERCOT typical-day scenarios.

Rows are intentionally narrow:
  - social_optimum: full-information feasible reference.
  - ours_pre: learned posted-price dispatch before security postprocess.
  - ours_post: learned posted-price dispatch after security postprocess.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "our_method"))

from baseline.baseline_common_multi import JointQPMulti  # noqa: E402
from data.ercot_profiles import num_scenarios as ercot_num_scenarios  # noqa: E402
from network.vpp_network_multi import build_network_multi  # noqa: E402
from our_method.evaluate_posted_price import (  # noqa: E402
    TYPE_CAP_RATIO,
    classify_sources,
    latest_run,
    load_state,
)
from our_method.postprocess_security import SecurityPostProcessor  # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti  # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti  # noqa: E402


SUMMARY_COLUMNS = [
    "name",
    "method",
    "stage",
    "postprocess_mode",
    "operation_cost",
    "info_rent_cost",
    "total_procurement_cost",
    "operation_cost_gap_pct",
    "dispatch_l1_gap_mwh",
    "der_energy_mwh",
    "renewable_energy_mwh",
    "renewable_share_pct",
    "grid_import_mwh",
    "feasible_rate_pct",
    "feasible_flag",
    "mt_floor_gap_mwh",
    "mt_floor_gap_max_mwh",
    "line_violation_max_mw",
    "voltage_violation_max_pu",
    "physical_cap_violation_max_mw",
    "non_mt_offer_violation_max_mw",
    "power_balance_residual_max_mw",
    "postprocess_mt_slack_mwh",
    "positive_adjustment_mwh",
    "negative_adjustment_mwh",
    "correction_l1_mwh",
    "mt_security_uplift_mwh",
    "positive_adjustment_payment",
    "utility_min",
    "utility_shortfall_cost",
    "utility_mean",
    "postprocess_status",
]


def default_run_for_checkpoints(root: str, checkpoints: list) -> str:
    runs_dir = os.path.join(root, "runs")
    candidates = [
        os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    candidates.sort(key=os.path.getmtime, reverse=True)
    for ckpt in checkpoints:
        for run_dir in candidates:
            if os.path.exists(os.path.join(run_dir, ckpt)):
                return run_dir
    return latest_run(root)


def load_learned_mechanism(net: dict, run_dir: str, checkpoint: str, args):
    mech = VPPMechanismMulti(
        net,
        posted_price_cfg=dict(
            price_arch=args.price_arch,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_dropout=args.transformer_dropout,
            pi_buyback_ratio=args.pi_buyback_ratio,
            use_peer_bid_context=True,
            peer_bid_scale=args.peer_bid_scale,
            type_cap_ratio=TYPE_CAP_RATIO,
        ),
    )
    mech.load_state_dict(load_state(os.path.join(run_dir, checkpoint)),
                         strict=False)
    mech.eval()
    return mech


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def true_cost_per_der(types_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
    a = types_np[:, None, :, 0]
    b = types_np[:, None, :, 1]
    return (a * x_np ** 2 + b * x_np).sum(axis=1)


def source_energy_metrics(net: dict, x_np: np.ndarray,
                          source_types: list) -> dict:
    der_energy = float(x_np.sum(axis=(1, 2)).mean())
    out = {
        "der_energy_mwh": der_energy,
        "renewable_energy_mwh": 0.0,
        "renewable_share_pct": 0.0,
    }
    for typ in ("PV", "WT"):
        mask = np.asarray([s == typ for s in source_types], dtype=bool)
        if mask.any():
            out["renewable_energy_mwh"] += float(
                x_np[:, :, mask].sum(axis=(1, 2)).mean())
    out["renewable_share_pct"] = (
        100.0 * out["renewable_energy_mwh"] / max(der_energy, 1e-9))
    return out


def economics(net: dict, types_np: np.ndarray, x_np: np.ndarray,
              payment_np: np.ndarray, P_np: np.ndarray,
              source_types: list) -> dict:
    cost_i = true_cost_per_der(types_np, x_np)
    payment = payment_np.sum(axis=1)
    true_cost = cost_i.sum(axis=1)
    grid_cost = (np.asarray(net["pi_DA_profile"])[None, :] * P_np).sum(axis=1)
    utility = payment_np - cost_i
    out = {
        "operation_cost": float((true_cost + grid_cost).mean()),
        "info_rent_cost": float((payment - true_cost).mean()),
        "total_procurement_cost": float((payment + grid_cost).mean()),
        "der_payment": float(payment.mean()),
        "grid_cost": float(grid_cost.mean()),
        "true_der_cost": float(true_cost.mean()),
        "grid_import_mwh": float(P_np.sum(axis=1).mean()),
        "utility_min": float(utility.min()),
        "utility_shortfall_cost": float(max(0.0, -utility.min())),
        "utility_mean": float(utility.mean()),
    }
    out.update(source_energy_metrics(net, x_np, source_types))
    return out


def status_counter(statuses: list) -> str:
    if not statuses:
        return ""
    return ";".join(
        f"{k}:{v}" for k, v in sorted(Counter(statuses).items()))


def security_metrics(net: dict, x_np: np.ndarray, P_np: np.ndarray,
                     offer_cap_np: np.ndarray, ess_net_np: np.ndarray,
                     x_ref_np: np.ndarray = None, rho_np: np.ndarray = None,
                     mt_slack_np: np.ndarray = None, statuses: list = None,
                     tol: float = 1e-4) -> dict:
    load = np.asarray(net["load_profile"], dtype=np.float64)
    x_bar = np.asarray(net["x_bar_profile"], dtype=np.float64)
    A_flow = np.asarray(net["A_flow"], dtype=np.float64)
    A_volt = np.asarray(net["A_volt"], dtype=np.float64)
    A_flow_ess = np.asarray(net["A_flow_ess"], dtype=np.float64)
    A_volt_ess = np.asarray(net["A_volt_ess"], dtype=np.float64)
    flow_up = np.asarray(net["flow_margin_up_profile"], dtype=np.float64)
    flow_dn = np.asarray(net["flow_margin_dn_profile"], dtype=np.float64)
    volt_up = np.asarray(net["volt_margin_up_profile"], dtype=np.float64)
    volt_dn = np.asarray(net["volt_margin_dn_profile"], dtype=np.float64)
    mt_idx = np.asarray(net["mt_indices"], dtype=np.int64)
    non_mt_idx = np.asarray(
        [i for i in range(x_np.shape[2]) if i not in set(mt_idx.tolist())],
        dtype=np.int64,
    )

    if offer_cap_np is None:
        offer_cap_np = np.broadcast_to(x_bar[None, :, :], x_np.shape)
    if ess_net_np is None:
        ess_net_np = np.zeros((x_np.shape[0], x_np.shape[1],
                               A_flow_ess.shape[1]), dtype=np.float64)

    flow = x_np @ A_flow.T + ess_net_np @ A_flow_ess.T
    flow_viol = np.maximum(flow - flow_up[None, :, :],
                           -flow - flow_dn[None, :, :])
    flow_viol = np.maximum(flow_viol, 0.0)

    volt = x_np @ A_volt.T + ess_net_np @ A_volt_ess.T
    volt_viol = np.maximum(volt - volt_up[None, :, :],
                           -volt - volt_dn[None, :, :])
    volt_viol = np.maximum(volt_viol, 0.0)

    cap_viol = np.maximum(x_np - x_bar[None, :, :], -x_np)
    cap_viol = np.maximum(cap_viol, 0.0)

    floor = float(net["ctrl_min_ratio"]) * load[None, :]
    mt_dispatch = x_np[:, :, mt_idx].sum(axis=2)
    mt_floor_gap_by_sample = np.maximum(floor - mt_dispatch, 0.0).sum(axis=1)
    mt_offer = offer_cap_np[:, :, mt_idx].sum(axis=2)
    mt_offer_gap_by_sample = np.maximum(floor - mt_offer, 0.0).sum(axis=1)

    non_mt_offer_viol = 0.0
    if non_mt_idx.size > 0:
        non_mt_offer_viol = float(np.maximum(
            x_np[:, :, non_mt_idx] - offer_cap_np[:, :, non_mt_idx],
            0.0,
        ).max())
    mt_security_uplift = np.maximum(
        x_np[:, :, mt_idx] - offer_cap_np[:, :, mt_idx], 0.0)

    balance = (x_np.sum(axis=2) + ess_net_np.sum(axis=2) + P_np
               - load[None, :])

    if x_ref_np is None:
        pos_adj = np.zeros_like(x_np)
        neg_adj = np.zeros_like(x_np)
    else:
        delta = x_np - x_ref_np
        pos_adj = np.maximum(delta, 0.0)
        neg_adj = np.maximum(-delta, 0.0)

    slack_total = 0.0
    if mt_slack_np is not None:
        slack_total = float(np.nansum(mt_slack_np, axis=1).mean())

    feasible = (
        float(flow_viol.max()) <= tol
        and float(volt_viol.max()) <= tol
        and float(cap_viol.max()) <= tol
        and float(np.abs(balance).max()) <= tol
        and float(mt_floor_gap_by_sample.max()) <= tol
        and slack_total <= tol
    )

    out = {
        "feasible_flag": int(feasible),
        "feasible_rate_pct": 100.0 * float(feasible),
        "line_violation_max_mw": float(flow_viol.max()),
        "voltage_violation_max_pu": float(volt_viol.max()),
        "physical_cap_violation_max_mw": float(cap_viol.max()),
        "power_balance_residual_max_mw": float(np.abs(balance).max()),
        "mt_floor_gap_mwh": float(mt_floor_gap_by_sample.mean()),
        "mt_floor_gap_max_mwh": float(mt_floor_gap_by_sample.max()),
        "mt_offer_gap_mwh": float(mt_offer_gap_by_sample.mean()),
        "mt_security_uplift_mwh": float(mt_security_uplift.sum(axis=(1, 2)).mean()),
        "non_mt_offer_violation_max_mw": non_mt_offer_viol,
        "positive_adjustment_mwh": float(pos_adj.sum(axis=(1, 2)).mean()),
        "negative_adjustment_mwh": float(neg_adj.sum(axis=(1, 2)).mean()),
        "correction_l1_mwh": float((pos_adj + neg_adj).sum(axis=(1, 2)).mean()),
        "postprocess_mt_slack_mwh": slack_total,
        "postprocess_status": status_counter(statuses or []),
    }
    if rho_np is not None:
        out["positive_adjustment_payment"] = float(
            (rho_np * pos_adj).sum(axis=(1, 2)).mean())
    return out


def per_time_rows(method_name: str, stage: str, mode: str, net: dict,
                  x_np: np.ndarray, P_np: np.ndarray,
                  offer_cap_np: np.ndarray, ess_net_np: np.ndarray,
                  scenario_idx, scenario_date: str,
                  x_ref_np: np.ndarray = None) -> list:
    load = np.asarray(net["load_profile"], dtype=np.float64)
    A_flow = np.asarray(net["A_flow"], dtype=np.float64)
    A_volt = np.asarray(net["A_volt"], dtype=np.float64)
    A_flow_ess = np.asarray(net["A_flow_ess"], dtype=np.float64)
    A_volt_ess = np.asarray(net["A_volt_ess"], dtype=np.float64)
    flow_up = np.asarray(net["flow_margin_up_profile"], dtype=np.float64)
    flow_dn = np.asarray(net["flow_margin_dn_profile"], dtype=np.float64)
    volt_up = np.asarray(net["volt_margin_up_profile"], dtype=np.float64)
    volt_dn = np.asarray(net["volt_margin_dn_profile"], dtype=np.float64)
    mt_idx = np.asarray(net["mt_indices"], dtype=np.int64)
    if offer_cap_np is None:
        offer_cap_np = np.broadcast_to(
            np.asarray(net["x_bar_profile"])[None, :, :], x_np.shape)
    if ess_net_np is None:
        ess_net_np = np.zeros((x_np.shape[0], x_np.shape[1],
                               A_flow_ess.shape[1]), dtype=np.float64)
    if x_ref_np is None:
        pos_adj = np.zeros_like(x_np)
    else:
        pos_adj = np.maximum(x_np - x_ref_np, 0.0)

    flow = x_np @ A_flow.T + ess_net_np @ A_flow_ess.T
    flow_viol = np.maximum(flow - flow_up[None, :, :],
                           -flow - flow_dn[None, :, :])
    flow_viol = np.maximum(flow_viol, 0.0)
    volt = x_np @ A_volt.T + ess_net_np @ A_volt_ess.T
    volt_viol = np.maximum(volt - volt_up[None, :, :],
                           -volt - volt_dn[None, :, :])
    volt_viol = np.maximum(volt_viol, 0.0)
    floor = float(net["ctrl_min_ratio"]) * load
    mt_dispatch = x_np[:, :, mt_idx].sum(axis=2)
    mt_offer = offer_cap_np[:, :, mt_idx].sum(axis=2)
    balance = x_np.sum(axis=2) + ess_net_np.sum(axis=2) + P_np - load[None, :]

    rows = []
    for t in range(x_np.shape[1]):
        rows.append({
            "scenario_idx": scenario_idx,
            "scenario_date": scenario_date,
            "name": method_name,
            "stage": stage,
            "postprocess_mode": mode,
            "hour": t,
            "load_mw": float(load[t]),
            "mt_floor_mw": float(floor[t]),
            "mt_dispatch_mw": float(mt_dispatch[:, t].mean()),
            "mt_offer_mw": float(mt_offer[:, t].mean()),
            "mt_floor_gap_mw": float(
                np.maximum(floor[t] - mt_dispatch[:, t], 0.0).mean()),
            "positive_adjustment_mwh": float(pos_adj[:, t, :].sum(axis=1).mean()),
            "der_dispatch_mw": float(x_np[:, t, :].sum(axis=1).mean()),
            "grid_import_mw": float(P_np[:, t].mean()),
            "line_violation_max_mw": float(flow_viol[:, t, :].max()),
            "voltage_violation_max_pu": float(volt_viol[:, t, :].max()),
            "power_balance_residual_max_mw": float(np.abs(balance[:, t]).max()),
        })
    return rows


def solve_social_reference(net: dict, types: torch.Tensor):
    qp = JointQPMulti(net)
    types_np = to_numpy(types)
    B, N, _ = types_np.shape
    T = int(net["T"])
    n_ess = int(net["ess_params"]["n_ess"])
    x_out = np.zeros((B, T, N), dtype=np.float64)
    P_out = np.zeros((B, T), dtype=np.float64)
    ess_net_out = np.zeros((B, T, n_ess), dtype=np.float64)
    statuses = []

    for b in range(B):
        a = np.broadcast_to(types_np[b, :, 0][None, :], (T, N))
        c = np.broadcast_to(types_np[b, :, 1][None, :], (T, N))
        x_np, P_np, status = qp.solve(a, c)
        x_out[b] = x_np
        P_out[b] = P_np
        statuses.append(status)
        ess = qp.last_ess
        if ess is not None:
            ess_net_out[b] = ess["P_d"] - ess["P_c"]

    payment = true_cost_per_der(types_np, x_out)
    return x_out, P_out, payment, ess_net_out, statuses


def add_reference_metrics(row: dict, x_np: np.ndarray, x_social_np: np.ndarray,
                          social_cost: float) -> dict:
    out = dict(row)
    out["dispatch_l1_gap_mwh"] = float(
        np.abs(x_np - x_social_np).sum(axis=(1, 2)).mean())
    if social_cost > 1e-9:
        out["operation_cost_gap_pct"] = (
            100.0 * (float(out["operation_cost"]) - social_cost) / social_cost)
    else:
        out["operation_cost_gap_pct"] = 0.0
    return out


def evaluate_social_optimum(net: dict, types: torch.Tensor,
                            source_types: list, args,
                            scenario_idx, scenario_date: str):
    types_np = to_numpy(types)
    x_soc, P_soc, p_soc, ess_soc, statuses = solve_social_reference(net, types)
    offer_phys = np.broadcast_to(
        np.asarray(net["x_bar_profile"])[None, :, :], x_soc.shape)

    row = {}
    row.update(economics(net, types_np, x_soc, p_soc, P_soc, source_types))
    row.update(security_metrics(
        net, x_soc, P_soc, offer_phys, ess_soc,
        statuses=statuses, tol=args.feasibility_tol))
    row.update({
        "name": "social_optimum",
        "method": "social_optimum",
        "stage": "reference",
        "postprocess_mode": "not_applicable",
        "checkpoint": "",
        "run_dir": "",
        "scenario_idx": scenario_idx,
        "scenario_date": scenario_date,
    })
    row = add_reference_metrics(row, x_soc, x_soc, row["operation_cost"])

    time_rows = per_time_rows(
        "social_optimum", "reference", "not_applicable",
        net, x_soc, P_soc, offer_phys, ess_soc, scenario_idx, scenario_date)
    return row, time_rows, x_soc, row["operation_cost"]


def evaluate_ours(name: str, mech, net: dict, types: torch.Tensor,
                  x_social_np: np.ndarray, social_cost: float,
                  source_types: list, pp: SecurityPostProcessor,
                  args, scenario_idx, scenario_date: str,
                  checkpoint: str = "", run_dir: str = ""):
    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()

    dc3 = mech.dc3_opf
    ess_pre = None
    if getattr(dc3, "_last_P_d", None) is not None:
        ess_pre = to_numpy(dc3._last_P_d - dc3._last_P_c)

    types_np = to_numpy(types)
    x_pre_np = to_numpy(x_pre).astype(np.float64)
    rho_np = to_numpy(rho).astype(np.float64)
    p_pre_np = to_numpy(p_pre).astype(np.float64)
    P_pre_np = to_numpy(P_pre).astype(np.float64)
    offer_np = to_numpy(offer_cap).astype(np.float64)

    rows = []
    time_rows = []

    pre = {}
    pre.update(economics(net, types_np, x_pre_np, p_pre_np, P_pre_np,
                         source_types))
    pre.update(security_metrics(
        net, x_pre_np, P_pre_np, offer_np, ess_pre,
        rho_np=rho_np, tol=args.feasibility_tol))
    pre.update({
        "name": f"{name}:pre",
        "method": "ours",
        "stage": "pre",
        "postprocess_mode": "none",
        "checkpoint": checkpoint,
        "run_dir": run_dir,
        "scenario_idx": scenario_idx,
        "scenario_date": scenario_date,
    })
    rows.append(add_reference_metrics(pre, x_pre_np, x_social_np, social_cost))
    time_rows.extend(per_time_rows(
        f"{name}:pre", "pre", "none", net, x_pre_np, P_pre_np,
        offer_np, ess_pre, scenario_idx, scenario_date))

    post = pp.process_batch(x_pre_np, P_pre_np, rho_np, offer_np)
    x_post_np = post.x.astype(np.float64)
    P_post_np = post.P_VPP.astype(np.float64)
    p_post_np = (rho_np * x_post_np).sum(axis=1)
    ess_post = None
    if post.P_d is not None and post.P_c is not None:
        ess_post = post.P_d.astype(np.float64) - post.P_c.astype(np.float64)

    row = {}
    row.update(economics(net, types_np, x_post_np, p_post_np, P_post_np,
                         source_types))
    row.update(security_metrics(
        net, x_post_np, P_post_np, offer_np, ess_post,
        x_ref_np=x_pre_np, rho_np=rho_np,
        mt_slack_np=post.mt_slack, statuses=post.status,
        tol=args.feasibility_tol))
    row.update({
        "name": f"{name}:post",
        "method": "ours",
        "stage": "post",
        "postprocess_mode": "mt_uplift_enabled",
        "checkpoint": checkpoint,
        "run_dir": run_dir,
        "scenario_idx": scenario_idx,
        "scenario_date": scenario_date,
    })
    rows.append(add_reference_metrics(row, x_post_np, x_social_np, social_cost))
    time_rows.extend(per_time_rows(
        f"{name}:post", "post", "mt_uplift_enabled", net,
        x_post_np, P_post_np, offer_np, ess_post, scenario_idx,
        scenario_date, x_ref_np=x_pre_np))

    return rows, time_rows


def write_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value, digits: int = 5) -> str:
    if value in ("", None):
        return ""
    try:
        if np.isnan(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_table(path: str, rows: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |",
    ]
    for row in rows:
        values = []
        for col in SUMMARY_COLUMNS:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                value = fmt_value(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _as_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def merge_statuses(statuses: list) -> str:
    counts = Counter()
    passthrough = []
    for status in statuses:
        if not status:
            continue
        for item in str(status).split(";"):
            if not item:
                continue
            if ":" not in item:
                passthrough.append(item)
                continue
            label, count = item.rsplit(":", 1)
            try:
                counts[label] += int(float(count))
            except ValueError:
                passthrough.append(item)
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    parts.extend(sorted(set(passthrough)))
    return ";".join(parts)


def aggregate_summary_rows(rows: list) -> list:
    grouped = {}
    key_cols = ("name", "method", "stage", "postprocess_mode", "checkpoint")
    for row in rows:
        key = tuple(row.get(col, "") for col in key_cols)
        grouped.setdefault(key, []).append(row)

    max_cols = {
        "mt_floor_gap_max_mwh",
        "line_violation_max_mw",
        "voltage_violation_max_pu",
        "physical_cap_violation_max_mw",
        "non_mt_offer_violation_max_mw",
        "power_balance_residual_max_mw",
        "postprocess_mt_slack_mwh",
        "utility_shortfall_cost",
    }
    min_cols = {"utility_min", "feasible_flag"}

    out = []
    for _key, group in grouped.items():
        merged = dict(group[0])
        all_cols = sorted({k for row in group for k in row})
        for col in all_cols:
            vals = [row.get(col, "") for row in group]
            nums = [_as_float(v) for v in vals]
            if col in {"scenario_idx", "scenario_date"}:
                merged[col] = "multiple" if len(group) > 1 else vals[0]
            elif col == "postprocess_status":
                merged[col] = merge_statuses(vals)
            elif all(v is not None for v in nums):
                if col in max_cols:
                    merged[col] = float(np.max(nums))
                elif col in min_cols:
                    merged[col] = float(np.min(nums))
                else:
                    merged[col] = float(np.mean(nums))
            elif len({str(v) for v in vals}) == 1:
                merged[col] = vals[0]
            else:
                merged[col] = vals[0]
        merged["n_eval_rows"] = len(group)
        out.append(merged)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run directory. Defaults to latest run containing a requested checkpoint.")
    parser.add_argument("--checkpoints", nargs="*", default=["model_best.pth"])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--scenario-idx", type=int, default=0,
                        help="ERCOT scenario index when not using all scenarios.")
    parser.add_argument("--all-ercot-scenarios", action="store_true",
                        help="Evaluate all 24 ERCOT typical-day scenarios.")
    parser.add_argument("--pi-clip-factor", type=float, default=3.0)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--adjustment-weight", type=float, default=1000.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e7)
    parser.add_argument("--feasibility-tol", type=float, default=1e-3)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def scenario_indices(args):
    if args.all_ercot_scenarios:
        return list(range(ercot_num_scenarios()))
    return [args.scenario_idx]


def build_eval_network(args, scenario_idx: int):
    return build_network_multi(
        scenario_idx=int(scenario_idx),
        ctrl_min_ratio=args.ctrl_min_ratio,
        pi_clip_factor=args.pi_clip_factor,
    )


def evaluate_one_scenario(args, run_dir: str, scenario_idx: int):
    net = build_eval_network(args, scenario_idx)
    scenario_date = str(net.get("scenario_date", ""))
    source_types = classify_sources(net)
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed + int(scenario_idx))
    types = prior.sample(args.samples, device="cpu")

    pp = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=True,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )

    print("Solving constrained social optimum reference...")
    social_row, social_time, x_social_np, social_cost = evaluate_social_optimum(
        net, types, source_types, args, scenario_idx, scenario_date)

    rows = [social_row]
    time_rows = list(social_time)

    for ckpt in args.checkpoints:
        ckpt_path = os.path.join(run_dir, ckpt)
        if not os.path.exists(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue
        print(f"Evaluating learned posted price: {ckpt}")
        mech = load_learned_mechanism(net, run_dir, ckpt, args)
        r, t = evaluate_ours(
            f"ours_{ckpt}",
            mech,
            net,
            types,
            x_social_np,
            social_cost,
            source_types,
            pp,
            args,
            scenario_idx,
            scenario_date,
            checkpoint=ckpt,
            run_dir=run_dir,
        )
        rows.extend(r)
        time_rows.extend(t)

    return rows, time_rows


def main():
    args = parse_args()
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoints(_ROOT, args.checkpoints))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results_ercot")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    time_rows = []
    for sc in scenario_indices(args):
        print("\n" + "=" * 80)
        print(f"Experiment 2 setting: ERCOT scenario {sc}")
        print("=" * 80)
        r, t = evaluate_one_scenario(args, run_dir, sc)
        rows.extend(r)
        time_rows.extend(t)

    detailed_csv = os.path.join(out_dir, "pre_post_feasibility_detailed.csv")
    summary_csv = os.path.join(out_dir, "pre_post_feasibility_summary.csv")
    summary_md = os.path.join(out_dir, "pre_post_feasibility_summary.md")
    time_csv = os.path.join(out_dir, "pre_post_feasibility_by_time.csv")
    config_path = os.path.join(out_dir, "pre_post_feasibility_config.json")

    summary_rows = aggregate_summary_rows(rows)
    write_csv(detailed_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_markdown_table(summary_md, summary_rows)
    write_csv(time_csv, time_rows)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["data_source"] = "ercot"
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed CSV: {detailed_csv}")
    print(f"Saved summary CSV : {summary_csv}")
    print(f"Saved summary MD  : {summary_md}")
    print(f"Saved by-time CSV : {time_csv}")
    print(f"Saved config      : {config_path}")
    print("\nSummary rows:")
    for row in summary_rows:
        print(
            f"  {row['name']:<32} "
            f"feas={fmt_value(row.get('feasible_rate_pct'), 2):>7}% "
            f"op={fmt_value(row.get('operation_cost')):<10} "
            f"gap={fmt_value(row.get('operation_cost_gap_pct'), 3):>8}% "
            f"dispatch={fmt_value(row.get('dispatch_l1_gap_mwh')):<9} "
            f"corr={fmt_value(row.get('correction_l1_mwh')):<9} "
            f"line={fmt_value(row.get('line_violation_max_mw')):<9} "
            f"volt={fmt_value(row.get('voltage_violation_max_pu')):<9}"
        )


if __name__ == "__main__":
    main()
