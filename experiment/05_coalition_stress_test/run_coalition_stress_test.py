#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 5: Coalition and Collusion Stress Test
-------------------------------------------------
Evaluate coordinated cost-bid deviations by groups of DERs.

The current 33-bus model treats physical availability as public information,
so this script studies coalition behavior through cost-bid manipulation and
economic withholding proxies. Utility is always computed with the true type.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "our_method"))

from baseline.baseline_common_multi import JointQPMulti  # noqa: E402
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


DETAIL_COLUMNS = [
    "method",
    "stage",
    "coalition_id",
    "coalition_group",
    "coalition_size",
    "member_indices",
    "member_labels",
    "member_types",
    "strategy",
    "strategy_param",
    "coalition_utility_truth_mean",
    "coalition_utility_misreport_mean",
    "coalition_utility_gain_mean",
    "coalition_regret_mean",
    "coalition_regret_max_sample",
    "per_member_regret_mean",
    "member_gain_mean_min",
    "member_gain_mean_max",
    "all_member_mean_gain_nonnegative",
    "procurement_truth",
    "procurement_misreport",
    "procurement_delta",
    "social_cost_truth",
    "social_cost_misreport",
    "social_cost_delta",
    "info_rent_truth",
    "info_rent_misreport",
    "info_rent_delta",
    "der_payment_delta",
    "grid_cost_delta",
    "mt_floor_gap_truth",
    "mt_floor_gap_misreport",
    "mt_floor_gap_delta",
    "postprocess_mt_slack_delta",
    "positive_adjustment_delta",
    "offer_cap_delta_mwh",
    "coalition_rho_delta_mean",
    "coalition_rho_delta_max",
    "outside_rho_delta_mean",
    "all_rho_delta_mean",
    "utility_min_misreport",
]

WORST_COLUMNS = [
    "method",
    "stage",
    "coalition_group",
    "coalition_size",
    "coalition_id",
    "member_labels",
    "member_types",
    "strategy",
    "strategy_param",
    "coalition_regret_mean",
    "coalition_regret_max_sample",
    "per_member_regret_mean",
    "coalition_utility_gain_mean",
    "procurement_delta",
    "info_rent_delta",
    "mt_floor_gap_delta",
    "positive_adjustment_delta",
    "offer_cap_delta_mwh",
    "coalition_rho_delta_mean",
    "outside_rho_delta_mean",
]

HEADLINE_COLUMNS = [
    "method",
    "stage",
    "n_size_buckets",
    "worst_coalition_group",
    "worst_coalition_size",
    "worst_member_labels",
    "worst_strategy",
    "worst_strategy_param",
    "worst_regret_mean",
    "worst_regret_max_sample",
    "worst_per_member_regret_mean",
    "procurement_delta_at_worst",
    "info_rent_delta_at_worst",
    "mt_floor_gap_delta_at_worst",
    "positive_adjustment_delta_at_worst",
    "offer_cap_delta_at_worst",
    "coalition_rho_delta_mean_at_worst",
    "outside_rho_delta_mean_at_worst",
]


def default_run_for_checkpoint(root: str, checkpoint: str) -> str:
    runs_dir = os.path.join(root, "runs")
    candidates = [
        os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    candidates.sort(key=os.path.getmtime, reverse=True)
    for run_dir in candidates:
        if os.path.exists(os.path.join(run_dir, checkpoint)):
            return run_dir
    return latest_run(root)


def load_mechanism(net: dict, run_dir: str, checkpoint: str,
                   args, use_peer_bid_context: bool) -> VPPMechanismMulti:
    mech = VPPMechanismMulti(
        net,
        posted_price_cfg=dict(
            price_arch=args.price_arch,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_dropout=args.transformer_dropout,
            pi_buyback_ratio=args.pi_buyback_ratio,
            use_peer_bid_context=use_peer_bid_context,
            peer_bid_scale=args.peer_bid_scale,
            type_cap_ratio=TYPE_CAP_RATIO,
        ),
    )
    mech.load_state_dict(load_state(os.path.join(run_dir, checkpoint)),
                         strict=False)
    mech.eval()
    return mech


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def true_cost_per_der_np(types_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
    a = types_np[:, None, :, 0]
    b = types_np[:, None, :, 1]
    return (a * x_np ** 2 + b * x_np).sum(axis=1)


def reported_cost_per_der_np(bids_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
    a = bids_np[:, None, :, 0]
    b = bids_np[:, None, :, 1]
    return (a * x_np ** 2 + b * x_np).sum(axis=1)


def stage_metrics(net: dict, true_types_np: np.ndarray, x_np: np.ndarray,
                  p_np: np.ndarray, P_np: np.ndarray,
                  positive_adjustment_np: np.ndarray = None,
                  mt_slack_np: np.ndarray = None) -> dict:
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float64)
    true_cost_i = true_cost_per_der_np(true_types_np, x_np)
    utility = p_np - true_cost_i
    payment = p_np.sum(axis=1)
    true_cost = true_cost_i.sum(axis=1)
    grid_cost = (pi[None, :] * P_np).sum(axis=1)

    mt_idx = np.asarray(net["mt_indices"], dtype=np.int64)
    load = np.asarray(net["load_profile"], dtype=np.float64)
    floor = float(net["ctrl_min_ratio"]) * load[None, :]
    mt_dispatch = x_np[:, :, mt_idx].sum(axis=2)
    mt_gap = np.maximum(floor - mt_dispatch, 0.0).sum(axis=1)

    if positive_adjustment_np is None:
        positive_adjustment = np.zeros(x_np.shape[0], dtype=np.float64)
    else:
        positive_adjustment = positive_adjustment_np.sum(axis=(1, 2))
    if mt_slack_np is None:
        mt_slack = np.zeros(x_np.shape[0], dtype=np.float64)
    else:
        mt_slack = np.nansum(mt_slack_np, axis=1)

    return {
        "utility": utility,
        "procurement_cost": float((payment + grid_cost).mean()),
        "social_cost_true": float((true_cost + grid_cost).mean()),
        "der_payment": float(payment.mean()),
        "grid_cost": float(grid_cost.mean()),
        "true_der_cost": float(true_cost.mean()),
        "info_rent": float((payment - true_cost).mean()),
        "utility_min": float(utility.min()),
        "mt_floor_gap_mwh": float(mt_gap.mean()),
        "mt_floor_gap_max_mwh": float(mt_gap.max()),
        "postprocess_mt_slack_mwh": float(mt_slack.mean()),
        "positive_adjustment_mwh": float(positive_adjustment.mean()),
    }


def evaluate_posted_price(mech: VPPMechanismMulti, net: dict,
                          true_types: torch.Tensor, bids: torch.Tensor,
                          postprocessor: SecurityPostProcessor) -> dict:
    true_np = to_numpy(true_types)
    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(bids)
        offer_cap = mech._last_offer_cap.detach().clone()

    x_pre_np = to_numpy(x_pre)
    rho_np = to_numpy(rho)
    P_pre_np = to_numpy(P_pre)
    p_pre_np = to_numpy(p_pre)
    offer_np = to_numpy(offer_cap)

    post = postprocessor.process_batch(x_pre_np, P_pre_np, rho_np, offer_np)
    x_post_np = post.x.astype(np.float64)
    P_post_np = post.P_VPP.astype(np.float64)
    p_post_np = (rho_np * x_post_np).sum(axis=1)

    return {
        "rho": rho_np,
        "offer_cap": offer_np,
        "offer_cap_mwh": float(offer_np.sum(axis=(1, 2)).mean()),
        "pre": stage_metrics(net, true_np, x_pre_np, p_pre_np, P_pre_np),
        "post": stage_metrics(
            net, true_np, x_post_np, p_post_np, P_post_np,
            positive_adjustment_np=post.positive_adjustment,
            mt_slack_np=post.mt_slack,
        ),
        "post_status": post.status,
    }


def solve_bid_opf(qp: JointQPMulti, bids_np: np.ndarray):
    B, N = bids_np.shape[:2]
    T = qp.T
    x_out = np.zeros((B, T, N), dtype=np.float32)
    P_out = np.zeros((B, T), dtype=np.float32)
    statuses = []
    for b in range(B):
        a = np.broadcast_to(bids_np[b, :, 0][None, :], (T, N))
        c = np.broadcast_to(bids_np[b, :, 1][None, :], (T, N))
        x_np, P_np, status = qp.solve(a, c)
        x_out[b] = x_np
        P_out[b] = P_np
        statuses.append(status)
    return x_out.astype(np.float64), P_out.astype(np.float64), statuses


def evaluate_bid_opf(qp: JointQPMulti, net: dict, true_types_np: np.ndarray,
                     bids_np: np.ndarray, settlement: str) -> dict:
    x_np, P_np, statuses = solve_bid_opf(qp, bids_np)
    if settlement == "pay_as_bid":
        p_np = reported_cost_per_der_np(bids_np, x_np)
    elif settlement == "uniform_da":
        rho = np.asarray(net["pi_DA_profile"], dtype=np.float64)[None, :, None]
        p_np = (rho * x_np).sum(axis=1)
    else:
        raise ValueError(f"Unknown settlement: {settlement}")
    metrics = stage_metrics(net, true_types_np, x_np, p_np, P_np)
    metrics["status"] = statuses
    return {"cleared": metrics}


def strategy_candidates(args):
    specs = []
    for scale in args.overreport_scales:
        specs.append(dict(
            strategy="group_cost_overreport",
            strategy_param=f"scale={scale:g}",
            kind="scale",
            a_scale=float(scale),
            b_scale=float(scale),
        ))
        specs.append(dict(
            strategy="group_linear_cost_overreport",
            strategy_param=f"b_scale={scale:g}",
            kind="scale",
            a_scale=1.0,
            b_scale=float(scale),
        ))
    for scale in args.underreport_scales:
        specs.append(dict(
            strategy="group_cost_underreport",
            strategy_param=f"scale={scale:g}",
            kind="scale",
            a_scale=float(scale),
            b_scale=float(scale),
        ))
    specs.append(dict(
        strategy="group_high_cost_withholding_proxy",
        strategy_param="bid=upper_bound",
        kind="set_hi",
    ))
    specs.append(dict(
        strategy="group_low_cost_pressure",
        strategy_param="bid=lower_bound",
        kind="set_lo",
    ))
    return specs


def apply_strategy(types: torch.Tensor, prior: DERTypePriorMulti,
                   members: list, spec: dict) -> torch.Tensor:
    bids = types.clone()
    idx = torch.tensor(members, dtype=torch.long, device=bids.device)
    if spec["kind"] == "scale":
        bids[:, idx, 0] = bids[:, idx, 0] * float(spec["a_scale"])
        bids[:, idx, 1] = bids[:, idx, 1] * float(spec["b_scale"])
    elif spec["kind"] == "set_hi":
        bids[:, idx, :] = prior.hi[idx].to(bids.device).unsqueeze(0)
    elif spec["kind"] == "set_lo":
        bids[:, idx, :] = prior.lo[idx].to(bids.device).unsqueeze(0)
    else:
        raise ValueError(f"Unknown strategy kind: {spec['kind']}")
    return prior.project(bids)


def type_groups(source_types: list) -> dict:
    groups = defaultdict(list)
    for i, typ in enumerate(source_types):
        groups[str(typ).upper()].append(i)
    groups["RENEWABLE"] = [
        i for i, typ in enumerate(source_types)
        if str(typ).upper() in {"PV", "WT"}
    ]
    groups["CONTROLLABLE"] = [
        i for i, typ in enumerate(source_types)
        if str(typ).upper() in {"MT", "DG", "DR"}
    ]
    groups["ALL"] = list(range(len(source_types)))
    return dict(groups)


def select_combinations(indices: list, size: int, max_count: int,
                        rng: np.random.Generator) -> list:
    combos = list(itertools.combinations(indices, size))
    if max_count <= 0 or len(combos) <= max_count:
        return [list(c) for c in combos]
    chosen = sorted(rng.choice(len(combos), size=max_count, replace=False))
    return [list(combos[i]) for i in chosen]


def build_coalitions(labels: list, source_types: list, args) -> list:
    rng = np.random.default_rng(args.coalition_seed)
    groups = type_groups(source_types)
    out = []
    for group_name in args.coalition_groups:
        key = group_name.upper()
        if key not in groups or not groups[key]:
            continue
        indices = groups[key]
        max_size = min(int(args.max_coalition_size), len(indices))
        for size in range(1, max_size + 1):
            combos = select_combinations(
                indices, size, int(args.max_combinations_per_bucket), rng)
            for rank, members in enumerate(combos):
                out.append({
                    "coalition_id": f"{key}_k{size}_{rank:03d}",
                    "coalition_group": key,
                    "coalition_size": size,
                    "members": members,
                    "member_indices": ",".join(str(i) for i in members),
                    "member_labels": ",".join(labels[i] for i in members),
                    "member_types": ",".join(source_types[i] for i in members),
                })
    return out


def price_delta_row(truth_eval: dict, mis_eval: dict, members: list) -> dict:
    rho_truth = truth_eval.get("rho")
    rho_mis = mis_eval.get("rho")
    if rho_truth is None or rho_mis is None:
        return {
            "coalition_rho_delta_mean": "",
            "coalition_rho_delta_max": "",
            "outside_rho_delta_mean": "",
            "all_rho_delta_mean": "",
        }
    member_mask = np.zeros(rho_truth.shape[2], dtype=bool)
    member_mask[members] = True
    coalition = np.abs(rho_mis[:, :, member_mask] - rho_truth[:, :, member_mask])
    outside = np.abs(rho_mis[:, :, ~member_mask] - rho_truth[:, :, ~member_mask])
    return {
        "coalition_rho_delta_mean": float(coalition.mean()),
        "coalition_rho_delta_max": float(coalition.max()),
        "outside_rho_delta_mean": float(outside.mean()) if outside.size else "",
        "all_rho_delta_mean": float(np.abs(rho_mis - rho_truth).mean()),
    }


def offer_cap_delta(truth_eval: dict, mis_eval: dict) -> object:
    if "offer_cap_mwh" not in truth_eval or "offer_cap_mwh" not in mis_eval:
        return ""
    return float(mis_eval["offer_cap_mwh"] - truth_eval["offer_cap_mwh"])


def make_detail_row(method: str, stage: str, coalition: dict, spec: dict,
                    truth_eval: dict, mis_eval: dict) -> dict:
    truth = truth_eval[stage]
    mis = mis_eval[stage]
    members = coalition["members"]
    u_truth_members = truth["utility"][:, members]
    u_mis_members = mis["utility"][:, members]
    member_gain = u_mis_members - u_truth_members
    coalition_gain = member_gain.sum(axis=1)
    member_gain_mean = member_gain.mean(axis=0)
    regret = np.maximum(coalition_gain, 0.0)

    row = {
        "method": method,
        "stage": stage,
        "coalition_id": coalition["coalition_id"],
        "coalition_group": coalition["coalition_group"],
        "coalition_size": coalition["coalition_size"],
        "member_indices": coalition["member_indices"],
        "member_labels": coalition["member_labels"],
        "member_types": coalition["member_types"],
        "strategy": spec["strategy"],
        "strategy_param": spec["strategy_param"],
        "coalition_utility_truth_mean": float(u_truth_members.sum(axis=1).mean()),
        "coalition_utility_misreport_mean": float(u_mis_members.sum(axis=1).mean()),
        "coalition_utility_gain_mean": float(coalition_gain.mean()),
        "coalition_regret_mean": float(regret.mean()),
        "coalition_regret_max_sample": float(regret.max()),
        "per_member_regret_mean": float(regret.mean() / max(1, len(members))),
        "member_gain_mean_min": float(member_gain_mean.min()),
        "member_gain_mean_max": float(member_gain_mean.max()),
        "all_member_mean_gain_nonnegative": bool((member_gain_mean >= -1e-8).all()),
        "procurement_truth": truth["procurement_cost"],
        "procurement_misreport": mis["procurement_cost"],
        "procurement_delta": mis["procurement_cost"] - truth["procurement_cost"],
        "social_cost_truth": truth["social_cost_true"],
        "social_cost_misreport": mis["social_cost_true"],
        "social_cost_delta": mis["social_cost_true"] - truth["social_cost_true"],
        "info_rent_truth": truth["info_rent"],
        "info_rent_misreport": mis["info_rent"],
        "info_rent_delta": mis["info_rent"] - truth["info_rent"],
        "der_payment_delta": mis["der_payment"] - truth["der_payment"],
        "grid_cost_delta": mis["grid_cost"] - truth["grid_cost"],
        "mt_floor_gap_truth": truth["mt_floor_gap_mwh"],
        "mt_floor_gap_misreport": mis["mt_floor_gap_mwh"],
        "mt_floor_gap_delta": mis["mt_floor_gap_mwh"] - truth["mt_floor_gap_mwh"],
        "postprocess_mt_slack_delta": (
            mis.get("postprocess_mt_slack_mwh", 0.0)
            - truth.get("postprocess_mt_slack_mwh", 0.0)
        ),
        "positive_adjustment_delta": (
            mis.get("positive_adjustment_mwh", 0.0)
            - truth.get("positive_adjustment_mwh", 0.0)
        ),
        "offer_cap_delta_mwh": offer_cap_delta(truth_eval, mis_eval),
        "utility_min_misreport": mis["utility_min"],
    }
    row.update(price_delta_row(truth_eval, mis_eval, members))
    return row


def summarize_details(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["stage"], row["coalition_group"],
            row["coalition_size"], row["strategy"],
        )
        grouped[key].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        method, stage, coalition_group, size, strategy = key
        worst = max(group, key=lambda r: r["coalition_regret_mean"])
        out.append({
            "method": method,
            "stage": stage,
            "coalition_group": coalition_group,
            "coalition_size": size,
            "strategy": strategy,
            "n_coalitions": len(group),
            "coalition_regret_mean": float(np.mean([
                r["coalition_regret_mean"] for r in group
            ])),
            "coalition_regret_max_case": float(max(
                r["coalition_regret_mean"] for r in group
            )),
            "coalition_regret_max_sample": float(max(
                r["coalition_regret_max_sample"] for r in group
            )),
            "per_member_regret_mean": float(np.mean([
                r["per_member_regret_mean"] for r in group
            ])),
            "procurement_delta_mean": float(np.mean([
                r["procurement_delta"] for r in group
            ])),
            "procurement_delta_max": float(max(
                r["procurement_delta"] for r in group
            )),
            "info_rent_delta_mean": float(np.mean([
                r["info_rent_delta"] for r in group
            ])),
            "positive_adjustment_delta_mean": float(np.mean([
                r["positive_adjustment_delta"] for r in group
            ])),
            "worst_coalition_id": worst["coalition_id"],
            "worst_member_labels": worst["member_labels"],
            "worst_strategy_param": worst["strategy_param"],
        })
    return out


def worst_by_size(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["stage"], row["coalition_group"],
            row["coalition_size"],
        )
        grouped[key].append(row)
    return [
        max(group, key=lambda r: r["coalition_regret_mean"])
        for _, group in sorted(grouped.items())
    ]


def headline_summary(worst_rows: list) -> list:
    grouped = defaultdict(list)
    for row in worst_rows:
        grouped[(row["method"], row["stage"])].append(row)
    out = []
    for (method, stage), group in sorted(grouped.items()):
        worst = max(group, key=lambda r: r["coalition_regret_mean"])
        out.append({
            "method": method,
            "stage": stage,
            "n_size_buckets": len(group),
            "worst_coalition_group": worst["coalition_group"],
            "worst_coalition_size": worst["coalition_size"],
            "worst_member_labels": worst["member_labels"],
            "worst_strategy": worst["strategy"],
            "worst_strategy_param": worst["strategy_param"],
            "worst_regret_mean": worst["coalition_regret_mean"],
            "worst_regret_max_sample": worst["coalition_regret_max_sample"],
            "worst_per_member_regret_mean": worst["per_member_regret_mean"],
            "procurement_delta_at_worst": worst["procurement_delta"],
            "info_rent_delta_at_worst": worst["info_rent_delta"],
            "mt_floor_gap_delta_at_worst": worst["mt_floor_gap_delta"],
            "positive_adjustment_delta_at_worst": worst["positive_adjustment_delta"],
            "offer_cap_delta_at_worst": worst["offer_cap_delta_mwh"],
            "coalition_rho_delta_mean_at_worst": worst["coalition_rho_delta_mean"],
            "outside_rho_delta_mean_at_worst": worst["outside_rho_delta_mean"],
        })
    return out


def size_curve(worst_rows: list) -> list:
    out = []
    for row in worst_rows:
        out.append({
            "method": row["method"],
            "stage": row["stage"],
            "coalition_group": row["coalition_group"],
            "coalition_size": row["coalition_size"],
            "worst_coalition_id": row["coalition_id"],
            "worst_member_labels": row["member_labels"],
            "worst_strategy": row["strategy"],
            "worst_strategy_param": row["strategy_param"],
            "worst_regret_mean": row["coalition_regret_mean"],
            "worst_per_member_regret_mean": row["per_member_regret_mean"],
            "procurement_delta": row["procurement_delta"],
            "info_rent_delta": row["info_rent_delta"],
            "positive_adjustment_delta": row["positive_adjustment_delta"],
            "offer_cap_delta_mwh": row["offer_cap_delta_mwh"],
        })
    return out


def write_csv(path: str, rows: list, fieldnames: list = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits: int = 5):
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: str, rows: list, cols: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                value = fmt(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint", default="model_best_constr.pth")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--coalition-seed", type=int, default=20260507)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--coalition-groups", nargs="*",
                        default=["MT", "renewable"])
    parser.add_argument("--max-coalition-size", type=int, default=3)
    parser.add_argument("--max-combinations-per-bucket", type=int, default=10)
    parser.add_argument("--overreport-scales", nargs="*", type=float,
                        default=[1.25])
    parser.add_argument("--underreport-scales", nargs="*", type=float,
                        default=[0.8])
    parser.add_argument("--skip-public-context-baseline", action="store_true")
    parser.add_argument("--skip-bid-opf-baseline", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoint(_ROOT, args.checkpoint))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    net = build_network_multi(
        constant_price=False,
        ctrl_min_ratio=args.ctrl_min_ratio,
    )
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed)
    types = prior.sample(args.samples, device="cpu")
    types_np = to_numpy(types)
    labels = list(net["der_labels"])
    src_types = classify_sources(net)
    coalitions = build_coalitions(labels, src_types, args)
    strategy_specs = strategy_candidates(args)
    if not coalitions:
        raise RuntimeError("No coalitions were generated. Check --coalition-groups.")

    print(f"Generated {len(coalitions)} coalition candidates.")
    print(f"Generated {len(strategy_specs)} coalition strategies.")

    postprocessor = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=True,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )

    method_evals = {}
    print("Evaluating truthful learned posted-price mechanism...")
    peer_mech = load_mechanism(
        net, run_dir, args.checkpoint, args, use_peer_bid_context=True)
    method_evals["learned_peer_posted_price"] = (
        lambda bids: evaluate_posted_price(peer_mech, net, types, bids, postprocessor),
        evaluate_posted_price(peer_mech, net, types, types, postprocessor),
        ["pre", "post"],
    )

    if not args.skip_public_context_baseline:
        print("Evaluating truthful public-context-only posted-price baseline...")
        public_mech = load_mechanism(
            net, run_dir, args.checkpoint, args, use_peer_bid_context=False)
        method_evals["learned_public_only_posted_price"] = (
            lambda bids: evaluate_posted_price(public_mech, net, types, bids, postprocessor),
            evaluate_posted_price(public_mech, net, types, types, postprocessor),
            ["pre", "post"],
        )

    if not args.skip_bid_opf_baseline:
        qp = JointQPMulti(net)
        for settlement in ("pay_as_bid", "uniform_da"):
            method = f"bid_dependent_opf_{settlement}"
            print(f"Evaluating truthful {method} baseline...")
            truth_eval = evaluate_bid_opf(qp, net, types_np, types_np, settlement)
            method_evals[method] = (
                lambda bids, settlement=settlement: evaluate_bid_opf(
                    qp, net, types_np, to_numpy(bids), settlement),
                truth_eval,
                ["cleared"],
            )

    detailed_rows = []
    total_cases = len(method_evals) * len(coalitions) * len(strategy_specs)
    case_idx = 0
    for method, (eval_fn, truth_eval, stages) in method_evals.items():
        for coalition in coalitions:
            for spec in strategy_specs:
                case_idx += 1
                if case_idx % 50 == 0 or case_idx == 1:
                    print(f"  coalition case {case_idx}/{total_cases}")
                bids = apply_strategy(types, prior, coalition["members"], spec)
                mis_eval = eval_fn(bids)
                for stage in stages:
                    detailed_rows.append(make_detail_row(
                        method, stage, coalition, spec, truth_eval, mis_eval))

    summary_rows = summarize_details(detailed_rows)
    worst_rows = worst_by_size(detailed_rows)
    headline_rows = headline_summary(worst_rows)
    curve_rows = size_curve(worst_rows)

    detailed_path = os.path.join(out_dir, "coalition_stress_detailed.csv")
    summary_path = os.path.join(out_dir, "coalition_stress_summary.csv")
    worst_path = os.path.join(out_dir, "worst_coalition_by_size.csv")
    curve_path = os.path.join(out_dir, "coalition_size_curve.csv")
    headline_path = os.path.join(out_dir, "worst_coalition_summary.csv")
    headline_md_path = os.path.join(out_dir, "worst_coalition_summary.md")
    config_path = os.path.join(out_dir, "coalition_stress_config.json")

    write_csv(detailed_path, detailed_rows, DETAIL_COLUMNS)
    write_csv(summary_path, summary_rows)
    write_csv(worst_path, worst_rows, WORST_COLUMNS)
    write_csv(curve_path, curve_rows)
    write_csv(headline_path, headline_rows, HEADLINE_COLUMNS)
    write_markdown(headline_md_path, headline_rows, HEADLINE_COLUMNS)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["coalitions"] = coalitions
        config["strategies"] = strategy_specs
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed rows       : {detailed_path}")
    print(f"Saved strategy summary    : {summary_path}")
    print(f"Saved worst by size       : {worst_path}")
    print(f"Saved size curve          : {curve_path}")
    print(f"Saved headline summary    : {headline_path}")
    print(f"Saved headline MD         : {headline_md_path}")
    print(f"Saved config              : {config_path}")
    print("\nWorst-coalition summary:")
    for row in headline_rows:
        print(
            f"  {row['method']:<36} {row['stage']:<7} "
            f"worst_regret={fmt(row['worst_regret_mean']):<10} "
            f"k={row['worst_coalition_size']} "
            f"{row['worst_coalition_group']} "
            f"{row['worst_member_labels']} "
            f"{row['worst_strategy']}({row['worst_strategy_param']})"
        )


if __name__ == "__main__":
    main()
