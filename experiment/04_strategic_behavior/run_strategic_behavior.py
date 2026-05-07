#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 4: Strategic Behavior Stress Test
--------------------------------------------
Evaluate unilateral strategic bid deviations for the posted-price mechanism
and bid-dependent OPF settlement baselines.

For each DER, this script changes only that DER's report while keeping the
true type fixed for utility calculation. It reports pre/post utility gains,
system-cost changes, and price-manipulation diagnostics.
"""

import argparse
import csv
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


BEST_RESPONSE_COLUMNS = [
    "method",
    "stage",
    "der_idx",
    "der_label",
    "source_type",
    "strategy",
    "strategy_param",
    "regret_mean",
    "regret_max_sample",
    "utility_truth_mean",
    "utility_misreport_mean",
    "procurement_delta",
    "info_rent_delta",
    "mt_floor_gap_delta",
    "positive_adjustment_delta",
    "own_rho_delta_max",
    "other_rho_delta_mean",
]

BEST_RESPONSE_SUMMARY_COLUMNS = [
    "method",
    "stage",
    "n_der",
    "regret_mean_across_der",
    "regret_max_der",
    "regret_max_sample",
    "worst_der_idx",
    "worst_der_label",
    "worst_der_type",
    "worst_strategy",
    "worst_strategy_param",
    "procurement_delta_at_worst",
    "info_rent_delta_at_worst",
    "mt_floor_gap_delta_at_worst",
    "positive_adjustment_delta_at_worst",
]


def source_type(label: str, der_type: str) -> str:
    if label.startswith("PV"):
        return "PV"
    if label.startswith("WT"):
        return "WT"
    if label.startswith("MT"):
        return "MT"
    if der_type == "DR":
        return "DR"
    return "DG"


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
            strategy="cost_overreport",
            strategy_param=f"scale={scale:g}",
            kind="scale",
            a_scale=float(scale),
            b_scale=float(scale),
        ))
        specs.append(dict(
            strategy="linear_cost_overreport",
            strategy_param=f"b_scale={scale:g}",
            kind="scale",
            a_scale=1.0,
            b_scale=float(scale),
        ))
    for scale in args.underreport_scales:
        specs.append(dict(
            strategy="cost_underreport",
            strategy_param=f"scale={scale:g}",
            kind="scale",
            a_scale=float(scale),
            b_scale=float(scale),
        ))
    specs.append(dict(
        strategy="high_cost_withholding_proxy",
        strategy_param="bid=upper_bound",
        kind="set_hi",
    ))
    specs.append(dict(
        strategy="low_cost_quantity_pressure",
        strategy_param="bid=lower_bound",
        kind="set_lo",
    ))
    return specs


def apply_strategy(types: torch.Tensor, prior: DERTypePriorMulti,
                   der_idx: int, spec: dict) -> torch.Tensor:
    bids = types.clone()
    if spec["kind"] == "scale":
        bids[:, der_idx, 0] = bids[:, der_idx, 0] * float(spec["a_scale"])
        bids[:, der_idx, 1] = bids[:, der_idx, 1] * float(spec["b_scale"])
    elif spec["kind"] == "set_hi":
        bids[:, der_idx, :] = prior.hi[der_idx].view(1, 2)
    elif spec["kind"] == "set_lo":
        bids[:, der_idx, :] = prior.lo[der_idx].view(1, 2)
    else:
        raise ValueError(f"Unknown strategy kind: {spec['kind']}")
    return prior.project(bids)


def price_delta_row(truth_eval: dict, mis_eval: dict, der_idx: int) -> dict:
    rho_truth = truth_eval.get("rho")
    rho_mis = mis_eval.get("rho")
    if rho_truth is None or rho_mis is None:
        return {
            "own_rho_delta_max": "",
            "other_rho_delta_mean": "",
            "all_rho_delta_mean": "",
        }
    own = np.abs(rho_mis[:, :, der_idx] - rho_truth[:, :, der_idx])
    mask = np.ones(rho_truth.shape[2], dtype=bool)
    mask[der_idx] = False
    other = np.abs(rho_mis[:, :, mask] - rho_truth[:, :, mask])
    return {
        "own_rho_delta_max": float(own.max()),
        "other_rho_delta_mean": float(other.mean()),
        "all_rho_delta_mean": float(np.abs(rho_mis - rho_truth).mean()),
    }


def make_detail_row(method: str, stage: str, der_idx: int,
                    labels: list, source_types: list,
                    spec: dict, truth_eval: dict, mis_eval: dict) -> dict:
    truth = truth_eval[stage]
    mis = mis_eval[stage]
    u_truth = truth["utility"][:, der_idx]
    u_mis = mis["utility"][:, der_idx]
    gain = u_mis - u_truth
    row = {
        "method": method,
        "stage": stage,
        "der_idx": der_idx,
        "der_label": labels[der_idx],
        "source_type": source_types[der_idx],
        "strategy": spec["strategy"],
        "strategy_param": spec["strategy_param"],
        "utility_truth_mean": float(u_truth.mean()),
        "utility_misreport_mean": float(u_mis.mean()),
        "utility_gain_mean": float(gain.mean()),
        "regret_mean": float(np.maximum(gain, 0.0).mean()),
        "regret_max_sample": float(np.maximum(gain, 0.0).max()),
        "procurement_truth": truth["procurement_cost"],
        "procurement_misreport": mis["procurement_cost"],
        "procurement_delta": mis["procurement_cost"] - truth["procurement_cost"],
        "social_cost_truth": truth["social_cost_true"],
        "social_cost_misreport": mis["social_cost_true"],
        "social_cost_delta": mis["social_cost_true"] - truth["social_cost_true"],
        "info_rent_truth": truth["info_rent"],
        "info_rent_misreport": mis["info_rent"],
        "info_rent_delta": mis["info_rent"] - truth["info_rent"],
        "mt_floor_gap_truth": truth["mt_floor_gap_mwh"],
        "mt_floor_gap_misreport": mis["mt_floor_gap_mwh"],
        "mt_floor_gap_delta": mis["mt_floor_gap_mwh"] - truth["mt_floor_gap_mwh"],
        "positive_adjustment_truth": truth.get("positive_adjustment_mwh", 0.0),
        "positive_adjustment_misreport": mis.get("positive_adjustment_mwh", 0.0),
        "positive_adjustment_delta": (
            mis.get("positive_adjustment_mwh", 0.0)
            - truth.get("positive_adjustment_mwh", 0.0)
        ),
        "utility_min_misreport": mis["utility_min"],
    }
    row.update(price_delta_row(truth_eval, mis_eval, der_idx))
    return row


def summarize_details(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["strategy"], row["stage"])].append(row)
    out = []
    for (method, strategy, stage), group in sorted(grouped.items()):
        best = max(group, key=lambda r: r["regret_mean"])
        out.append({
            "method": method,
            "strategy": strategy,
            "stage": stage,
            "n_der": len(group),
            "regret_mean_across_der": float(np.mean([r["regret_mean"] for r in group])),
            "regret_max_der": float(max(r["regret_mean"] for r in group)),
            "regret_max_sample": float(max(r["regret_max_sample"] for r in group)),
            "utility_gain_mean_across_der": float(np.mean([r["utility_gain_mean"] for r in group])),
            "procurement_delta_mean": float(np.mean([r["procurement_delta"] for r in group])),
            "procurement_delta_max": float(max(r["procurement_delta"] for r in group)),
            "info_rent_delta_mean": float(np.mean([r["info_rent_delta"] for r in group])),
            "mt_floor_gap_delta_mean": float(np.mean([r["mt_floor_gap_delta"] for r in group])),
            "positive_adjustment_delta_mean": float(np.mean([r["positive_adjustment_delta"] for r in group])),
            "worst_der_idx": best["der_idx"],
            "worst_der_label": best["der_label"],
            "worst_der_type": best["source_type"],
            "worst_strategy_param": best["strategy_param"],
        })
    return out


def best_response_rows(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["stage"], row["der_idx"])].append(row)
    out = []
    for _, group in sorted(grouped.items()):
        out.append(max(group, key=lambda r: r["regret_mean"]))
    return out


def best_response_summary(best_rows: list) -> list:
    grouped = defaultdict(list)
    for row in best_rows:
        grouped[(row["method"], row["stage"])].append(row)
    out = []
    for (method, stage), group in sorted(grouped.items()):
        worst = max(group, key=lambda r: r["regret_mean"])
        out.append({
            "method": method,
            "stage": stage,
            "n_der": len(group),
            "regret_mean_across_der": float(np.mean([r["regret_mean"] for r in group])),
            "regret_max_der": float(max(r["regret_mean"] for r in group)),
            "regret_max_sample": float(max(r["regret_max_sample"] for r in group)),
            "worst_der_idx": worst["der_idx"],
            "worst_der_label": worst["der_label"],
            "worst_der_type": worst["source_type"],
            "worst_strategy": worst["strategy"],
            "worst_strategy_param": worst["strategy_param"],
            "procurement_delta_at_worst": worst["procurement_delta"],
            "info_rent_delta_at_worst": worst["info_rent_delta"],
            "mt_floor_gap_delta_at_worst": worst["mt_floor_gap_delta"],
            "positive_adjustment_delta_at_worst": worst["positive_adjustment_delta"],
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
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--overreport-scales", nargs="*", type=float,
                        default=[1.1, 1.25, 1.5])
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
    strategy_specs = strategy_candidates(args)

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
    for method, (eval_fn, truth_eval, stages) in method_evals.items():
        for spec in strategy_specs:
            for der_idx in range(len(labels)):
                bids = apply_strategy(types, prior, der_idx, spec)
                mis_eval = eval_fn(bids)
                for stage in stages:
                    detailed_rows.append(make_detail_row(
                        method, stage, der_idx, labels, src_types,
                        spec, truth_eval, mis_eval))

    summary_rows = summarize_details(detailed_rows)
    best_rows = best_response_rows(detailed_rows)
    best_summary_rows = best_response_summary(best_rows)

    detailed_path = os.path.join(out_dir, "strategic_behavior_detailed.csv")
    summary_path = os.path.join(out_dir, "strategic_behavior_summary.csv")
    best_path = os.path.join(out_dir, "best_response_by_der.csv")
    best_summary_path = os.path.join(out_dir, "best_response_summary.csv")
    best_md_path = os.path.join(out_dir, "best_response_summary.md")
    config_path = os.path.join(out_dir, "strategic_behavior_config.json")
    write_csv(detailed_path, detailed_rows)
    write_csv(summary_path, summary_rows)
    write_csv(best_path, best_rows, BEST_RESPONSE_COLUMNS)
    write_csv(best_summary_path, best_summary_rows, BEST_RESPONSE_SUMMARY_COLUMNS)
    write_markdown(best_md_path, best_summary_rows, BEST_RESPONSE_SUMMARY_COLUMNS)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["strategies"] = strategy_specs
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed rows       : {detailed_path}")
    print(f"Saved strategy summary    : {summary_path}")
    print(f"Saved best response rows  : {best_path}")
    print(f"Saved best response summary: {best_summary_path}")
    print(f"Saved best response MD    : {best_md_path}")
    print(f"Saved config              : {config_path}")
    print("\nBest-response summary:")
    for row in best_summary_rows:
        print(
            f"  {row['method']:<36} {row['stage']:<7} "
            f"regret_mean={fmt(row['regret_mean_across_der']):<10} "
            f"regret_max={fmt(row['regret_max_der']):<10} "
            f"worst={row['worst_der_label']} "
            f"{row['worst_strategy']}({row['worst_strategy_param']})"
        )


if __name__ == "__main__":
    main()
