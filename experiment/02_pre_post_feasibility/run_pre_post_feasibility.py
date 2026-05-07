#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: Pre/Post Feasibility
----------------------------------
Compare preliminary posted-price dispatch, security postprocess with MT
security uplift, and security postprocess without MT security uplift.

The purpose is to show:
  1. why the postprocess layer is needed,
  2. whether postprocess actually restores physical feasibility,
  3. how much correction burden remains,
  4. whether MT security uplift is essential for the MT-floor constraint.
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
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "our_method"))

from network.opf_layer_multi import DC3OPFLayerMulti  # noqa: E402
from network.vpp_network_multi import build_network_multi  # noqa: E402
from our_method.evaluate_posted_price import (  # noqa: E402
    TYPE_CAP_RATIO,
    latest_run,
    load_state,
)
from our_method.postprocess_security import SecurityPostProcessor  # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti  # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti  # noqa: E402


SUMMARY_COLUMNS = [
    "name",
    "stage",
    "postprocess_mode",
    "feasible_flag",
    "mt_floor_gap_mwh",
    "mt_floor_gap_max_mwh",
    "postprocess_mt_slack_mwh",
    "line_violation_max_mw",
    "voltage_violation_max_pu",
    "physical_cap_violation_max_mw",
    "non_mt_offer_violation_max_mw",
    "power_balance_residual_max_mw",
    "mt_offer_gap_mwh",
    "mt_security_uplift_mwh",
    "positive_adjustment_mwh",
    "negative_adjustment_mwh",
    "correction_l1_mwh",
    "positive_adjustment_payment",
    "procurement_cost",
    "social_cost_true",
    "info_rent",
    "utility_min",
    "postprocess_status",
]


class FixedPostedPriceMechanism(torch.nn.Module):
    """Non-learned posted-price baseline used as a feasibility stress row."""

    def __init__(self, net: dict, ratio: float,
                 pi_buyback_ratio: float = 0.1,
                 type_cap_ratio: dict = None):
        super().__init__()
        self.net = net
        self.T = int(net["T"])
        self.N = int(net["n_ders"])
        self.ratio = float(ratio)
        self.dc3_opf = DC3OPFLayerMulti(net)
        self.register_buffer(
            "x_bar_profile",
            torch.tensor(net["x_bar_profile"], dtype=torch.float32),
            persistent=False,
        )

        type_cap_ratio = type_cap_ratio or TYPE_CAP_RATIO
        type_names = []
        for label, der_type in zip(net["der_labels"], net["der_type"]):
            if label.startswith("PV"):
                type_names.append("PV")
            elif label.startswith("WT"):
                type_names.append("WT")
            elif label.startswith("MT"):
                type_names.append("MT")
            elif der_type == "DR":
                type_names.append("DR")
            else:
                type_names.append("DG")

        cap_ratio = np.asarray([type_cap_ratio[t] for t in type_names],
                               dtype=np.float32)
        pi = np.asarray(net["pi_DA_profile"], dtype=np.float32)
        floor = float(pi_buyback_ratio) * pi[:, None]
        cap = pi[:, None] * cap_ratio[None, :]
        rho = np.minimum(np.maximum(self.ratio * pi[:, None], floor), cap)
        self.register_buffer(
            "rho_fixed",
            torch.tensor(rho, dtype=torch.float32),
            persistent=False,
        )

    def _accepted_supply_cap(self, bids: torch.Tensor,
                             rho: torch.Tensor) -> torch.Tensor:
        a = bids[..., 0].unsqueeze(1).clamp(min=1e-6)
        b = bids[..., 1].unsqueeze(1)
        q = ((rho - b) / (2.0 * a)).clamp(min=0.0)
        return torch.min(q, self.x_bar_profile.unsqueeze(0))

    def forward(self, bids: torch.Tensor):
        B = bids.shape[0]
        rho = self.rho_fixed.to(bids.device).unsqueeze(0).expand(B, -1, -1)
        offer_cap = self._accepted_supply_cap(bids, rho)
        x, P = self.dc3_opf(rho, supply_cap=offer_cap)
        p = (rho * x).sum(dim=1)
        self._last_offer_cap = offer_cap
        return x, rho, p, P


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


def load_learned_mechanism(net: dict, run_dir: str, checkpoint: str,
                           args, use_peer_bid_context: bool):
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
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def true_cost_per_der(types_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
    a = types_np[:, None, :, 0]
    b = types_np[:, None, :, 1]
    return (a * x_np ** 2 + b * x_np).sum(axis=1)


def economics(net: dict, types_np: np.ndarray, x_np: np.ndarray,
              rho_np: np.ndarray, P_np: np.ndarray) -> dict:
    p_np = (rho_np * x_np).sum(axis=1)
    cost_i = true_cost_per_der(types_np, x_np)
    payment = p_np.sum(axis=1)
    true_cost = cost_i.sum(axis=1)
    grid_cost = (np.asarray(net["pi_DA_profile"])[None, :] * P_np).sum(axis=1)
    utility = p_np - cost_i
    return {
        "procurement_cost": float((payment + grid_cost).mean()),
        "social_cost_true": float((true_cost + grid_cost).mean()),
        "der_payment": float(payment.mean()),
        "grid_cost": float(grid_cost.mean()),
        "true_der_cost": float(true_cost.mean()),
        "info_rent": float((payment - true_cost).mean()),
        "utility_min": float(utility.min()),
        "utility_mean": float(utility.mean()),
    }


def security_metrics(net: dict, x_np: np.ndarray, P_np: np.ndarray,
                     offer_cap_np: np.ndarray, ess_net_np: np.ndarray,
                     x_ref_np: np.ndarray = None, rho_np: np.ndarray = None,
                     mt_slack_np: np.ndarray = None,
                     statuses: list = None,
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
    non_mt_idx = np.asarray([i for i in range(x_np.shape[2])
                             if i not in set(mt_idx.tolist())],
                            dtype=np.int64)

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
    }
    if rho_np is not None:
        out["positive_adjustment_payment"] = float(
            (rho_np * pos_adj).sum(axis=(1, 2)).mean())
    if statuses is not None:
        out["postprocess_status"] = ";".join(
            f"{k}:{v}" for k, v in sorted(Counter(statuses).items()))
    return out


def per_time_rows(method_name: str, stage: str, mode: str, net: dict,
                  x_np: np.ndarray, P_np: np.ndarray,
                  offer_cap_np: np.ndarray, ess_net_np: np.ndarray,
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
            "name": method_name,
            "stage": stage,
            "postprocess_mode": mode,
            "hour": t,
            "load_mw": float(load[t]),
            "mt_floor_mw": float(floor[t]),
            "mt_dispatch_mw": float(mt_dispatch[:, t].mean()),
            "mt_offer_mw": float(mt_offer[:, t].mean()),
            "mt_floor_gap_mw": float(np.maximum(floor[t] - mt_dispatch[:, t], 0.0).mean()),
            "positive_adjustment_mwh": float(pos_adj[:, t, :].sum(axis=1).mean()),
            "der_dispatch_mw": float(x_np[:, t, :].sum(axis=1).mean()),
            "grid_import_mw": float(P_np[:, t].mean()),
            "line_violation_max_mw": float(flow_viol[:, t, :].max()),
            "voltage_violation_max_pu": float(volt_viol[:, t, :].max()),
            "power_balance_residual_max_mw": float(np.abs(balance[:, t]).max()),
        })
    return rows


def evaluate_method(name: str, mech, net: dict, types: torch.Tensor,
                    pp_uplift: SecurityPostProcessor,
                    pp_no_uplift: SecurityPostProcessor,
                    checkpoint: str = "", run_dir: str = ""):
    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()

    dc3 = mech.dc3_opf
    ess_pre = None
    if getattr(dc3, "_last_P_d", None) is not None:
        ess_pre = to_numpy(dc3._last_P_d - dc3._last_P_c)

    types_np = to_numpy(types)
    x_pre_np = to_numpy(x_pre)
    rho_np = to_numpy(rho)
    P_pre_np = to_numpy(P_pre)
    offer_np = to_numpy(offer_cap)

    rows = []
    time_rows = []

    pre = {}
    pre.update(economics(net, types_np, x_pre_np, rho_np, P_pre_np))
    pre.update(security_metrics(
        net, x_pre_np, P_pre_np, offer_np, ess_pre,
        rho_np=rho_np, statuses=["preliminary_dc3"] * x_pre_np.shape[0]))
    pre.update({
        "name": name,
        "stage": "pre",
        "postprocess_mode": "none",
        "checkpoint": checkpoint,
        "run_dir": run_dir,
    })
    rows.append(pre)
    time_rows.extend(per_time_rows(
        name, "pre", "none", net, x_pre_np, P_pre_np, offer_np, ess_pre))

    for mode, pp in [
        ("mt_uplift_enabled", pp_uplift),
        ("no_mt_uplift", pp_no_uplift),
    ]:
        post = pp.process_batch(x_pre_np, P_pre_np, rho_np, offer_np)
        x_post_np = post.x.astype(np.float64)
        P_post_np = post.P_VPP.astype(np.float64)
        ess_post = None
        if post.P_d is not None and post.P_c is not None:
            ess_post = post.P_d.astype(np.float64) - post.P_c.astype(np.float64)
        row = {}
        row.update(economics(net, types_np, x_post_np, rho_np, P_post_np))
        row.update(security_metrics(
            net, x_post_np, P_post_np, offer_np, ess_post,
            x_ref_np=x_pre_np, rho_np=rho_np,
            mt_slack_np=post.mt_slack, statuses=post.status))
        row.update({
            "name": name,
            "stage": "post",
            "postprocess_mode": mode,
            "checkpoint": checkpoint,
            "run_dir": run_dir,
        })
        rows.append(row)
        time_rows.extend(per_time_rows(
            name, "post", mode, net, x_post_np, P_post_np,
            offer_np, ess_post, x_ref_np=x_pre_np))

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run directory. Defaults to latest run containing a requested checkpoint.")
    parser.add_argument("--checkpoints", nargs="*", default=[
        "model_best_constr.pth",
        "model_best_loss.pth",
        "model_best.pth",
        "final_model.pth",
    ])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--skip-public-context-ablation", action="store_true")
    parser.add_argument("--fixed-price-ratios", nargs="*", type=float,
                        default=[0.7])
    parser.add_argument("--skip-fixed-price", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--feasibility-tol", type=float, default=1e-4)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoints(_ROOT, args.checkpoints))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    net = build_network_multi(
        constant_price=False,
        ctrl_min_ratio=args.ctrl_min_ratio,
    )
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed)
    types = prior.sample(args.samples, device="cpu")

    pp_uplift = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=True,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )
    pp_no_uplift = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=False,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )

    rows = []
    time_rows = []

    if not args.skip_fixed_price:
        for ratio in args.fixed_price_ratios:
            print(f"Evaluating fixed posted price ratio={ratio:.3f}...")
            fixed = FixedPostedPriceMechanism(
                net, ratio=ratio,
                pi_buyback_ratio=args.pi_buyback_ratio,
                type_cap_ratio=TYPE_CAP_RATIO,
            )
            r, t = evaluate_method(
                f"fixed_price_ratio_{ratio:.2f}",
                fixed, net, types, pp_uplift, pp_no_uplift)
            rows.extend(r)
            time_rows.extend(t)

    for ckpt in args.checkpoints:
        ckpt_path = os.path.join(run_dir, ckpt)
        if not os.path.exists(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        print(f"Evaluating learned posted price with peer context: {ckpt}")
        mech = load_learned_mechanism(
            net, run_dir, ckpt, args, use_peer_bid_context=True)
        r, t = evaluate_method(
            f"learned_peer_{ckpt}",
            mech, net, types, pp_uplift, pp_no_uplift,
            checkpoint=ckpt, run_dir=run_dir)
        rows.extend(r)
        time_rows.extend(t)

        if not args.skip_public_context_ablation:
            print(f"Evaluating public-context-only ablation: {ckpt}")
            no_peer = load_learned_mechanism(
                net, run_dir, ckpt, args, use_peer_bid_context=False)
            r, t = evaluate_method(
                f"learned_public_only_{ckpt}",
                no_peer, net, types, pp_uplift, pp_no_uplift,
                checkpoint=ckpt, run_dir=run_dir)
            rows.extend(r)
            time_rows.extend(t)

    summary_csv = os.path.join(out_dir, "pre_post_feasibility_summary.csv")
    summary_md = os.path.join(out_dir, "pre_post_feasibility_summary.md")
    time_csv = os.path.join(out_dir, "pre_post_feasibility_by_time.csv")
    config_path = os.path.join(out_dir, "pre_post_feasibility_config.json")
    write_csv(summary_csv, rows)
    write_markdown_table(summary_md, rows)
    write_csv(time_csv, time_rows)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        json.dump(config, f, indent=2)

    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary MD : {summary_md}")
    print(f"Saved by-time CSV: {time_csv}")
    print(f"Saved config     : {config_path}")
    print("\nFeasibility rows:")
    for row in rows:
        print(
            f"  {row['name']:<42} {row['stage']:<4} "
            f"{row['postprocess_mode']:<18} "
            f"feas={row['feasible_flag']} "
            f"MTgap={fmt_value(row['mt_floor_gap_mwh']):<10} "
            f"slack={fmt_value(row['postprocess_mt_slack_mwh']):<10} "
            f"line={fmt_value(row['line_violation_max_mw']):<10} "
            f"volt={fmt_value(row['voltage_violation_max_pu']):<10} "
            f"corr={fmt_value(row['positive_adjustment_mwh'])}"
        )


if __name__ == "__main__":
    main()
