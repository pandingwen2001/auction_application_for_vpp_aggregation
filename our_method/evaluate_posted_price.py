#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_posted_price.py
------------------------
Evaluate the posted-price procurement pipeline:

  posted price -> DER offer cap -> preliminary OPF -> security post-process
  -> same-rho settlement -> constrained social-optimum comparison
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, _THIS_DIR)

from network.vpp_network_multi import build_network_multi       # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti    # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti          # noqa: E402
from our_method.postprocess_security import SecurityPostProcessor  # noqa: E402
from baseline.baseline_social_opt_multi import SocialOptimumMechanismMulti  # noqa: E402


TYPE_CAP_RATIO = dict(PV=0.70, WT=0.70, DG=0.80, MT=1.70, DR=0.80)


def latest_run(root: str) -> str:
    runs_dir = os.path.join(root, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"No runs directory found: {runs_dir}")
    candidates = [
        os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {runs_dir}")
    return max(candidates, key=os.path.getmtime)


def true_cost_per_der(types: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    a = types[..., 0].unsqueeze(1)
    b = types[..., 1].unsqueeze(1)
    return (a * x.pow(2) + b * x).sum(dim=1)


def classify_sources(net: dict):
    out = []
    for label, der_type in zip(net["der_labels"], net["der_type"]):
        if label.startswith("PV"):
            out.append("PV")
        elif label.startswith("WT"):
            out.append("WT")
        elif label.startswith("MT"):
            out.append("MT")
        elif der_type == "DR":
            out.append("DR")
        else:
            out.append("DG")
    return out


def mt_gap(net: dict, x: torch.Tensor):
    mt_idx = torch.tensor(net["mt_indices"], dtype=torch.long, device=x.device)
    load = torch.tensor(net["load_profile"], dtype=torch.float32, device=x.device)
    floor = float(net["ctrl_min_ratio"]) * load.view(1, -1)
    mt = x[:, :, mt_idx].sum(dim=-1)
    return torch.relu(floor - mt).sum(dim=1), mt.sum(dim=1), floor.sum()


def metric_row(name: str, net: dict, types: torch.Tensor,
               x: torch.Tensor, p: torch.Tensor, P_VPP: torch.Tensor,
               rho: torch.Tensor = None, offer_cap: torch.Tensor = None,
               post_result=None, x_social: torch.Tensor = None,
               source_types=None, price_components=None):
    def finite_abs_mean(arr) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or np.isnan(arr).all():
            return 0.0
        return float(np.nanmean(np.abs(arr)))

    def finite_abs_max(arr) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or np.isnan(arr).all():
            return 0.0
        return float(np.nanmax(np.abs(arr)))

    pi = torch.tensor(net["pi_DA_profile"], dtype=torch.float32, device=x.device)
    tc_i = true_cost_per_der(types, x)
    true_cost = tc_i.sum(dim=1)
    payment = p.sum(dim=1)
    grid_cost = (pi.view(1, -1) * P_VPP).sum(dim=1)
    gap, mt_gen, mt_floor = mt_gap(net, x)

    row = {
        "name": name,
        "procurement_cost": float((payment + grid_cost).mean()),
        "social_cost_true": float((true_cost + grid_cost).mean()),
        "der_payment": float(payment.mean()),
        "grid_cost": float(grid_cost.mean()),
        "true_der_cost": float(true_cost.mean()),
        "info_rent": float((payment - true_cost).mean()),
        "utility_min": float((p - tc_i).min()),
        "utility_mean": float((p - tc_i).mean()),
        "der_energy_mwh": float(x.sum(dim=(1, 2)).mean()),
        "grid_import_mwh": float(P_VPP.sum(dim=1).mean()),
        "mt_gen_mwh": float(mt_gen.mean()),
        "mt_floor_mwh": float(mt_floor),
        "mt_floor_gap_mwh": float(gap.mean()),
        "mt_floor_gap_max": float(gap.max()),
    }

    if rho is not None:
        row["rho_mean"] = float(rho.mean())
        row["rho_min"] = float(rho.min())
        row["rho_max"] = float(rho.max())
    if price_components is not None:
        for key in ("rho_base", "rho_type",
                    "rho_security", "rho_scarcity", "rho_peer_bid",
                    "rho_unclamped"):
            value = price_components.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu()
            row[f"{key}_mean"] = float(np.asarray(value).mean())
        unclamped = price_components.get("rho_unclamped")
        total = price_components.get("rho_total")
        if unclamped is not None and total is not None:
            if torch.is_tensor(unclamped):
                unclamped = unclamped.detach().cpu().numpy()
            if torch.is_tensor(total):
                total = total.detach().cpu().numpy()
            row["rho_projection_gap"] = float(np.abs(unclamped - total).mean())
    if offer_cap is not None:
        mt_idx = torch.tensor(net["mt_indices"], dtype=torch.long, device=x.device)
        mt_offer = offer_cap[:, :, mt_idx].sum(dim=-1)
        floor = float(net["ctrl_min_ratio"]) * torch.tensor(
            net["load_profile"], dtype=torch.float32, device=x.device).view(1, -1)
        row["offer_cap_mwh"] = float(offer_cap.sum(dim=(1, 2)).mean())
        row["mt_offer_cap_mwh"] = float(mt_offer.sum(dim=1).mean())
        row["mt_offer_gap_mwh"] = float(torch.relu(floor - mt_offer).sum(dim=1).mean())
    if post_result is not None:
        pos_adj = torch.tensor(post_result.positive_adjustment, dtype=torch.float32)
        row["positive_adjustment_mwh"] = float(pos_adj.sum(dim=(1, 2)).mean())
        if rho is not None:
            row["positive_adjustment_payment"] = float((rho * pos_adj).sum(dim=(1, 2)).mean())
        row["postprocess_mt_slack_mwh"] = float(np.nansum(post_result.mt_slack, axis=1).mean())
        row["postprocess_status"] = ";".join(
            f"{k}:{v}" for k, v in sorted(Counter(post_result.status).items()))

        duals = getattr(post_result, "duals", {}) or {}
        if duals:
            line_dual = np.concatenate([
                np.asarray(duals.get("flow_up", [])).reshape(-1),
                np.asarray(duals.get("flow_down", [])).reshape(-1),
            ])
            voltage_dual = np.concatenate([
                np.asarray(duals.get("voltage_up", [])).reshape(-1),
                np.asarray(duals.get("voltage_down", [])).reshape(-1),
            ])
            row["post_balance_dual_abs_mean"] = finite_abs_mean(duals.get("balance", []))
            row["post_mt_floor_dual_abs_mean"] = finite_abs_mean(duals.get("mt_floor", []))
            row["post_mt_floor_dual_abs_max"] = finite_abs_max(duals.get("mt_floor", []))
            row["post_line_dual_abs_mean"] = finite_abs_mean(line_dual)
            row["post_line_dual_abs_max"] = finite_abs_max(line_dual)
            row["post_voltage_dual_abs_mean"] = finite_abs_mean(voltage_dual)
            row["post_voltage_dual_abs_max"] = finite_abs_max(voltage_dual)
            row["post_der_cap_dual_abs_mean"] = finite_abs_mean(duals.get("der_cap", []))
            row["post_der_cap_dual_abs_max"] = finite_abs_max(duals.get("der_cap", []))

        summary = getattr(post_result, "correction_summary", {}) or {}
        by_type = summary.get("by_type", {})
        for typ, values in by_type.items():
            for key, value in values.items():
                row[f"{typ}_{key}"] = float(value)

    if x_social is not None:
        diff = (x - x_social).abs()
        row["dispatch_l1_gap_mwh"] = float(diff.sum(dim=(1, 2)).mean())
        row["dispatch_mae"] = float(diff.mean())

    if source_types is not None:
        for typ in ("PV", "WT", "MT"):
            mask = torch.tensor([s == typ for s in source_types],
                                dtype=torch.bool, device=x.device)
            if not mask.any():
                continue
            row[f"{typ}_energy_mwh"] = float(x[:, :, mask].sum(dim=(1, 2)).mean())
            row[f"{typ}_true_cost"] = float(tc_i[:, mask].sum(dim=1).mean())
            row[f"{typ}_payment"] = float(p[:, mask].sum(dim=1).mean())
            if rho is not None:
                row[f"{typ}_rho_mean"] = float(rho[:, :, mask].mean())
            if x_social is not None:
                row[f"{typ}_dispatch_l1_gap_mwh"] = float(
                    (x[:, :, mask] - x_social[:, :, mask]).abs().sum(dim=(1, 2)).mean())

    return row


def load_state(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_checkpoint(run_dir: str, ckpt_name: str, args):
    net = build_network_multi(constant_price=False, ctrl_min_ratio=args.ctrl_min_ratio)
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed)
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
    ckpt_path = os.path.join(run_dir, ckpt_name)
    mech.load_state_dict(load_state(ckpt_path), strict=False)
    mech.eval()

    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()
        price_components = getattr(mech, "_last_price_components_detached", None)

    pp = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=not args.no_mt_security_uplift,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
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

    social = SocialOptimumMechanismMulti(net)
    social.eval()
    with torch.no_grad():
        x_soc, _, p_soc, P_soc = social(types)

    rows = [
        metric_row(f"{ckpt_name}:pre", net, types, x_pre, p_pre, P_pre,
                   rho=rho, offer_cap=offer_cap, x_social=x_soc,
                   source_types=source_types,
                   price_components=price_components),
        metric_row(f"{ckpt_name}:post", net, types, x_post, p_post, P_post,
                   rho=rho, offer_cap=offer_cap, post_result=post,
                   x_social=x_soc, source_types=source_types,
                   price_components=price_components),
        metric_row("constrained_social_opt", net, types, x_soc, p_soc, P_soc,
                   x_social=x_soc, source_types=source_types),
    ]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run directory. Defaults to latest original/runs/*.")
    parser.add_argument("--checkpoints", nargs="*", default=[
        "model_best_loss.pth", "model_best_constr.pth", "model_best.pth",
        "final_model.pth",
    ])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--disable-peer-bid-context", action="store_true",
                        help="Use the old public-context-only posted price.")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--no-mt-security-uplift", action="store_true",
                        help="Do not allow MT correction beyond accepted offer cap.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run) if args.run else latest_run(_ROOT)
    rows = []
    for ckpt in args.checkpoints:
        if not os.path.exists(os.path.join(run_dir, ckpt)):
            continue
        rows.extend(evaluate_checkpoint(run_dir, ckpt, args))

    if not rows:
        raise FileNotFoundError(f"No requested checkpoints found in {run_dir}")

    fieldnames = sorted({k for row in rows for k in row})
    out_path = args.out or os.path.join(run_dir, "eval_postprocess.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Run: {run_dir}")
    print(f"Saved: {out_path}")
    key_cols = [
        "name", "procurement_cost", "social_cost_true", "info_rent",
        "mt_offer_gap_mwh", "mt_floor_gap_mwh", "positive_adjustment_mwh",
        "postprocess_mt_slack_mwh", "dispatch_l1_gap_mwh",
    ]
    for row in rows:
        print("\n" + row["name"])
        for key in key_cols[1:]:
            if key in row:
                print(f"  {key}: {row[key]:.6f}" if isinstance(row[key], float)
                      else f"  {key}: {row[key]}")
        if "postprocess_status" in row:
            print(f"  status: {row['postprocess_status']}")


if __name__ == "__main__":
    main()
