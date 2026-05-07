#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_correction_feedback.py
-------------------------------
Build a correction-feedback dataset from a trained posted-price checkpoint.

The dataset is the Stage-2 bridge for security-correction-aware learning:

  checkpoint -> posted price / preliminary dispatch
  -> security postprocess
  -> primal correction arrays + QP dual signals

Outputs:
  * .npz with sample-level arrays for fine-tuning
  * *_by_time.csv with hourly correction and dual summaries
  * *_by_type.csv with DER-type correction summaries
  * *_summary.csv with one-row headline metrics
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
from our_method.evaluate_posted_price import (                  # noqa: E402
    TYPE_CAP_RATIO,
    classify_sources,
    latest_run,
    load_state,
)


DUAL_KEYS = (
    "balance",
    "flow_up",
    "flow_down",
    "voltage_up",
    "voltage_down",
    "der_cap",
    "mt_floor",
    "ess_charge_cap",
    "ess_discharge_cap",
)

PRICE_COMPONENT_KEYS = (
    "rho_base",
    "rho_type",
    "rho_security_main",
    "rho_scarcity_main",
    "rho_security_residual",
    "rho_scarcity_residual",
    "rho_security",
    "rho_scarcity",
    "rho_peer_bid",
    "rho_unclamped",
    "rho_total",
)


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


def batch_ranges(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def default_output_path(run_dir: str, checkpoint: str, samples: int) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    return os.path.join(run_dir, f"correction_feedback_{stem}_n{samples}.npz")


def build_dataset(args):
    run_dir = os.path.abspath(args.run) if args.run else latest_run(_ROOT)
    ckpt_path = os.path.join(run_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    net = build_network_multi(
        constant_price=False,
        ctrl_min_ratio=args.ctrl_min_ratio,
    )
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed)
    types_all = prior.sample(args.samples, device="cpu")

    mech = VPPMechanismMulti(
        net,
        posted_price_cfg=dict(
            price_arch=args.price_arch,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_dropout=args.transformer_dropout,
            pi_buyback_ratio=args.pi_buyback_ratio,
            use_peer_bid_context=not args.disable_peer_bid_context,
            peer_bid_scale=args.peer_bid_scale,
            type_cap_ratio=TYPE_CAP_RATIO,
        ),
    )
    mech.load_state_dict(load_state(ckpt_path), strict=False)
    mech.eval()

    postprocessor = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=not args.no_mt_security_uplift,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )

    arrays = {
        "types": [],
        "rho": [],
        "x_pre": [],
        "P_pre": [],
        "p_pre": [],
        "offer_cap": [],
        "x_post": [],
        "P_post": [],
        "p_post": [],
        "positive_adjustment": [],
        "mt_slack": [],
    }
    for key in DUAL_KEYS:
        arrays[f"dual_{key}"] = []
    for key in PRICE_COMPONENT_KEYS:
        arrays[key] = []
    statuses = []

    with torch.no_grad():
        for start, end in batch_ranges(args.samples, args.batch_size):
            types = types_all[start:end]
            x_pre, rho, p_pre, P_pre = mech(types)
            offer_cap = mech._last_offer_cap.detach().clone()
            price_components = getattr(mech, "_last_price_components_detached", {})

            rho_np = rho.detach().cpu().numpy()
            post = postprocessor.process_batch(
                x_pre.detach().cpu().numpy(),
                P_pre.detach().cpu().numpy(),
                rho_np,
                offer_cap.detach().cpu().numpy(),
            )
            p_post = (rho_np * post.x).sum(axis=1)

            arrays["types"].append(types.detach().cpu().numpy())
            arrays["rho"].append(rho_np)
            arrays["x_pre"].append(x_pre.detach().cpu().numpy())
            arrays["P_pre"].append(P_pre.detach().cpu().numpy())
            arrays["p_pre"].append(p_pre.detach().cpu().numpy())
            arrays["offer_cap"].append(offer_cap.detach().cpu().numpy())
            arrays["x_post"].append(post.x)
            arrays["P_post"].append(post.P_VPP)
            arrays["p_post"].append(p_post.astype(np.float32))
            arrays["positive_adjustment"].append(post.positive_adjustment)
            arrays["mt_slack"].append(post.mt_slack)
            for key in DUAL_KEYS:
                arrays[f"dual_{key}"].append(post.duals[key])
            for key in PRICE_COMPONENT_KEYS:
                value = price_components[key]
                arrays[key].append(value.detach().cpu().numpy())
            statuses.extend(post.status)

    data = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    data["status"] = np.asarray(statuses)
    data["der_labels"] = np.asarray(net["der_labels"])
    data["source_types"] = np.asarray(classify_sources(net))
    data["load_profile"] = np.asarray(net["load_profile"], dtype=np.float32)
    data["pi_DA_profile"] = np.asarray(net["pi_DA_profile"], dtype=np.float32)
    data["mt_indices"] = np.asarray(net["mt_indices"], dtype=np.int64)
    data["run_dir"] = np.asarray(run_dir)
    data["checkpoint"] = np.asarray(args.checkpoint)
    return run_dir, data, Counter(statuses)


def write_by_time_csv(path: str, data: dict):
    pos = data["positive_adjustment"]
    neg = np.maximum(data["x_pre"] - data["x_post"], 0.0)
    delta_x = data["x_post"] - data["x_pre"]
    delta_p = data["P_post"] - data["P_pre"]

    flow_dual = np.abs(data["dual_flow_up"]) + np.abs(data["dual_flow_down"])
    voltage_dual = np.abs(data["dual_voltage_up"]) + np.abs(data["dual_voltage_down"])

    rows = []
    T = pos.shape[1]
    for t in range(T):
        rows.append({
            "hour": t,
            "positive_adjustment_mwh": float(pos[:, t, :].sum(axis=1).mean()),
            "negative_adjustment_mwh": float(neg[:, t, :].sum(axis=1).mean()),
            "net_der_adjustment_mwh": float(delta_x[:, t, :].sum(axis=1).mean()),
            "pvpp_adjustment_mwh": float(delta_p[:, t].mean()),
            "mt_slack_mwh": float(np.nanmean(data["mt_slack"][:, t])),
            "balance_dual_abs_mean": finite_abs_mean(data["dual_balance"][:, t]),
            "mt_floor_dual_abs_mean": finite_abs_mean(data["dual_mt_floor"][:, t]),
            "line_dual_abs_mean": finite_abs_mean(flow_dual[:, t, :]),
            "line_dual_abs_max": finite_abs_max(flow_dual[:, t, :]),
            "voltage_dual_abs_mean": finite_abs_mean(voltage_dual[:, t, :]),
            "voltage_dual_abs_max": finite_abs_max(voltage_dual[:, t, :]),
            "rho_base_mean": float(data["rho_base"][:, t, :].mean()),
            "rho_type_mean": float(data["rho_type"][:, t, :].mean()),
            "rho_security_mean": float(data["rho_security"][:, t, :].mean()),
            "rho_scarcity_mean": float(data["rho_scarcity"][:, t, :].mean()),
        })
    write_rows(path, rows)


def write_by_type_csv(path: str, data: dict):
    source_types = data["source_types"]
    pos = data["positive_adjustment"]
    neg = np.maximum(data["x_pre"] - data["x_post"], 0.0)
    delta = data["x_post"] - data["x_pre"]
    rows = []
    for typ in ("PV", "WT", "DG", "MT", "DR"):
        mask = source_types == typ
        if not mask.any():
            continue
        uplift = np.maximum(data["x_post"][:, :, mask] - data["offer_cap"][:, :, mask], 0.0)
        rows.append({
            "type": typ,
            "positive_adjustment_mwh": float(pos[:, :, mask].sum(axis=(1, 2)).mean()),
            "negative_adjustment_mwh": float(neg[:, :, mask].sum(axis=(1, 2)).mean()),
            "net_adjustment_mwh": float(delta[:, :, mask].sum(axis=(1, 2)).mean()),
            "additional_payment": float(
                (data["rho"][:, :, mask] * pos[:, :, mask]).sum(axis=(1, 2)).mean()),
            "security_uplift_mwh": float(uplift.sum(axis=(1, 2)).mean()),
            "security_uplift_payment": float(
                (data["rho"][:, :, mask] * uplift).sum(axis=(1, 2)).mean()),
        })
    write_rows(path, rows)


def write_summary_csv(path: str, data: dict, status_counts: Counter):
    line_dual = np.concatenate([
        data["dual_flow_up"].reshape(-1),
        data["dual_flow_down"].reshape(-1),
    ])
    voltage_dual = np.concatenate([
        data["dual_voltage_up"].reshape(-1),
        data["dual_voltage_down"].reshape(-1),
    ])
    pos = data["positive_adjustment"]
    neg = np.maximum(data["x_pre"] - data["x_post"], 0.0)
    row = {
        "samples": int(data["types"].shape[0]),
        "status": ";".join(f"{k}:{v}" for k, v in sorted(status_counts.items())),
        "positive_adjustment_mwh": float(pos.sum(axis=(1, 2)).mean()),
        "negative_adjustment_mwh": float(neg.sum(axis=(1, 2)).mean()),
        "additional_payment": float((data["rho"] * pos).sum(axis=(1, 2)).mean()),
        "mt_slack_mwh": float(np.nansum(data["mt_slack"], axis=1).mean()),
        "balance_dual_abs_mean": finite_abs_mean(data["dual_balance"]),
        "mt_floor_dual_abs_mean": finite_abs_mean(data["dual_mt_floor"]),
        "mt_floor_dual_abs_max": finite_abs_max(data["dual_mt_floor"]),
        "line_dual_abs_max": finite_abs_max(line_dual),
        "voltage_dual_abs_max": finite_abs_max(voltage_dual),
        "der_cap_dual_abs_max": finite_abs_max(data["dual_der_cap"]),
    }
    for key in PRICE_COMPONENT_KEYS:
        row[f"{key}_mean"] = float(data[key].mean())
    row["rho_projection_gap"] = float(
        np.abs(data["rho_unclamped"] - data["rho_total"]).mean())
    write_rows(path, [row])


def write_rows(path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def companion_path(npz_path: str, suffix: str) -> str:
    root, _ = os.path.splitext(npz_path)
    return f"{root}_{suffix}.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run directory. Defaults to latest original/runs/*.")
    parser.add_argument("--checkpoint", default="model_best_loss.pth")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--disable-peer-bid-context", action="store_true",
                        help="Use the old public-context-only posted price.")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--no-mt-security-uplift", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir, data, status_counts = build_dataset(args)
    out_path = args.out or default_output_path(run_dir, args.checkpoint, args.samples)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **data)

    by_time_path = companion_path(out_path, "by_time")
    by_type_path = companion_path(out_path, "by_type")
    summary_path = companion_path(out_path, "summary")
    write_by_time_csv(by_time_path, data)
    write_by_type_csv(by_type_path, data)
    write_summary_csv(summary_path, data, status_counts)

    print(f"Saved dataset: {out_path}")
    print(f"Saved by-time summary: {by_time_path}")
    print(f"Saved by-type summary: {by_type_path}")
    print(f"Saved headline summary: {summary_path}")
    print("Status:", ";".join(f"{k}:{v}" for k, v in sorted(status_counts.items())))


if __name__ == "__main__":
    main()
