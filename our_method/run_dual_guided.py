#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dual_guided.py
------------------
Stage-3 fine-tuning for the structured posted-price mechanism.

This script keeps the mechanism bid-independent.  OPF postprocess duals are
used only as teacher signals for the decomposed public-context price heads:

  voltage/line dual marginal signal -> rho_security
  MT floor dual / MT correction      -> rho_scarcity

By default, the legacy type/context price head is frozen and only the
security/scarcity heads are updated.  A small anchor and component
regularization keep the decomposition identifiable without dominating the
dual-guided signal.
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, _THIS_DIR)

from network.vpp_network_multi import build_network_multi       # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti    # noqa: E402
from our_method.evaluate_posted_price import (                  # noqa: E402
    TYPE_CAP_RATIO,
    evaluate_checkpoint,
    latest_run,
    load_state,
)
from our_method.generate_correction_feedback import build_dataset  # noqa: E402


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_torch(data: dict, device: torch.device) -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
            dtype = torch.long if value.dtype.kind in "iu" else torch.float32
            tensor = torch.tensor(value, dtype=dtype, device=device)
            if dtype.is_floating_point:
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            out[key] = tensor
    return out


def normalize_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.quantile(x.abs().reshape(-1), 0.95).clamp(min=eps)
    return x / denom


def build_dual_targets(net: dict, data_t: dict,
                       security_scale: float,
                       scarcity_scale: float,
                       device: torch.device) -> dict:
    """
    Return signed residual price-adder targets [T,N] in $/MWh.

    security residual target:
      signed negative marginal congestion cost.  If extra DER injection worsens
      active voltage/line constraints, target is negative; if it relieves them,
      target is positive.

    scarcity residual target:
      positive MT-only signal from MT floor duals and observed MT corrections.
    """
    A_flow = torch.tensor(net["A_flow"], dtype=torch.float32, device=device)
    A_volt = torch.tensor(net["A_volt"], dtype=torch.float32, device=device)
    mt_idx = torch.tensor(net["mt_indices"], dtype=torch.long, device=device)
    B, T, N = data_t["rho"].shape

    flow_grad = (
        data_t["dual_flow_up"] @ A_flow
        - data_t["dual_flow_down"] @ A_flow
    )                                                       # [B,T,N]
    volt_grad = (
        data_t["dual_voltage_up"] @ A_volt
        - data_t["dual_voltage_down"] @ A_volt
    )                                                       # [B,T,N]
    marginal_congestion = flow_grad + volt_grad
    security_target = -marginal_congestion.mean(dim=0)      # [T,N]
    security_target = normalize_abs(security_target) * float(security_scale)
    security_target = security_target.clamp(
        min=-float(security_scale), max=float(security_scale))

    mt_dual = data_t["dual_mt_floor"].abs().mean(dim=0)     # [T]
    mt_pos = data_t["positive_adjustment"].index_select(
        -1, mt_idx).sum(dim=-1).mean(dim=0)                 # [T]
    dual_pressure = normalize_abs(mt_dual)
    pos_pressure = normalize_abs(mt_pos)
    pressure = (dual_pressure.clamp(min=0.0) + pos_pressure.clamp(min=0.0)) * 0.5
    pressure = pressure / pressure.max().clamp(min=1e-8)
    scarcity_t = pressure * float(scarcity_scale)           # [T]
    scarcity_target = torch.zeros(T, N, dtype=torch.float32, device=device)
    scarcity_target[:, mt_idx] = scarcity_t[:, None]

    return {
        "security_delta_target": security_target,
        "scarcity_delta_target": scarcity_target,
        "mt_pressure": scarcity_t,
    }


def reset_last_layer(module: torch.nn.Module):
    last = None
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            last = layer
    if last is not None:
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)


def configure_trainable_heads(mech: VPPMechanismMulti,
                              train_all_price_heads: bool,
                              reset_guided_heads: bool):
    pp = mech.posted_price_net
    for p in mech.parameters():
        p.requires_grad_(False)

    if pp.price_arch == "transformer":
        guided_modules = (pp.tr_security_residual_head,
                          pp.tr_scarcity_residual_head)
    else:
        guided_modules = (pp.security_residual_mlp,
                          pp.scarcity_residual_mlp)

    if reset_guided_heads:
        for module in guided_modules:
            reset_last_layer(module)

    if train_all_price_heads:
        for p in pp.parameters():
            p.requires_grad_(True)
    else:
        for module in guided_modules:
            for p in module.parameters():
                p.requires_grad_(True)


def mt_offer_gap_from_last_forward(mech: VPPMechanismMulti) -> torch.Tensor:
    offer_cap = mech._last_offer_cap
    mt_idx = torch.tensor(mech.net_multi["mt_indices"], dtype=torch.long,
                          device=offer_cap.device)
    load = torch.tensor(mech.net_multi["load_profile"], dtype=torch.float32,
                        device=offer_cap.device)
    floor = float(mech.net_multi.get("ctrl_min_ratio", 0.0)) * load.view(1, -1)
    mt_offer = offer_cap.index_select(-1, mt_idx).sum(dim=-1)
    return torch.relu(floor - mt_offer).sum(dim=1).mean()


def mt_offer_gap_from_arrays(net: dict, offer_cap: torch.Tensor) -> torch.Tensor:
    mt_idx = torch.tensor(net["mt_indices"], dtype=torch.long,
                          device=offer_cap.device)
    load = torch.tensor(net["load_profile"], dtype=torch.float32,
                        device=offer_cap.device)
    floor = float(net.get("ctrl_min_ratio", 0.0)) * load.view(1, -1)
    mt_offer = offer_cap.index_select(-1, mt_idx).sum(dim=-1)
    return torch.relu(floor - mt_offer).sum(dim=1).mean()


def loss_terms(mech: VPPMechanismMulti, types: torch.Tensor,
               rho_ref: torch.Tensor, targets: dict, cfg: dict) -> dict:
    x, rho, p, P = mech(types)
    comp = mech._last_price_components

    security_target = targets["security_delta_target"].view(1, mech.T, mech.N)
    scarcity_target = targets["scarcity_delta_target"].view(1, mech.T, mech.N)
    security_delta = comp["rho_security_residual"]
    scarcity_delta = comp["rho_scarcity_residual"]

    security_loss = F.mse_loss(security_delta, security_target.expand_as(rho))
    scarcity_loss = F.mse_loss(scarcity_delta, scarcity_target.expand_as(rho))
    anchor_loss = F.mse_loss(rho, rho_ref)
    proj_loss = (comp["rho_unclamped"] - comp["rho_total"]).abs().mean()
    add_loss = (security_delta.pow(2).mean()
                + scarcity_delta.pow(2).mean())
    mt_offer_gap = mt_offer_gap_from_last_forward(mech)
    baseline_gap = torch.tensor(float(cfg["baseline_mt_offer_gap"]),
                                dtype=torch.float32, device=rho.device)
    gap_tolerance = torch.tensor(float(cfg["mt_gap_tolerance"]),
                                 dtype=torch.float32, device=rho.device)
    mt_gap_guard = torch.relu(mt_offer_gap - baseline_gap - gap_tolerance).pow(2)

    total = (
        cfg["lambda_security"] * security_loss
        + cfg["lambda_scarcity"] * scarcity_loss
        + cfg["lambda_anchor"] * anchor_loss
        + cfg["lambda_projection"] * proj_loss
        + cfg["lambda_additive"] * add_loss
        + cfg["lambda_mt_offer_gap"] * mt_gap_guard
    )

    return {
        "loss": total,
        "security_loss": security_loss.detach(),
        "scarcity_loss": scarcity_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "projection_loss": proj_loss.detach(),
        "additive_loss": add_loss.detach(),
        "mt_offer_gap": mt_offer_gap.detach(),
        "mt_gap_guard": mt_gap_guard.detach(),
        "system_cost": mech.system_cost(p, P).detach(),
        "rho_mean": rho.detach().mean(),
        "rho_base_mean": comp["rho_base"].detach().mean(),
        "rho_type_mean": comp["rho_type"].detach().mean(),
        "rho_security_main_mean": comp["rho_security_main"].detach().mean(),
        "rho_scarcity_main_mean": comp["rho_scarcity_main"].detach().mean(),
        "rho_security_residual_mean": security_delta.detach().mean(),
        "rho_scarcity_residual_mean": scarcity_delta.detach().mean(),
        "rho_security_mean": comp["rho_security"].detach().mean(),
        "rho_scarcity_mean": comp["rho_scarcity"].detach().mean(),
        "rho_peer_bid_mean": comp.get(
            "rho_peer_bid", torch.zeros_like(rho)).detach().mean(),
        "rho_projection_gap": (comp["rho_unclamped"].detach()
                               - comp["rho_total"].detach()).abs().mean(),
    }


def write_csv(path: str, rows: list):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_union_csv(path: str, rows: list):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_targets(path: str, targets: dict):
    np.savez_compressed(
        path,
        security_delta_target=targets["security_delta_target"].detach().cpu().numpy(),
        scarcity_delta_target=targets["scarcity_delta_target"].detach().cpu().numpy(),
        mt_pressure=targets["mt_pressure"].detach().cpu().numpy(),
    )


def selection_checkpoint_candidates(out_dir: str, iters: int, save_iter: int) -> list:
    candidates = ["model_initial.pth", "model_best.pth"]
    if save_iter > 0:
        candidates.extend(f"model_{step}.pth"
                          for step in range(save_iter, iters + 1, save_iter))
    candidates.append("final_model.pth")

    out = []
    seen = set()
    for ckpt in candidates:
        if ckpt in seen:
            continue
        if os.path.exists(os.path.join(out_dir, ckpt)):
            out.append(ckpt)
            seen.add(ckpt)
    return out


def row_float(row: dict, key: str) -> float:
    raw = row.get(key, 0.0)
    return 0.0 if raw in ("", None) else float(raw)


def offer_correction_metric(row: dict, args) -> float:
    """Secondary metric: lower means less postprocess correction burden."""
    return (
        row_float(row, "mt_offer_gap_mwh")
        + args.selection_offer_adjustment_weight
        * row_float(row, "positive_adjustment_mwh")
    )


def score_feasible_rent_row(row: dict, args) -> float:
    """Primary metric: postprocess MT-floor feasibility, then low rent/cost."""
    floor_gap = row_float(row, "mt_floor_gap_mwh")
    mt_slack = row_float(row, "postprocess_mt_slack_mwh")
    floor_violation = max(0.0, floor_gap - args.selection_mt_floor_tol)
    slack_violation = max(0.0, mt_slack - args.selection_mt_slack_tol)
    feasible = floor_violation <= 0.0 and slack_violation <= 0.0

    correction = offer_correction_metric(row, args)
    score = (
        args.selection_info_rent_weight * row_float(row, "info_rent")
        + args.selection_procurement_weight * row_float(row, "procurement_cost")
        + args.selection_offer_gap_weight * row_float(row, "mt_offer_gap_mwh")
        + args.selection_correction_weight * correction
        + args.selection_mt_floor_weight * floor_violation
        + args.selection_mt_slack_weight * slack_violation
    )
    row["selection_mt_floor_feasible"] = int(feasible)
    row["selection_floor_violation_mwh"] = floor_violation
    row["selection_mt_slack_violation_mwh"] = slack_violation
    row["selection_offer_correction_metric"] = correction
    row["selection_feasible_rent_score"] = score
    return score


def run_post_train_selection(out_dir: str, args):
    ckpts = selection_checkpoint_candidates(out_dir, args.iters, args.save_iter)
    if not ckpts:
        print("Postprocess selection skipped: no checkpoints found.")
        return None

    eval_args = argparse.Namespace(
        samples=args.selection_eval_samples,
        seed=args.selection_eval_seed,
        ctrl_min_ratio=args.ctrl_min_ratio,
        pi_buyback_ratio=args.pi_buyback_ratio,
        peer_bid_scale=args.peer_bid_scale,
        price_arch=args.price_arch,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        disable_peer_bid_context=args.disable_peer_bid_context,
        adjustment_weight=1.0,
        settlement_weight=1e-3,
        mt_slack_weight=1e5,
        no_mt_security_uplift=False,
    )

    rows = []
    for ckpt in ckpts:
        rows.extend(evaluate_checkpoint(out_dir, ckpt, eval_args))

    post_rows = [
        row for row in rows
        if row.get("name", "").endswith(":post")
        and row.get("name") != "constrained_social_opt"
    ]
    if not post_rows:
        print("Postprocess selection skipped: no postprocess rows found.")
        return None

    for row in post_rows:
        if "selection_feasible_rent_score" not in row:
            score_feasible_rent_row(row, args)
    eligible_rows = [
        row for row in post_rows
        if row["selection_mt_floor_feasible"]
    ]
    selection_pool = eligible_rows or post_rows
    best_row = min(selection_pool,
                   key=lambda row: row["selection_feasible_rent_score"])

    selected_ckpt = best_row["name"].split(":")[0]
    selected_path = os.path.join(out_dir, selected_ckpt)
    best_feasible_rent_path = os.path.join(out_dir, "model_best_feasible_rent.pth")
    shutil.copy2(selected_path, best_feasible_rent_path)
    alias_rows = [
        {
            **row,
            "name": row["name"].replace(
                f"{selected_ckpt}:", "model_best_feasible_rent.pth:", 1),
        }
        for row in rows
        if row.get("name", "").startswith(f"{selected_ckpt}:")
    ]
    rows.extend(alias_rows)

    eval_path = os.path.join(out_dir, "eval_postprocess.csv")
    write_union_csv(eval_path, rows)

    summary = {
        "selection_objective": "postprocess_mt_floor_feasible_low_info_rent",
        "selected_checkpoint": selected_ckpt,
        "saved_as": "model_best_feasible_rent.pth",
        "selection_eval_samples": args.selection_eval_samples,
        "selection_eval_seed": args.selection_eval_seed,
        "selection_mt_floor_tol": args.selection_mt_floor_tol,
        "selection_mt_slack_tol": args.selection_mt_slack_tol,
        "selection_info_rent_weight": args.selection_info_rent_weight,
        "selection_procurement_weight": args.selection_procurement_weight,
        "selection_offer_gap_weight": args.selection_offer_gap_weight,
        "selection_correction_weight": args.selection_correction_weight,
        "selection_offer_adjustment_weight": args.selection_offer_adjustment_weight,
        "selection_mt_floor_weight": args.selection_mt_floor_weight,
        "selection_mt_slack_weight": args.selection_mt_slack_weight,
        "eligible_checkpoint_count": len(eligible_rows),
        "selected_post_metrics": best_row,
    }
    with open(os.path.join(out_dir, "selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Selection eval saved: {eval_path}")
    print(
        "Best feasible low-rent checkpoint: "
        f"{selected_ckpt} -> model_best_feasible_rent.pth "
        f"(score={best_row['selection_feasible_rent_score']:.4f}, "
        f"MTfloor_gap={float(best_row.get('mt_floor_gap_mwh', 0.0)):.6f}, "
        f"MTslack={float(best_row.get('postprocess_mt_slack_mwh', 0.0)):.6f}, "
        f"info_rent={float(best_row.get('info_rent', 0.0)):.2f}, "
        f"cost={float(best_row.get('procurement_cost', 0.0)):.2f}, "
        f"offer_gap={float(best_row.get('mt_offer_gap_mwh', 0.0)):.4f})"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Source run directory. Defaults to latest original/runs/*.")
    parser.add_argument("--checkpoint", default="model_best_constr.pth")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--feedback-batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260427)
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
    parser.add_argument("--security-scale", type=float, default=0.5)
    parser.add_argument("--scarcity-scale", type=float, default=1.0)
    parser.add_argument("--lambda-security", type=float, default=1.0)
    parser.add_argument("--lambda-scarcity", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)
    parser.add_argument("--lambda-projection", type=float, default=0.01)
    parser.add_argument("--lambda-additive", type=float, default=0.001)
    parser.add_argument("--lambda-mt-offer-gap", type=float, default=2.0)
    parser.add_argument("--mt-gap-tolerance", type=float, default=0.25)
    parser.add_argument("--print-iter", type=int, default=50)
    parser.add_argument("--save-iter", type=int, default=250)
    parser.add_argument("--train-all-price-heads", action="store_true",
                        help="By default only security/scarcity residual heads train.")
    parser.add_argument("--reset-guided-heads", action="store_true",
                        help="Zero the final security/scarcity residual layers before fine-tuning.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-selection-eval", action="store_true",
                        help="Skip post-training postprocess eval and model_best_feasible_rent.pth selection.")
    parser.add_argument("--selection-eval-samples", type=int, default=24,
                        help="Number of type samples for post-training postprocess selection.")
    parser.add_argument("--selection-eval-seed", type=int, default=20260426,
                        help="Seed for post-training postprocess selection samples.")
    parser.add_argument("--selection-mt-floor-tol", type=float, default=1e-3,
                        help="Maximum postprocess MT floor gap MWh to count as feasible.")
    parser.add_argument("--selection-mt-slack-tol", type=float, default=1e-3,
                        help="Maximum postprocess MT slack MWh to count as feasible.")
    parser.add_argument("--selection-info-rent-weight", type=float, default=1.0,
                        help="Weight on postprocess info rent in feasible low-rent selection.")
    parser.add_argument("--selection-procurement-weight", type=float, default=0.05,
                        help="Secondary weight on postprocess procurement cost.")
    parser.add_argument("--selection-offer-gap-weight", type=float, default=0.05,
                        help="Small tie-breaker weight on MT offer gap.")
    parser.add_argument("--selection-correction-weight", type=float, default=0.05,
                        help="Small tie-breaker weight on the correction-burden metric.")
    parser.add_argument("--selection-offer-adjustment-weight", type=float,
                        default=0.5,
                        help="Weight on positive adjustment inside the correction-burden metric.")
    parser.add_argument("--selection-mt-floor-weight", type=float, default=1e6,
                        help="Penalty weight for postprocess MT floor infeasibility.")
    parser.add_argument("--selection-mt-slack-weight", type=float, default=1e6,
                        help="Penalty weight for postprocess MT slack infeasibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.abspath(args.run) if args.run else latest_run(_ROOT)

    feedback_args = argparse.Namespace(
        run=run_dir,
        checkpoint=args.checkpoint,
        samples=args.samples,
        batch_size=args.feedback_batch_size,
        seed=args.seed,
        ctrl_min_ratio=args.ctrl_min_ratio,
        pi_buyback_ratio=args.pi_buyback_ratio,
        peer_bid_scale=args.peer_bid_scale,
        price_arch=args.price_arch,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        disable_peer_bid_context=args.disable_peer_bid_context,
        no_mt_security_uplift=False,
        adjustment_weight=1.0,
        settlement_weight=1e-3,
        mt_slack_weight=1e5,
    )
    print("Building correction-feedback dataset...")
    _, feedback, status_counts = build_dataset(feedback_args)
    print("  Postprocess status:",
          ";".join(f"{k}:{v}" for k, v in sorted(status_counts.items())))

    net = build_network_multi(constant_price=False,
                              ctrl_min_ratio=args.ctrl_min_ratio)
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
    ).to(device)
    ckpt_path = os.path.join(run_dir, args.checkpoint)
    mech.load_state_dict(load_state(ckpt_path), strict=False)
    mech.train()
    configure_trainable_heads(
        mech,
        train_all_price_heads=args.train_all_price_heads,
        reset_guided_heads=args.reset_guided_heads,
    )

    data_t = as_torch(feedback, device)
    targets = build_dual_targets(
        net, data_t,
        security_scale=args.security_scale,
        scarcity_scale=args.scarcity_scale,
        device=device,
    )
    baseline_mt_offer_gap = float(
        mt_offer_gap_from_arrays(net, data_t["offer_cap"]).detach().cpu())

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_dir = os.path.join(run_dir, f"dual_guided_{stem}_{ts}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    save_targets(os.path.join(out_dir, "dual_targets.npz"), targets)
    config = vars(args).copy()
    config["baseline_mt_offer_gap"] = baseline_mt_offer_gap
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    params = [p for p in mech.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    cfg = vars(args)
    cfg["baseline_mt_offer_gap"] = baseline_mt_offer_gap
    history = []
    best_score = float("inf")
    n = data_t["types"].shape[0]
    rho_ref_all = data_t["rho"]

    print(f"Fine-tuning on {device}; trainable params="
          f"{sum(p.numel() for p in params):,}; out={out_dir}")
    print(f"Baseline MT offer gap (offer sufficiency): "
          f"{baseline_mt_offer_gap:.4f} "
          f"(guard tolerance {args.mt_gap_tolerance:.4f})")
    torch.save(mech.state_dict(), os.path.join(out_dir, "model_initial.pth"))
    torch.save(mech.state_dict(), os.path.join(out_dir, "model_best.pth"))
    for it in range(1, args.iters + 1):
        sel = torch.randint(0, n, (args.batch_size,), device=device)
        types = data_t["types"].index_select(0, sel)
        rho_ref = rho_ref_all.index_select(0, sel)

        opt.zero_grad()
        terms = loss_terms(mech, types, rho_ref, targets, cfg)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if it % args.print_iter == 0 or it == 1:
            row = {"iter": it}
            for key, value in terms.items():
                row[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            history.append(row)
            print(
                f"[{it:5d}/{args.iters}] "
                f"loss={row['loss']:.4f} "
                f"sec={row['security_loss']:.4f} "
                f"scar={row['scarcity_loss']:.4f} "
                f"anchor={row['anchor_loss']:.4f} "
                f"MTofferGap={row['mt_offer_gap']:.4f} "
                f"guard={row['mt_gap_guard']:.4f} "
                f"rho={row['rho_mean']:.2f} "
                f"sec_mean={row['rho_security_mean']:.2f} "
                f"scar_mean={row['rho_scarcity_mean']:.2f} "
                f"peer_mean={row['rho_peer_bid_mean']:.2f} "
                f"d_sec={row['rho_security_residual_mean']:.2f} "
                f"d_scar={row['rho_scarcity_residual_mean']:.2f}"
            )

        loss_value = float(terms["loss"].detach().cpu())
        gap_value = float(terms["mt_offer_gap"].detach().cpu())
        guard_ok = gap_value <= baseline_mt_offer_gap + args.mt_gap_tolerance
        score = loss_value
        if guard_ok and score < best_score:
            best_score = score
            torch.save(mech.state_dict(), os.path.join(out_dir, "model_best.pth"))
        if it % args.save_iter == 0:
            torch.save(mech.state_dict(), os.path.join(out_dir, f"model_{it}.pth"))

    torch.save(mech.state_dict(), os.path.join(out_dir, "final_model.pth"))
    write_csv(os.path.join(out_dir, "history.csv"), history)
    if not args.skip_selection_eval:
        run_post_train_selection(out_dir, args)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
