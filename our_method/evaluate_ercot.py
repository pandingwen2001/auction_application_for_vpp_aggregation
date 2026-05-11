#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_ercot.py
-----------------
Evaluate a trained checkpoint across all 24 ERCOT scenarios.

For each (checkpoint, scenario) pair we:
  1) build the scenario's net_multi,
  2) load the checkpoint into a VPPMechanismMulti on that net,
  3) sample DER types under the prior,
  4) compute preliminary dispatch via the learned mechanism,
  5) project to physical feasibility through SecurityPostProcessor,
  6) compute the constrained social-optimum baseline,
  7) record economic and feasibility metrics.

Output: a wide CSV with one row per (checkpoint, scenario, stage)
and a short stdout summary aggregated across the 24 scenarios.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "data"))
sys.path.insert(0, _THIS_DIR)

from network.vpp_network_multi import build_network_multi       # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti    # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti          # noqa: E402
from our_method.postprocess_security import SecurityPostProcessor  # noqa: E402
from baseline.baseline_social_opt_multi import SocialOptimumMechanismMulti  # noqa: E402
from data.ercot_profiles import num_scenarios                    # noqa: E402

from our_method.evaluate_posted_price import (                   # noqa: E402
    classify_sources, metric_row, load_state, TYPE_CAP_RATIO,
    latest_run,
)


def evaluate_scenario(scenario_idx: int, mech_state_dict: dict, args):
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

    pre_row = metric_row("pre", net, types, x_pre, p_pre, P_pre,
                         rho=rho, offer_cap=offer_cap, x_social=x_soc,
                         source_types=source_types,
                         price_components=price_components)
    post_row = metric_row("post", net, types, x_post, p_post, P_post,
                          rho=rho, offer_cap=offer_cap, post_result=post,
                          x_social=x_soc, source_types=source_types,
                          price_components=price_components)
    soc_row = metric_row("social_opt", net, types, x_soc, p_soc, P_soc,
                         x_social=x_soc, source_types=source_types)

    for row in (pre_row, post_row, soc_row):
        row["scenario_idx"] = int(scenario_idx)
        row["scenario_date"] = str(net.get("scenario_date"))
    return [pre_row, post_row, soc_row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=[
        "model_best_loss.pth", "model_best.pth", "final_model.pth",
    ])
    parser.add_argument("--samples", type=int, default=16,
                        help="number of DER type samples per scenario")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-clip-factor", type=float, default=3.0)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--disable-peer-bid-context", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--no-mt-security-uplift", action="store_true")
    parser.add_argument("--out", default=None,
                        help="CSV output (default: <run>/eval_ercot.csv)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run) if args.run else latest_run(_ROOT)
    print(f"Run dir : {run_dir}")
    print(f"Scenarios: 0..{num_scenarios()-1}  ({num_scenarios()} total)")
    print(f"Samples / scenario: {args.samples}")
    print()

    rows = []
    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(run_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {ckpt_name} (file not found)")
            continue
        print(f"[checkpoint] {ckpt_name}")
        state = load_state(ckpt_path)
        for sc in range(num_scenarios()):
            scenario_rows = evaluate_scenario(sc, state, args)
            for r in scenario_rows:
                r["checkpoint"] = ckpt_name
            rows.extend(scenario_rows)
            pre = scenario_rows[0]
            post = scenario_rows[1]
            soc = scenario_rows[2]
            print(f"  sc {sc:2d} {pre['scenario_date']:10s}  "
                  f"proc(pre)={pre['procurement_cost']:7.1f} "
                  f"proc(post)={post['procurement_cost']:7.1f}  "
                  f"rent={post['info_rent']:6.1f}  "
                  f"social(soc)={soc['social_cost_true']:7.1f}  "
                  f"MTgap={post.get('mt_floor_gap_mwh',0):.3f}  "
                  f"posAdj={post.get('positive_adjustment_mwh',0):.3f}")
        print()

    if not rows:
        raise FileNotFoundError("No checkpoints evaluated")

    fieldnames = sorted({k for row in rows for k in row})
    out_path = args.out or os.path.join(run_dir, "eval_ercot.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved {len(rows)} rows -> {out_path}")

    # --- Summary across scenarios per (checkpoint, stage) ---
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("AGGREGATE: mean across 24 scenarios")
    print("=" * 90)
    agg = df.groupby(["checkpoint", "name"]).agg(
        proc=("procurement_cost", "mean"),
        social=("social_cost_true", "mean"),
        rent=("info_rent", "mean"),
        rent_max=("info_rent", "max"),
        mt_gap=("mt_floor_gap_mwh", "mean"),
        mt_gap_max=("mt_floor_gap_mwh", "max"),
    )
    agg["rent_ratio"] = agg["rent"] / agg["proc"]
    print(agg.to_string(float_format=lambda x: f"{x:9.3f}"))

    # Compare post vs social_opt (welfare gap)
    print("\n" + "=" * 90)
    print("POST vs SOCIAL_OPT: per-checkpoint procurement gap")
    print("=" * 90)
    post = df[df["name"] == "post"].set_index(["checkpoint", "scenario_idx"])
    soc = df[df["name"] == "social_opt"].set_index(["checkpoint", "scenario_idx"])
    if not post.empty and not soc.empty:
        joined = post[["procurement_cost"]].rename(
            columns={"procurement_cost": "proc_post"}).join(
            soc[["social_cost_true"]].rename(
                columns={"social_cost_true": "social_opt"}))
        joined["abs_gap"] = joined["proc_post"] - joined["social_opt"]
        joined["rel_premium"] = (joined["proc_post"] - joined["social_opt"]
                                 ) / joined["social_opt"]
        per_ckpt = joined.groupby(level="checkpoint").agg(
            proc_post_mean=("proc_post", "mean"),
            social_opt_mean=("social_opt", "mean"),
            abs_gap_mean=("abs_gap", "mean"),
            rel_premium_mean=("rel_premium", "mean"),
            rel_premium_max=("rel_premium", "max"),
        )
        print(per_ckpt.to_string(float_format=lambda x: f"{x:9.4f}"))


if __name__ == "__main__":
    main()
