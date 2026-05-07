#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3: Ablation Study
----------------------------
Evaluate which parts of the posted-price mechanism matter.

This script performs evaluation-time component ablations on a trained
checkpoint and optional checkpoint-selection comparisons:

  - full learned mechanism
  - public-context-only price rule
  - remove peer-bid component
  - base + type only
  - remove security component
  - remove scarcity component
  - remove residual dual-guided heads
  - compare selector/extra checkpoints when available

All rows use the same sampled DER types and security postprocess settings.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "our_method"))

from baseline.baseline_social_opt_multi import SocialOptimumMechanismMulti  # noqa: E402
from network.vpp_network_multi import build_network_multi  # noqa: E402
from our_method.evaluate_posted_price import (  # noqa: E402
    TYPE_CAP_RATIO,
    classify_sources,
    latest_run,
    load_state,
    metric_row,
)
from our_method.postprocess_security import SecurityPostProcessor  # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti  # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti  # noqa: E402


PAPER_COLUMNS = [
    "ablation_id",
    "ablation_name",
    "name",
    "paper_role",
    "checkpoint",
    "use_peer_bid_context",
    "procurement_cost",
    "social_cost_true",
    "info_rent",
    "utility_min",
    "mt_offer_gap_mwh",
    "mt_floor_gap_mwh",
    "postprocess_mt_slack_mwh",
    "positive_adjustment_mwh",
    "positive_adjustment_payment",
    "dispatch_l1_gap_mwh",
    "postprocess_status",
]


ABLATIONS = [
    dict(
        ablation_id="A0",
        ablation_name="full_peer_context",
        paper_role="proposed_full",
        use_peer_bid_context=True,
        component_scale={},
        description="Full learned posted price with own-bid-excluded peer context.",
    ),
    dict(
        ablation_id="A1",
        ablation_name="public_context_only",
        paper_role="remove_peer_bid_context",
        use_peer_bid_context=False,
        component_scale={},
        description="Instantiate the learned price rule without peer-bid context.",
    ),
    dict(
        ablation_id="A2",
        ablation_name="zero_peer_bid_component",
        paper_role="remove_peer_bid_component",
        use_peer_bid_context=True,
        component_scale={"peer_bid": 0.0},
        description="Keep architecture but zero the peer-bid price component.",
    ),
    dict(
        ablation_id="A3",
        ablation_name="base_type_only",
        paper_role="remove_security_scarcity_peer",
        use_peer_bid_context=True,
        component_scale={
            "security_main": 0.0,
            "scarcity_main": 0.0,
            "security_residual": 0.0,
            "scarcity_residual": 0.0,
            "peer_bid": 0.0,
        },
        description="Only base and type/context components remain.",
    ),
    dict(
        ablation_id="A4",
        ablation_name="no_security_component",
        paper_role="remove_security",
        use_peer_bid_context=True,
        component_scale={
            "security_main": 0.0,
            "security_residual": 0.0,
        },
        description="Remove network/security price adders.",
    ),
    dict(
        ablation_id="A5",
        ablation_name="no_scarcity_component",
        paper_role="remove_scarcity",
        use_peer_bid_context=True,
        component_scale={
            "scarcity_main": 0.0,
            "scarcity_residual": 0.0,
        },
        description="Remove controllable-resource scarcity price adders.",
    ),
    dict(
        ablation_id="A6",
        ablation_name="no_residual_heads",
        paper_role="remove_dual_guided_residual",
        use_peer_bid_context=True,
        component_scale={
            "security_residual": 0.0,
            "scarcity_residual": 0.0,
        },
        description="Remove postprocess-dual residual heads, if trained.",
    ),
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


def apply_component_scale(mech: VPPMechanismMulti, scale: dict):
    pp = mech.posted_price_net
    pp.reset_component_scale()
    if scale:
        pp.set_component_scale(**scale)


def add_metadata(row: dict, spec: dict, stage: str,
                 checkpoint: str, run_dir: str) -> dict:
    out = dict(row)
    out["ablation_id"] = spec.get("ablation_id", "")
    out["ablation_name"] = spec.get("ablation_name", "")
    out["paper_role"] = spec.get("paper_role", "")
    out["ablation_description"] = spec.get("description", "")
    out["component_scale"] = json.dumps(spec.get("component_scale", {}),
                                        sort_keys=True)
    out["use_peer_bid_context"] = int(bool(spec.get("use_peer_bid_context", True)))
    out["stage"] = stage
    out["checkpoint"] = checkpoint
    out["run_dir"] = run_dir
    return out


def evaluate_spec(spec: dict, net: dict, types: torch.Tensor,
                  x_social: torch.Tensor, source_types: list,
                  postprocessor: SecurityPostProcessor,
                  run_dir: str, checkpoint: str, args) -> list:
    mech = load_mechanism(
        net,
        run_dir,
        checkpoint,
        args,
        use_peer_bid_context=bool(spec.get("use_peer_bid_context", True)),
    )
    apply_component_scale(mech, spec.get("component_scale", {}))

    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()
        price_components = getattr(mech, "_last_price_components_detached", None)

    post = postprocessor.process_batch(
        x_pre.detach().cpu().numpy(),
        P_pre.detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
        offer_cap.detach().cpu().numpy(),
    )
    x_post = torch.tensor(post.x, dtype=torch.float32)
    P_post = torch.tensor(post.P_VPP, dtype=torch.float32)
    p_post = (rho * x_post).sum(dim=1)

    pre_row = metric_row(
        f"{spec['ablation_name']}:{checkpoint}:pre",
        net,
        types,
        x_pre,
        p_pre,
        P_pre,
        rho=rho,
        offer_cap=offer_cap,
        x_social=x_social,
        source_types=source_types,
        price_components=price_components,
    )
    post_row = metric_row(
        f"{spec['ablation_name']}:{checkpoint}:post",
        net,
        types,
        x_post,
        p_post,
        P_post,
        rho=rho,
        offer_cap=offer_cap,
        post_result=post,
        x_social=x_social,
        source_types=source_types,
        price_components=price_components,
    )
    return [
        add_metadata(pre_row, spec, "pre", checkpoint, run_dir),
        add_metadata(post_row, spec, "post", checkpoint, run_dir),
    ]


def social_row(net: dict, types: torch.Tensor, x_social: torch.Tensor,
               p_social: torch.Tensor, P_social: torch.Tensor,
               source_types: list) -> dict:
    row = metric_row(
        "constrained_social_opt",
        net,
        types,
        x_social,
        p_social,
        P_social,
        x_social=x_social,
        source_types=source_types,
    )
    spec = dict(
        ablation_id="OPT",
        ablation_name="constrained_social_opt",
        paper_role="oracle_lower_bound",
        use_peer_bid_context=False,
        component_scale={},
        description="Oracle constrained social optimum with true types.",
    )
    row = add_metadata(row, spec, "feasible", "", "")
    row["postprocess_status"] = "oracle_qp"
    return row


def selector_specs(extra_checkpoints: list) -> list:
    specs = []
    for ckpt in extra_checkpoints:
        specs.append(dict(
            ablation_id="S",
            ablation_name=f"selector_full_{ckpt}",
            paper_role="checkpoint_selector_comparison",
            use_peer_bid_context=True,
            component_scale={},
            description="Full model evaluated at an alternative selected checkpoint.",
            checkpoint_override=ckpt,
        ))
    return specs


def write_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value, digits: int = 4) -> str:
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
        "| " + " | ".join(PAPER_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(PAPER_COLUMNS)) + " |",
    ]
    for row in rows:
        values = []
        for col in PAPER_COLUMNS:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                value = fmt_value(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def select_table_rows(rows: list) -> list:
    out = []
    for row in rows:
        if row.get("stage") in {"post", "feasible"}:
            out.append(row)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run directory. Defaults to latest run containing the base checkpoint.")
    parser.add_argument("--checkpoint", default="model_best_constr.pth",
                        help="Main checkpoint used for component ablations.")
    parser.add_argument("--extra-checkpoints", nargs="*", default=[
        "model_best_loss.pth",
        "model_best.pth",
        "model_best_feasible_rent.pth",
        "model_best_correction.pth",
        "final_model.pth",
    ], help="Optional full-model checkpoint selector comparisons.")
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
    parser.add_argument("--skip-selector-comparison", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints_for_run = [args.checkpoint] + list(args.extra_checkpoints)
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoints(_ROOT, checkpoints_for_run))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    net = build_network_multi(
        constant_price=False,
        ctrl_min_ratio=args.ctrl_min_ratio,
    )
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed)
    types = prior.sample(args.samples, device="cpu")
    source_types = classify_sources(net)

    postprocessor = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=True,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )

    print("Solving constrained social optimum reference...")
    social = SocialOptimumMechanismMulti(net)
    social.eval()
    with torch.no_grad():
        x_soc, _, p_soc, P_soc = social(types)

    rows = [
        social_row(net, types, x_soc, p_soc, P_soc, source_types)
    ]

    base_path = os.path.join(run_dir, args.checkpoint)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Main checkpoint not found: {base_path}")

    for spec in ABLATIONS:
        print(f"Evaluating {spec['ablation_id']} {spec['ablation_name']}...")
        rows.extend(evaluate_spec(
            spec, net, types, x_soc, source_types,
            postprocessor, run_dir, args.checkpoint, args))

    if not args.skip_selector_comparison:
        seen = {args.checkpoint}
        for spec in selector_specs(args.extra_checkpoints):
            ckpt = spec["checkpoint_override"]
            if ckpt in seen:
                continue
            seen.add(ckpt)
            if not os.path.exists(os.path.join(run_dir, ckpt)):
                print(f"Skipping missing selector checkpoint: {ckpt}")
                continue
            print(f"Evaluating selector checkpoint comparison: {ckpt}")
            rows.extend(evaluate_spec(
                spec, net, types, x_soc, source_types,
                postprocessor, run_dir, ckpt, args))

    detailed_path = os.path.join(out_dir, "ablation_detailed.csv")
    table_rows = select_table_rows(rows)
    table_csv_path = os.path.join(out_dir, "ablation_table.csv")
    table_md_path = os.path.join(out_dir, "ablation_table.md")
    config_path = os.path.join(out_dir, "ablation_config.json")
    write_csv(detailed_path, rows)
    write_csv(table_csv_path, table_rows)
    write_markdown_table(table_md_path, table_rows)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["available_ablations"] = ABLATIONS
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed rows: {detailed_path}")
    print(f"Saved table CSV   : {table_csv_path}")
    print(f"Saved table MD    : {table_md_path}")
    print(f"Saved config      : {config_path}")
    print("\nAblation headline rows:")
    for row in table_rows:
        print(
            f"  {row['ablation_id']:<3} {row['ablation_name']:<30} "
            f"cost={fmt_value(row.get('procurement_cost')):<10} "
            f"rent={fmt_value(row.get('info_rent')):<10} "
            f"offer_gap={fmt_value(row.get('mt_offer_gap_mwh')):<10} "
            f"corr={fmt_value(row.get('positive_adjustment_mwh')):<10} "
            f"MTgap={fmt_value(row.get('mt_floor_gap_mwh'))}"
        )


if __name__ == "__main__":
    main()
