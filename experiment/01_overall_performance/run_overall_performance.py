#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Overall Performance
---------------------------------
Build the headline comparison table for the posted-price VPP paper.

Rows include:
  - fixed posted-price baselines
  - bid-dependent OPF settlement baselines
  - learned posted-price checkpoints, with and without peer-bid context
  - constrained social optimum oracle

All rows use the same sampled DER types, network setting, postprocess settings,
and constrained social-optimum reference.
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

from baseline.baseline_common_multi import JointQPMulti, true_cost_total  # noqa: E402
from baseline.baseline_social_opt_multi import SocialOptimumMechanismMulti  # noqa: E402
from network.opf_layer_multi import DC3OPFLayerMulti  # noqa: E402
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
    "name",
    "category",
    "settlement",
    "procurement_cost",
    "social_cost_true",
    "info_rent",
    "utility_min",
    "mt_floor_gap_mwh",
    "postprocess_mt_slack_mwh",
    "mt_offer_gap_mwh",
    "positive_adjustment_mwh",
    "positive_adjustment_payment",
    "dispatch_l1_gap_mwh",
    "postprocess_status",
]


class FixedPostedPriceMechanism(torch.nn.Module):
    """Non-learned posted-price mechanism with rho_i,t = ratio * pi_DA_t."""

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

        labels = net["der_labels"]
        der_types = net["der_type"]
        type_names = [self._classify_source(label, der_type)
                      for label, der_type in zip(labels, der_types)]
        type_cap_ratio = type_cap_ratio or TYPE_CAP_RATIO
        cap_ratio = np.asarray([type_cap_ratio[t] for t in type_names],
                               dtype=np.float32)
        pi = np.asarray(net["pi_DA_profile"], dtype=np.float32)
        floor = float(pi_buyback_ratio) * pi[:, None]
        rho = self.ratio * pi[:, None]
        cap = pi[:, None] * cap_ratio[None, :]
        rho = np.minimum(np.maximum(rho, floor), cap)
        self.register_buffer(
            "rho_fixed",
            torch.tensor(rho, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _classify_source(label: str, der_type: str) -> str:
        if label.startswith("PV"):
            return "PV"
        if label.startswith("WT"):
            return "WT"
        if label.startswith("MT"):
            return "MT"
        if der_type == "DR":
            return "DR"
        return "DG"

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
        x, P_VPP = self.dc3_opf(rho, supply_cap=offer_cap)
        p = (rho * x).sum(dim=1)
        self._last_offer_cap = offer_cap
        return x, rho, p, P_VPP


def load_learned_mechanism(net: dict, run_dir: str, checkpoint: str,
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


def process_post(postprocessor: SecurityPostProcessor,
                 x_pre: torch.Tensor, P_pre: torch.Tensor,
                 rho: torch.Tensor, offer_cap: torch.Tensor):
    return postprocessor.process_batch(
        x_pre.detach().cpu().numpy(),
        P_pre.detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
        offer_cap.detach().cpu().numpy(),
    )


def add_metadata(row: dict, category: str, settlement: str,
                 stage: str, checkpoint: str = "",
                 run_dir: str = "") -> dict:
    out = dict(row)
    out["category"] = category
    out["settlement"] = settlement
    out["stage"] = stage
    out["checkpoint"] = checkpoint
    out["run_dir"] = run_dir
    return out


def evaluate_posted_price_mechanism(name: str, category: str,
                                    settlement: str, mech,
                                    net: dict, types: torch.Tensor,
                                    x_social: torch.Tensor,
                                    source_types: list,
                                    postprocessor: SecurityPostProcessor,
                                    checkpoint: str = "",
                                    run_dir: str = "") -> list:
    with torch.no_grad():
        x_pre, rho, p_pre, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()
        price_components = getattr(mech, "_last_price_components_detached", None)

    post = process_post(postprocessor, x_pre, P_pre, rho, offer_cap)
    x_post = torch.tensor(post.x, dtype=torch.float32)
    P_post = torch.tensor(post.P_VPP, dtype=torch.float32)
    p_post = (rho * x_post).sum(dim=1)

    rows = [
        metric_row(f"{name}:pre", net, types, x_pre, p_pre, P_pre,
                   rho=rho, offer_cap=offer_cap, x_social=x_social,
                   source_types=source_types,
                   price_components=price_components),
        metric_row(f"{name}:post", net, types, x_post, p_post, P_post,
                   rho=rho, offer_cap=offer_cap, post_result=post,
                   x_social=x_social, source_types=source_types,
                   price_components=price_components),
    ]
    return [
        add_metadata(rows[0], category, settlement, "pre", checkpoint, run_dir),
        add_metadata(rows[1], category, settlement, "post", checkpoint, run_dir),
    ]


def evaluate_social_optimum(net: dict, types: torch.Tensor,
                            x_social: torch.Tensor, p_social: torch.Tensor,
                            P_social: torch.Tensor, source_types: list) -> dict:
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
    return add_metadata(row, "oracle", "true_cost_payment", "feasible")


def evaluate_bid_opf_baselines(net: dict, types: torch.Tensor,
                               x_social: torch.Tensor,
                               source_types: list) -> list:
    """Bid-dependent OPF dispatch under truthful bids, with two settlements."""
    qp = JointQPMulti(net)
    B, T, N = types.shape[0], int(net["T"]), int(net["n_ders"])
    types_np = types.detach().cpu().numpy()
    x_out = np.zeros((B, T, N), dtype=np.float32)
    P_out = np.zeros((B, T), dtype=np.float32)
    statuses = []

    for b in range(B):
        a = np.broadcast_to(types_np[b, :, 0][None, :], (T, N))
        c = np.broadcast_to(types_np[b, :, 1][None, :], (T, N))
        x_np, P_np, status = qp.solve(a, c)
        x_out[b] = x_np
        P_out[b] = P_np
        statuses.append(status)

    x = torch.tensor(x_out, dtype=torch.float32)
    P = torch.tensor(P_out, dtype=torch.float32)
    pi = torch.tensor(net["pi_DA_profile"], dtype=torch.float32)
    rho_da = pi.view(1, T, 1).expand(B, T, N)

    p_pay_as_bid = true_cost_total(types, x)
    p_uniform_da = (rho_da * x).sum(dim=1)

    rows = []
    for name, p, settlement in [
        ("bid_dependent_opf_pay_as_bid", p_pay_as_bid, "pay_as_bid"),
        ("bid_dependent_opf_uniform_da", p_uniform_da, "uniform_DA_price"),
    ]:
        row = metric_row(name, net, types, x, p, P,
                         rho=rho_da if "uniform" in name else None,
                         x_social=x_social, source_types=source_types)
        row["postprocess_status"] = ";".join(
            f"{k}:{v}" for k, v in sorted(Counter(statuses).items()))
        row["positive_adjustment_mwh"] = 0.0
        row["positive_adjustment_payment"] = 0.0
        row["postprocess_mt_slack_mwh"] = 0.0
        rows.append(add_metadata(row, "bid_dependent_opf",
                                 settlement, "feasible"))
    return rows


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
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_table(path: str, rows: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = PAPER_COLUMNS
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt_value(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def select_paper_rows(rows: list) -> list:
    """Keep the headline rows for the paper-facing table."""
    out = []
    for row in rows:
        name = row.get("name", "")
        category = row.get("category", "")
        stage = row.get("stage", "")
        if category in {"oracle", "bid_dependent_opf"}:
            out.append(row)
        elif category in {"fixed_price", "learned_posted_price"} and stage == "post":
            out.append(row)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run directory. Defaults to latest runs/*.")
    parser.add_argument("--checkpoints", nargs="*", default=[
        "model_best_constr.pth",
        "model_best_loss.pth",
        "model_best.pth",
        "model_best_feasible_rent.pth",
        "model_best_correction.pth",
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
    parser.add_argument("--fixed-price-ratios", nargs="*", type=float,
                        default=[0.5, 0.7, 1.0],
                        help="Uniform fixed-price ratios relative to pi_DA.")
    parser.add_argument("--skip-fixed-price", action="store_true")
    parser.add_argument("--skip-bid-opf", action="store_true")
    parser.add_argument("--skip-public-context-ablation", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--no-mt-security-uplift", action="store_true")
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
    source_types = classify_sources(net)

    postprocessor = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=not args.no_mt_security_uplift,
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
        evaluate_social_optimum(net, types, x_soc, p_soc, P_soc, source_types)
    ]

    if not args.skip_bid_opf:
        print("Evaluating bid-dependent OPF settlement baselines...")
        rows.extend(evaluate_bid_opf_baselines(net, types, x_soc, source_types))

    if not args.skip_fixed_price:
        for ratio in args.fixed_price_ratios:
            print(f"Evaluating fixed posted price ratio={ratio:.3f}...")
            fixed = FixedPostedPriceMechanism(
                net,
                ratio=ratio,
                pi_buyback_ratio=args.pi_buyback_ratio,
                type_cap_ratio=TYPE_CAP_RATIO,
            )
            rows.extend(evaluate_posted_price_mechanism(
                name=f"fixed_price_ratio_{ratio:.2f}",
                category="fixed_price",
                settlement="posted_price",
                mech=fixed,
                net=net,
                types=types,
                x_social=x_soc,
                source_types=source_types,
                postprocessor=postprocessor,
            ))

    for ckpt in args.checkpoints:
        ckpt_path = os.path.join(run_dir, ckpt)
        if not os.path.exists(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        print(f"Evaluating learned posted price with peer context: {ckpt}")
        mech = load_learned_mechanism(
            net, run_dir, ckpt, args, use_peer_bid_context=True)
        rows.extend(evaluate_posted_price_mechanism(
            name=f"learned_peer_{ckpt}",
            category="learned_posted_price",
            settlement="own_bid_excluded_peer_context",
            mech=mech,
            net=net,
            types=types,
            x_social=x_soc,
            source_types=source_types,
            postprocessor=postprocessor,
            checkpoint=ckpt,
            run_dir=run_dir,
        ))

        if not args.skip_public_context_ablation:
            print(f"Evaluating public-context-only ablation: {ckpt}")
            no_peer = load_learned_mechanism(
                net, run_dir, ckpt, args, use_peer_bid_context=False)
            rows.extend(evaluate_posted_price_mechanism(
                name=f"learned_public_only_{ckpt}",
                category="learned_posted_price",
                settlement="public_context_only",
                mech=no_peer,
                net=net,
                types=types,
                x_social=x_soc,
                source_types=source_types,
                postprocessor=postprocessor,
                checkpoint=ckpt,
                run_dir=run_dir,
            ))

    detailed_path = os.path.join(out_dir, "overall_performance_detailed.csv")
    paper_rows = select_paper_rows(rows)
    paper_csv_path = os.path.join(out_dir, "overall_performance_table.csv")
    paper_md_path = os.path.join(out_dir, "overall_performance_table.md")
    config_path = os.path.join(out_dir, "overall_performance_config.json")
    write_csv(detailed_path, rows)
    write_csv(paper_csv_path, paper_rows)
    write_markdown_table(paper_md_path, paper_rows)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed rows: {detailed_path}")
    print(f"Saved paper table CSV: {paper_csv_path}")
    print(f"Saved paper table MD : {paper_md_path}")
    print(f"Saved config         : {config_path}")
    print("\nHeadline rows:")
    for row in paper_rows:
        print(
            f"  {row['name']:<45} "
            f"cost={fmt_value(row.get('procurement_cost')):<10} "
            f"rent={fmt_value(row.get('info_rent')):<10} "
            f"MTgap={fmt_value(row.get('mt_floor_gap_mwh')):<10} "
            f"corr={fmt_value(row.get('positive_adjustment_mwh'))}"
        )


if __name__ == "__main__":
    main()
