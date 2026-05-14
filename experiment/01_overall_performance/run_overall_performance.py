#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Overall Performance
---------------------------------
Build the headline comparison table for the posted-price VPP paper.

Rows include:
  - cooperative disaggregation baselines (VCG, Shapley, nucleolus)
  - DLMP settlement baseline
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

from baseline.cooperative_disaggregation_multi import cooperative_payoffs  # noqa: E402
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

try:
    from data.ercot_profiles import num_scenarios as ercot_num_scenarios  # noqa: E402
except Exception:  # pragma: no cover - Liu-only fallback
    ercot_num_scenarios = None


PAPER_COLUMNS = [
    "name",
    "category",
    "settlement",
    "operation_cost",
    "info_rent_cost",
    "total_procurement_cost",
    "operation_cost_gap_pct",
    "dispatch_l1_gap_mwh",
    "der_energy_mwh",
    "renewable_energy_mwh",
    "renewable_share_pct",
    "grid_import_mwh",
    "utility_min",
    "utility_shortfall_cost",
    "utility_mean",
    "feasible_rate_pct",
    "feasible_status",
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


def feasible_rate_from_status(status: str) -> float:
    """Convert solver/postprocess status counters into a paper-table rate."""
    status = str(status or "").strip()
    if status in {"", "not_applicable"}:
        return 100.0

    good = 0.0
    total = 0.0
    for item in status.split(";"):
        if not item:
            continue
        if ":" in item:
            label, count = item.rsplit(":", 1)
            try:
                count = float(count)
            except ValueError:
                count = 1.0
        else:
            label, count = item, 1.0
        label = label.strip().lower()
        total += count
        if label in {"optimal", "optimal_inaccurate"}:
            good += count
    return 100.0 * good / max(total, 1.0)


def add_table_metrics(row: dict, social_ref_cost: float = None) -> dict:
    """Add paper-facing aliases and efficiency/safety metrics."""
    out = dict(row)
    operation_cost = float(out.get("social_cost_true", 0.0))
    info_rent = float(out.get("info_rent", 0.0))
    total_procurement = float(out.get("procurement_cost",
                                      operation_cost + info_rent))
    out["operation_cost"] = operation_cost
    out["info_rent_cost"] = info_rent
    out["total_procurement_cost"] = total_procurement
    if social_ref_cost is not None and social_ref_cost > 1e-9:
        out["operation_cost_gap_pct"] = (
            100.0 * (operation_cost - social_ref_cost) / social_ref_cost)
    else:
        out["operation_cost_gap_pct"] = 0.0

    pv = float(out.get("PV_energy_mwh", 0.0) or 0.0)
    wt = float(out.get("WT_energy_mwh", 0.0) or 0.0)
    der = float(out.get("der_energy_mwh", 0.0) or 0.0)
    out["renewable_energy_mwh"] = pv + wt
    out["renewable_share_pct"] = 100.0 * (pv + wt) / max(der, 1e-9)
    out["feasible_status"] = out.get("postprocess_status", "")
    out["feasible_rate_pct"] = feasible_rate_from_status(
        out.get("postprocess_status", ""))
    out["utility_shortfall_cost"] = max(
        0.0, -float(out.get("utility_min", 0.0) or 0.0))
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


def evaluate_cooperative_disaggregation_baselines(
        net: dict, types: torch.Tensor,
        x_dispatch: torch.Tensor, P_dispatch: torch.Tensor,
        x_social: torch.Tensor, source_types: list,
        args) -> list:
    """Shapley/nucleolus surplus disaggregation on a common dispatch."""
    requested = tuple(args.cooperative_methods)
    types_np = types.detach().cpu().numpy()
    print("Computing cooperative surplus allocations: "
          + ", ".join(requested))
    payoff_np = cooperative_payoffs(
        types_np,
        net,
        methods=requested,
        nucleolus_tol=args.nucleolus_tol,
    )
    realized_cost = true_cost_total(types, x_dispatch)

    rows = []
    for method in requested:
        payoff = torch.tensor(payoff_np[method], dtype=torch.float32,
                              device=types.device)
        p = realized_cost + payoff
        if method == "shapley":
            name = "shapley_value_disaggregation"
            settlement = "shapley_surplus"
        elif method == "nucleolus":
            name = "nucleolus_disaggregation"
            settlement = "nucleolus_surplus"
        elif method == "vcg":
            name = "vcg_disaggregation"
            settlement = "vcg_marginal_contribution"
        else:
            name = f"{method}_disaggregation"
            settlement = f"{method}_surplus"

        row = metric_row(
            name,
            net,
            types,
            x_dispatch,
            p,
            P_dispatch,
            x_social=x_social,
            source_types=source_types,
        )
        row["cooperative_surplus"] = float(payoff.sum(dim=1).mean())
        row["positive_adjustment_mwh"] = 0.0
        row["positive_adjustment_payment"] = 0.0
        row["postprocess_mt_slack_mwh"] = 0.0
        row["postprocess_status"] = "not_applicable"
        rows.append(add_metadata(
            row,
            "cooperative_disaggregation",
            settlement,
            "feasible",
        ))
    return rows


def dlmp_price_tensor(net: dict, batch_size: int,
                      device: torch.device) -> torch.Tensor:
    """
    Liu-style active-power DLMP approximation.

    Liu et al. define DLMP as the active balance price plus loss and
    network-dual adders. Our active-power model does not expose reactive
    prices, so this baseline uses the active energy component with the
    linearised loss-factor term already used by the local network helper:

        lambda_i,t = pi_DA_t * (1 + LF_i).
    """
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float32)
    A_volt = np.asarray(net["A_volt"], dtype=np.float32)
    v_base = np.asarray(net.get("v_base_profile", net.get("v_base")),
                        dtype=np.float32)
    denom = float(np.maximum(np.mean(v_base), 1e-6))
    loss_factor = A_volt.sum(axis=0) / denom
    dlmp = pi[:, None] * (1.0 + loss_factor[None, :])
    dlmp = np.maximum(dlmp, 0.0).astype(np.float32)
    return torch.tensor(dlmp, dtype=torch.float32, device=device
                        ).unsqueeze(0).expand(batch_size, -1, -1)


def evaluate_dlmp_baseline(net: dict, types: torch.Tensor,
                           x_dispatch: torch.Tensor,
                           P_dispatch: torch.Tensor,
                           x_social: torch.Tensor,
                           source_types: list) -> dict:
    rho = dlmp_price_tensor(net, types.shape[0], types.device)
    p = (rho * x_dispatch).sum(dim=1)
    row = metric_row(
        "dlmp_settlement",
        net,
        types,
        x_dispatch,
        p,
        P_dispatch,
        rho=rho,
        x_social=x_social,
        source_types=source_types,
    )
    row["postprocess_status"] = "not_applicable"
    return add_metadata(row, "dlmp", "active_loss_dlmp", "feasible")


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
        if category in {"oracle", "bid_dependent_opf",
                        "cooperative_disaggregation", "dlmp"}:
            out.append(row)
        elif category in {"learned_posted_price"} and stage == "post":
            out.append(row)
    return out


def _as_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_paper_rows(rows: list) -> list:
    """Average selected paper rows across ERCOT scenarios, when applicable."""
    grouped = {}
    key_cols = ("name", "category", "settlement", "stage", "checkpoint")
    for row in rows:
        key = tuple(row.get(col, "") for col in key_cols)
        grouped.setdefault(key, []).append(row)

    out = []
    for _key, group in grouped.items():
        if len(group) == 1:
            row = dict(group[0])
            row["n_eval_rows"] = 1
            out.append(row)
            continue

        merged = dict(group[0])
        all_cols = sorted({k for row in group for k in row})
        for col in all_cols:
            vals = [row.get(col, "") for row in group]
            nums = [_as_float(v) for v in vals]
            if col in {"scenario_idx", "scenario_date"}:
                merged[col] = "multiple"
            elif all(v is not None for v in nums):
                if col == "utility_min":
                    merged[col] = float(np.min(nums))
                elif col == "utility_shortfall_cost":
                    merged[col] = float(np.max(nums))
                elif col == "feasible_rate_pct":
                    merged[col] = float(np.min(nums))
                else:
                    merged[col] = float(np.mean(nums))
            elif len({str(v) for v in vals}) == 1:
                merged[col] = vals[0]
            elif col == "postprocess_status":
                merged[col] = ";".join(sorted({str(v) for v in vals}))
            else:
                merged[col] = vals[0]
        merged["n_eval_rows"] = len(group)
        out.append(merged)
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
    parser.add_argument("--data-source", choices=["ercot", "liu"],
                        default="ercot",
                        help="Evaluation profile source. ERCOT is the paper default.")
    parser.add_argument("--scenario-idx", type=int, default=0,
                        help="ERCOT typical-day scenario index when not using all scenarios.")
    parser.add_argument("--all-ercot-scenarios", action="store_true",
                        help="Evaluate all ERCOT typical-day scenarios and aggregate the table.")
    parser.add_argument("--pi-clip-factor", type=float, default=3.0)
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
    parser.add_argument("--include-fixed-price", action="store_true",
                        help="Include fixed posted-price baselines.")
    parser.add_argument("--skip-fixed-price", action="store_true")
    parser.add_argument("--skip-bid-opf", action="store_true")
    parser.add_argument("--skip-dlmp", action="store_true")
    parser.add_argument("--skip-cooperative-disaggregation", action="store_true")
    parser.add_argument("--cooperative-methods", nargs="*",
                        default=["vcg", "shapley", "nucleolus"],
                        choices=["vcg", "shapley", "nucleolus"])
    parser.add_argument("--nucleolus-tol", type=float, default=1e-7)
    parser.add_argument("--skip-public-context-ablation", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--no-mt-security-uplift", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def build_eval_network(args, scenario_idx):
    if args.data_source == "ercot":
        return build_network_multi(
            scenario_idx=int(scenario_idx),
            ctrl_min_ratio=args.ctrl_min_ratio,
            pi_clip_factor=args.pi_clip_factor,
        )
    return build_network_multi(
        constant_price=False,
        ctrl_min_ratio=args.ctrl_min_ratio,
    )


def scenario_indices(args):
    if args.data_source != "ercot":
        return [None]
    if args.all_ercot_scenarios:
        if ercot_num_scenarios is None:
            raise RuntimeError("ERCOT profiles are not available.")
        return list(range(ercot_num_scenarios()))
    return [args.scenario_idx]


def evaluate_one_setting(args, run_dir: str, scenario_idx=None) -> list:
    net = build_eval_network(args, 0 if scenario_idx is None else scenario_idx)
    prior = DERTypePriorMulti(net)
    seed_offset = 0 if scenario_idx is None else int(scenario_idx)
    torch.manual_seed(args.seed + seed_offset)
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

    if not args.skip_cooperative_disaggregation:
        rows.extend(evaluate_cooperative_disaggregation_baselines(
            net, types, x_soc, P_soc, x_soc, source_types, args))

    if not args.skip_dlmp:
        print("Evaluating DLMP settlement baseline...")
        rows.append(evaluate_dlmp_baseline(
            net, types, x_soc, P_soc, x_soc, source_types))

    if not args.skip_bid_opf:
        print("Evaluating bid-dependent OPF settlement baselines...")
        rows.extend(evaluate_bid_opf_baselines(net, types, x_soc, source_types))

    if args.include_fixed_price and not args.skip_fixed_price:
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

    for row in rows:
        row["data_source"] = args.data_source
        row["scenario_idx"] = "" if scenario_idx is None else int(scenario_idx)
        row["scenario_date"] = str(net.get("scenario_date", "liu"))
    social_ref_cost = rows[0]["social_cost_true"]
    return [add_table_metrics(row, social_ref_cost) for row in rows]


def main():
    args = parse_args()
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoints(_ROOT, args.checkpoints))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sc in scenario_indices(args):
        label = "Liu profiles" if sc is None else f"ERCOT scenario {sc}"
        print("\n" + "=" * 80)
        print(f"Experiment 1 setting: {label}")
        print("=" * 80)
        rows.extend(evaluate_one_setting(args, run_dir, sc))

    detailed_path = os.path.join(out_dir, "overall_performance_detailed.csv")
    paper_rows = aggregate_paper_rows(select_paper_rows(rows))
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
            f"op={fmt_value(row.get('operation_cost')):<10} "
            f"rent={fmt_value(row.get('info_rent_cost')):<10} "
            f"total={fmt_value(row.get('total_procurement_cost')):<10} "
            f"gap={fmt_value(row.get('operation_cost_gap_pct'))}%"
        )


if __name__ == "__main__":
    main()
