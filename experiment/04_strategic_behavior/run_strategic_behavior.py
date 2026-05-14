#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 4: ERCOT Strategic Behavior
--------------------------------------
Evaluate unilateral strategic bid deviations on ERCOT scenarios.

Two attack classes are reported:
  1) fixed strategies: cost shading, cost inflation, and bound bids;
  2) projected black-box GD/SPSA best response over sampled bid reports.

Utility is always evaluated under the true type. A mechanism is safer when the
positive gain from misreporting is small.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "our_method"))

from baseline.baseline_common_multi import JointQPMulti  # noqa: E402
from baseline.cooperative_disaggregation_multi import cooperative_payoffs  # noqa: E402
from data.ercot_profiles import num_scenarios as ercot_num_scenarios  # noqa: E402
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


MAIN_METHODS = [
    "ours_posted_price",
    "vcg_disaggregation",
    "shapley_value_disaggregation",
    "nucleolus_disaggregation",
    "dlmp_settlement",
    "bid_dependent_opf_pay_as_bid",
    "bid_dependent_opf_uniform_da",
    "constrained_social_opt",
]

DEFAULT_SPSA_METHODS = [
    "ours_posted_price",
    "dlmp_settlement",
    "bid_dependent_opf_pay_as_bid",
    "bid_dependent_opf_uniform_da",
]

DETAIL_COLUMNS = [
    "attack_type",
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
    "utility_gain_mean",
    "procurement_delta",
    "info_rent_delta",
    "operation_cost_delta",
    "mt_floor_gap_delta",
    "positive_adjustment_delta",
    "own_rho_delta_max",
    "other_rho_delta_mean",
    "scenario_idx",
    "scenario_date",
]

SUMMARY_COLUMNS = [
    "attack_type",
    "method",
    "stage",
    "n_der",
    "n_eval_rows",
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
    "operation_cost_delta_at_worst",
    "positive_adjustment_delta_at_worst",
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
                   args) -> VPPMechanismMulti:
    mech = VPPMechanismMulti(
        net,
        posted_price_cfg=dict(
            price_arch=args.price_arch,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_dropout=args.transformer_dropout,
            pi_buyback_ratio=args.pi_buyback_ratio,
            use_peer_bid_context=True,
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
    return true_cost_per_der_np(bids_np, x_np)


def dlmp_price_tensor_np(net: dict, batch_size: int) -> np.ndarray:
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float64)
    A_volt = np.asarray(net["A_volt"], dtype=np.float64)
    v_base = np.asarray(net.get("v_base_profile", net.get("v_base")),
                        dtype=np.float64)
    denom = float(np.maximum(np.mean(v_base), 1e-6))
    loss_factor = A_volt.sum(axis=0) / denom
    dlmp = np.maximum(pi[:, None] * (1.0 + loss_factor[None, :]), 0.0)
    return np.broadcast_to(dlmp[None, :, :], (batch_size,) + dlmp.shape)


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
        "operation_cost": float((true_cost + grid_cost).mean()),
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


def solve_bid_opf(qp: JointQPMulti, bids_np: np.ndarray):
    B, N = bids_np.shape[:2]
    T = qp.T
    x_out = np.zeros((B, T, N), dtype=np.float64)
    P_out = np.zeros((B, T), dtype=np.float64)
    statuses = []
    for b in range(B):
        a = np.broadcast_to(bids_np[b, :, 0][None, :], (T, N))
        c = np.broadcast_to(bids_np[b, :, 1][None, :], (T, N))
        x_np, P_np, status = qp.solve(a, c)
        x_out[b] = x_np
        P_out[b] = P_np
        statuses.append(status)
    return x_out, P_out, statuses


def status_string(statuses: list) -> str:
    if not statuses:
        return ""
    return ";".join(f"{k}:{v}" for k, v in sorted(Counter(statuses).items()))


class StrategicEvaluator:
    def __init__(self, net: dict, types: torch.Tensor,
                 mech: VPPMechanismMulti, postprocessor: SecurityPostProcessor,
                 nucleolus_tol: float = 1e-7):
        self.net = net
        self.types = types
        self.types_np = to_numpy(types).astype(np.float64)
        self.B, self.N, _ = self.types_np.shape
        self.mech = mech
        self.postprocessor = postprocessor
        self.qp = JointQPMulti(net)
        self.nucleolus_tol = float(nucleolus_tol)
        self._social_eval = None

    def evaluate_ours(self, bids_np: np.ndarray) -> dict:
        bids = torch.tensor(bids_np, dtype=torch.float32)
        with torch.no_grad():
            x_pre, rho, p_pre, P_pre = self.mech(bids)
            offer_cap = self.mech._last_offer_cap.detach().clone()
        x_pre_np = to_numpy(x_pre).astype(np.float64)
        rho_np = to_numpy(rho).astype(np.float64)
        P_pre_np = to_numpy(P_pre).astype(np.float64)
        p_pre_np = to_numpy(p_pre).astype(np.float64)
        offer_np = to_numpy(offer_cap).astype(np.float64)

        post = self.postprocessor.process_batch(
            x_pre_np, P_pre_np, rho_np, offer_np)
        x_post_np = post.x.astype(np.float64)
        P_post_np = post.P_VPP.astype(np.float64)
        p_post_np = (rho_np * x_post_np).sum(axis=1)
        row = stage_metrics(
            self.net, self.types_np, x_post_np, p_post_np, P_post_np,
            positive_adjustment_np=post.positive_adjustment,
            mt_slack_np=post.mt_slack,
        )
        row["rho"] = rho_np
        row["status"] = status_string(post.status)
        return {"ours_posted_price": row}

    def evaluate_market(self, bids_np: np.ndarray,
                        methods: list = None) -> dict:
        methods = methods or [m for m in MAIN_METHODS
                              if m not in {"ours_posted_price",
                                           "constrained_social_opt"}]
        x_np, P_np, statuses = solve_bid_opf(self.qp, bids_np)
        out = {}
        status = status_string(statuses)

        reported_cost = reported_cost_per_der_np(bids_np, x_np)
        if "bid_dependent_opf_pay_as_bid" in methods:
            row = stage_metrics(self.net, self.types_np, x_np,
                                reported_cost, P_np)
            row["status"] = status
            out["bid_dependent_opf_pay_as_bid"] = row

        if "bid_dependent_opf_uniform_da" in methods:
            rho_da = np.asarray(self.net["pi_DA_profile"], dtype=np.float64
                                )[None, :, None]
            p_np = (rho_da * x_np).sum(axis=1)
            row = stage_metrics(self.net, self.types_np, x_np, p_np, P_np)
            row["status"] = status
            out["bid_dependent_opf_uniform_da"] = row

        if "dlmp_settlement" in methods:
            rho = dlmp_price_tensor_np(self.net, self.B)
            p_np = (rho * x_np).sum(axis=1)
            row = stage_metrics(self.net, self.types_np, x_np, p_np, P_np)
            row["status"] = status
            out["dlmp_settlement"] = row

        coop_methods = []
        if "vcg_disaggregation" in methods:
            coop_methods.append("vcg")
        if "shapley_value_disaggregation" in methods:
            coop_methods.append("shapley")
        if "nucleolus_disaggregation" in methods:
            coop_methods.append("nucleolus")
        if coop_methods:
            payoff = cooperative_payoffs(
                bids_np,
                self.net,
                methods=tuple(coop_methods),
                nucleolus_tol=self.nucleolus_tol,
            )
            mapping = {
                "vcg": "vcg_disaggregation",
                "shapley": "shapley_value_disaggregation",
                "nucleolus": "nucleolus_disaggregation",
            }
            for method in coop_methods:
                p_np = reported_cost + payoff[method]
                row = stage_metrics(self.net, self.types_np, x_np, p_np, P_np)
                row["status"] = status
                out[mapping[method]] = row
        return out

    def evaluate_social_opt(self) -> dict:
        if self._social_eval is not None:
            return {"constrained_social_opt": self._social_eval}
        x_np, P_np, statuses = solve_bid_opf(self.qp, self.types_np)
        p_np = true_cost_per_der_np(self.types_np, x_np)
        row = stage_metrics(self.net, self.types_np, x_np, p_np, P_np)
        row["status"] = status_string(statuses)
        self._social_eval = row
        return {"constrained_social_opt": row}

    def evaluate(self, bids_np: np.ndarray, methods: list = None) -> dict:
        methods = methods or MAIN_METHODS
        out = {}
        if "ours_posted_price" in methods:
            out.update(self.evaluate_ours(bids_np))
        market_methods = [m for m in methods
                          if m not in {"ours_posted_price",
                                       "constrained_social_opt"}]
        if market_methods:
            out.update(self.evaluate_market(bids_np, market_methods))
        if "constrained_social_opt" in methods:
            out.update(self.evaluate_social_opt())
        return out

    def utility_mean(self, method: str, der_idx: int,
                     bids_np: np.ndarray) -> float:
        return float(self.evaluate(bids_np, [method])[method]["utility"][:, der_idx].mean())


def strategy_candidates(args):
    specs = []
    for scale in args.inflation_scales:
        specs.append(dict(
            attack_type="fixed",
            strategy="cost_inflation",
            strategy_param=f"scale={scale:g}",
            kind="scale",
            a_scale=float(scale),
            b_scale=float(scale),
        ))
        specs.append(dict(
            attack_type="fixed",
            strategy="linear_inflation",
            strategy_param=f"b_scale={scale:g}",
            kind="scale",
            a_scale=1.0,
            b_scale=float(scale),
        ))
    for scale in args.shading_scales:
        specs.append(dict(
            attack_type="fixed",
            strategy="cost_shading",
            strategy_param=f"scale={scale:g}",
            kind="scale",
            a_scale=float(scale),
            b_scale=float(scale),
        ))
    specs.append(dict(
        attack_type="fixed",
        strategy="withholding_proxy",
        strategy_param="bid=upper_bound",
        kind="set_hi",
    ))
    specs.append(dict(
        attack_type="fixed",
        strategy="quantity_pressure",
        strategy_param="bid=lower_bound",
        kind="set_lo",
    ))
    return specs


def apply_strategy_np(types_np: np.ndarray, prior: DERTypePriorMulti,
                      der_idx: int, spec: dict) -> np.ndarray:
    bids = types_np.copy()
    lo = to_numpy(prior.lo).astype(np.float64)
    hi = to_numpy(prior.hi).astype(np.float64)
    if spec["kind"] == "scale":
        bids[:, der_idx, 0] *= float(spec["a_scale"])
        bids[:, der_idx, 1] *= float(spec["b_scale"])
    elif spec["kind"] == "set_hi":
        bids[:, der_idx, :] = hi[der_idx]
    elif spec["kind"] == "set_lo":
        bids[:, der_idx, :] = lo[der_idx]
    else:
        raise ValueError(f"Unknown strategy kind: {spec['kind']}")
    bids[:, der_idx, :] = np.clip(bids[:, der_idx, :], lo[der_idx], hi[der_idx])
    return bids


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


def make_detail_row(attack_type: str, method: str, der_idx: int,
                    labels: list, source_types: list, spec: dict,
                    truth_eval: dict, mis_eval: dict,
                    scenario_idx, scenario_date: str) -> dict:
    u_truth = truth_eval["utility"][:, der_idx]
    u_mis = mis_eval["utility"][:, der_idx]
    gain = u_mis - u_truth
    row = {
        "attack_type": attack_type,
        "method": method,
        "stage": "final",
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
        "procurement_truth": truth_eval["procurement_cost"],
        "procurement_misreport": mis_eval["procurement_cost"],
        "procurement_delta": mis_eval["procurement_cost"] - truth_eval["procurement_cost"],
        "operation_cost_truth": truth_eval["operation_cost"],
        "operation_cost_misreport": mis_eval["operation_cost"],
        "operation_cost_delta": mis_eval["operation_cost"] - truth_eval["operation_cost"],
        "info_rent_truth": truth_eval["info_rent"],
        "info_rent_misreport": mis_eval["info_rent"],
        "info_rent_delta": mis_eval["info_rent"] - truth_eval["info_rent"],
        "mt_floor_gap_truth": truth_eval["mt_floor_gap_mwh"],
        "mt_floor_gap_misreport": mis_eval["mt_floor_gap_mwh"],
        "mt_floor_gap_delta": (
            mis_eval["mt_floor_gap_mwh"] - truth_eval["mt_floor_gap_mwh"]),
        "positive_adjustment_truth": truth_eval.get("positive_adjustment_mwh", 0.0),
        "positive_adjustment_misreport": mis_eval.get("positive_adjustment_mwh", 0.0),
        "positive_adjustment_delta": (
            mis_eval.get("positive_adjustment_mwh", 0.0)
            - truth_eval.get("positive_adjustment_mwh", 0.0)),
        "utility_min_misreport": mis_eval["utility_min"],
        "scenario_idx": scenario_idx,
        "scenario_date": scenario_date,
    }
    row.update(price_delta_row(truth_eval, mis_eval, der_idx))
    return row


def evaluate_fixed_attacks(evaluator: StrategicEvaluator, prior: DERTypePriorMulti,
                           labels: list, source_types: list, strategies: list,
                           methods: list, scenario_idx, scenario_date: str) -> list:
    truth_eval = evaluator.evaluate(evaluator.types_np, methods)
    rows = []
    for spec in strategies:
        for der_idx in range(len(labels)):
            bids = apply_strategy_np(evaluator.types_np, prior, der_idx, spec)
            mis_eval = evaluator.evaluate(bids, methods)
            for method in methods:
                rows.append(make_detail_row(
                    "fixed", method, der_idx, labels, source_types,
                    spec, truth_eval[method], mis_eval[method],
                    scenario_idx, scenario_date))
    return rows


def normalized_from_bids(types_np: np.ndarray, prior: DERTypePriorMulti,
                         der_idx: int) -> np.ndarray:
    lo = to_numpy(prior.lo).astype(np.float64)[der_idx]
    hi = to_numpy(prior.hi).astype(np.float64)[der_idx]
    return np.clip((types_np[:, der_idx, :] - lo) / np.maximum(hi - lo, 1e-9),
                   0.0, 1.0)


def bids_from_normalized(types_np: np.ndarray, prior: DERTypePriorMulti,
                         der_idx: int, z: np.ndarray) -> np.ndarray:
    lo = to_numpy(prior.lo).astype(np.float64)[der_idx]
    hi = to_numpy(prior.hi).astype(np.float64)[der_idx]
    bids = types_np.copy()
    bids[:, der_idx, :] = lo + np.clip(z, 0.0, 1.0) * (hi - lo)
    return bids


def spsa_best_response(evaluator: StrategicEvaluator, prior: DERTypePriorMulti,
                       method: str, der_idx: int, args,
                       rng: np.random.Generator):
    z = normalized_from_bids(evaluator.types_np, prior, der_idx)
    truth_eval = evaluator.evaluate(evaluator.types_np, [method])[method]
    truth_u = float(truth_eval["utility"][:, der_idx].mean())
    best_z = z.copy()
    best_u = truth_u

    for k in range(int(args.gd_steps)):
        ak = float(args.gd_lr) / ((k + 1) ** 0.15)
        ck = float(args.gd_perturb) / ((k + 1) ** 0.10)
        delta = rng.choice([-1.0, 1.0], size=z.shape)
        z_plus = np.clip(z + ck * delta, 0.0, 1.0)
        z_minus = np.clip(z - ck * delta, 0.0, 1.0)
        u_plus = evaluator.utility_mean(
            method, der_idx, bids_from_normalized(
                evaluator.types_np, prior, der_idx, z_plus))
        u_minus = evaluator.utility_mean(
            method, der_idx, bids_from_normalized(
                evaluator.types_np, prior, der_idx, z_minus))
        grad = ((u_plus - u_minus) / max(2.0 * ck, 1e-9)) * delta
        z = np.clip(z + ak * grad, 0.0, 1.0)
        u_cur = evaluator.utility_mean(
            method, der_idx, bids_from_normalized(
                evaluator.types_np, prior, der_idx, z))
        if u_plus > best_u:
            best_u, best_z = u_plus, z_plus.copy()
        if u_minus > best_u:
            best_u, best_z = u_minus, z_minus.copy()
        if u_cur > best_u:
            best_u, best_z = u_cur, z.copy()

    bids_best = bids_from_normalized(evaluator.types_np, prior, der_idx, best_z)
    mis_eval = evaluator.evaluate(bids_best, [method])[method]
    return truth_eval, mis_eval


def evaluate_spsa_attacks(evaluator: StrategicEvaluator, prior: DERTypePriorMulti,
                          labels: list, source_types: list, methods: list,
                          args, scenario_idx, scenario_date: str) -> list:
    rows = []
    rng = np.random.default_rng(args.seed + 1000 + int(scenario_idx))
    for method in methods:
        print(f"  SPSA best response for {method}...")
        for der_idx in range(len(labels)):
            truth_eval, mis_eval = spsa_best_response(
                evaluator, prior, method, der_idx, args, rng)
            spec = dict(
                strategy="projected_spsa_gd",
                strategy_param=(
                    f"steps={args.gd_steps},lr={args.gd_lr:g},"
                    f"perturb={args.gd_perturb:g}"),
            )
            rows.append(make_detail_row(
                "optimal_spsa", method, der_idx, labels, source_types,
                spec, truth_eval, mis_eval, scenario_idx, scenario_date))
    return rows


def best_rows_by_der(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["attack_type"], row["method"], row["stage"],
                 row["der_idx"], row.get("scenario_idx", ""))].append(row)
    return [max(group, key=lambda r: r["regret_mean"])
            for group in grouped.values()]


def aggregate_best_summary(best_rows: list) -> list:
    grouped = defaultdict(list)
    for row in best_rows:
        grouped[(row["attack_type"], row["method"], row["stage"])].append(row)
    out = []
    for (attack_type, method, stage), group in sorted(grouped.items()):
        worst = max(group, key=lambda r: r["regret_mean"])
        out.append({
            "attack_type": attack_type,
            "method": method,
            "stage": stage,
            "n_der": len({r["der_idx"] for r in group}),
            "n_eval_rows": len(group),
            "regret_mean_across_der": float(np.mean(
                [r["regret_mean"] for r in group])),
            "regret_max_der": float(max(r["regret_mean"] for r in group)),
            "regret_max_sample": float(max(r["regret_max_sample"] for r in group)),
            "worst_der_idx": worst["der_idx"],
            "worst_der_label": worst["der_label"],
            "worst_der_type": worst["source_type"],
            "worst_strategy": worst["strategy"],
            "worst_strategy_param": worst["strategy_param"],
            "procurement_delta_at_worst": worst["procurement_delta"],
            "info_rent_delta_at_worst": worst["info_rent_delta"],
            "operation_cost_delta_at_worst": worst["operation_cost_delta"],
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


def parse_methods(value: str) -> list:
    if value.lower() == "all":
        return list(MAIN_METHODS)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint", default="model_best.pth")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--scenario-idx", type=int, default=0)
    parser.add_argument("--all-ercot-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--pi-clip-factor", type=float, default=3.0)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--inflation-scales", nargs="*", type=float,
                        default=[1.1, 1.25, 1.5])
    parser.add_argument("--shading-scales", nargs="*", type=float,
                        default=[0.5, 0.8])
    parser.add_argument("--methods", default="all",
                        help="Comma-separated methods or 'all'.")
    parser.add_argument("--spsa-methods",
                        default=",".join(DEFAULT_SPSA_METHODS),
                        help="Comma-separated methods for optimal SPSA search.")
    parser.add_argument("--include-cooperative-spsa", action="store_true",
                        help="Also run SPSA for VCG/Shapley/Nucleolus.")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-spsa", action="store_true")
    parser.add_argument("--gd-steps", type=int, default=30)
    parser.add_argument("--gd-lr", type=float, default=0.08)
    parser.add_argument("--gd-perturb", type=float, default=0.12)
    parser.add_argument("--adjustment-weight", type=float, default=1000.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e7)
    parser.add_argument("--nucleolus-tol", type=float, default=1e-7)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def scenario_indices(args):
    if args.all_ercot_scenarios:
        out = list(range(ercot_num_scenarios()))
        if args.max_scenarios is not None:
            out = out[:int(args.max_scenarios)]
        return out
    return [args.scenario_idx]


def build_eval_network(args, scenario_idx: int):
    return build_network_multi(
        scenario_idx=int(scenario_idx),
        ctrl_min_ratio=args.ctrl_min_ratio,
        pi_clip_factor=args.pi_clip_factor,
    )


def spsa_method_list(args, methods: list) -> list:
    selected = parse_methods(args.spsa_methods)
    if args.include_cooperative_spsa:
        selected.extend([
            "vcg_disaggregation",
            "shapley_value_disaggregation",
            "nucleolus_disaggregation",
        ])
    selected = [m for m in selected if m in methods]
    return list(dict.fromkeys(selected))


def evaluate_one_scenario(args, run_dir: str, scenario_idx: int,
                          methods: list, spsa_methods: list):
    net = build_eval_network(args, scenario_idx)
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed + int(scenario_idx))
    types = prior.sample(args.samples, device="cpu")
    labels = list(net["der_labels"])
    src_types = classify_sources(net)
    scenario_date = str(net.get("scenario_date", ""))

    postprocessor = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=True,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )
    mech = load_mechanism(net, run_dir, args.checkpoint, args)
    evaluator = StrategicEvaluator(
        net, types, mech, postprocessor, nucleolus_tol=args.nucleolus_tol)

    rows = []
    if not args.skip_fixed:
        print("Evaluating fixed strategic bid grid...")
        rows.extend(evaluate_fixed_attacks(
            evaluator, prior, labels, src_types, strategy_candidates(args),
            methods, scenario_idx, scenario_date))

    if not args.skip_spsa and spsa_methods:
        print("Evaluating projected SPSA/GD best responses...")
        rows.extend(evaluate_spsa_attacks(
            evaluator, prior, labels, src_types, spsa_methods,
            args, scenario_idx, scenario_date))
    return rows


def main():
    args = parse_args()
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoint(_ROOT, args.checkpoint))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results_ercot")
    os.makedirs(out_dir, exist_ok=True)

    methods = parse_methods(args.methods)
    bad = [m for m in methods if m not in MAIN_METHODS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    spsa_methods = spsa_method_list(args, methods)

    detailed_rows = []
    for sc in scenario_indices(args):
        print("\n" + "=" * 80)
        print(f"Experiment 4 setting: ERCOT scenario {sc}")
        print("=" * 80)
        detailed_rows.extend(evaluate_one_scenario(
            args, run_dir, sc, methods, spsa_methods))

    best_rows = best_rows_by_der(detailed_rows)
    summary_rows = aggregate_best_summary(best_rows)

    detailed_path = os.path.join(out_dir, "strategic_behavior_detailed.csv")
    best_path = os.path.join(out_dir, "best_response_by_der.csv")
    summary_path = os.path.join(out_dir, "best_response_summary.csv")
    summary_md_path = os.path.join(out_dir, "best_response_summary.md")
    config_path = os.path.join(out_dir, "strategic_behavior_config.json")

    write_csv(detailed_path, detailed_rows)
    write_csv(best_path, best_rows, DETAIL_COLUMNS)
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_markdown(summary_md_path, summary_rows, SUMMARY_COLUMNS)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["data_source"] = "ercot"
        config["methods"] = methods
        config["spsa_methods_resolved"] = spsa_methods
        config["fixed_strategies"] = strategy_candidates(args)
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed rows        : {detailed_path}")
    print(f"Saved best response rows   : {best_path}")
    print(f"Saved best response summary: {summary_path}")
    print(f"Saved best response MD     : {summary_md_path}")
    print(f"Saved config               : {config_path}")
    print("\nBest-response summary:")
    for row in summary_rows:
        print(
            f"  {row['attack_type']:<13} {row['method']:<34} "
            f"regret_mean={fmt(row['regret_mean_across_der']):<10} "
            f"regret_max={fmt(row['regret_max_der']):<10} "
            f"worst={row['worst_der_label']} "
            f"{row['worst_strategy']}({row['worst_strategy_param']})"
        )


if __name__ == "__main__":
    main()
