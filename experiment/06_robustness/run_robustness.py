#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6: Robustness Under Load Uncertainty
-----------------------------------------------
Evaluate feasibility and economic stability under deterministic load-scale
stress and random hourly load-profile uncertainty.

Each scenario rebuilds the 33-bus physical margins from the perturbed load
profile, then evaluates the same trained checkpoint and baselines.
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
sys.path.insert(0, os.path.join(_ROOT, "network"))
sys.path.insert(0, os.path.join(_ROOT, "our_method"))

from baseline.baseline_common_multi import JointQPMulti  # noqa: E402
from network.opf_layer_multi import DC3OPFLayerMulti  # noqa: E402
from network.vpp_network_multi import build_network_multi, _extract_loads_q  # noqa: E402
from our_method.evaluate_posted_price import (  # noqa: E402
    TYPE_CAP_RATIO,
    classify_sources,
    latest_run,
    load_state,
)
from our_method.postprocess_security import SecurityPostProcessor  # noqa: E402
from our_method.trainer_multi import DERTypePriorMulti  # noqa: E402
from our_method.vpp_mechanism_multi import VPPMechanismMulti  # noqa: E402


SUMMARY_COLUMNS = [
    "method",
    "stage",
    "scenario_group",
    "n_rows",
    "feasible_rate_mean",
    "feasible_all_count",
    "procurement_cost_mean",
    "procurement_cost_std",
    "info_rent_mean",
    "info_rent_std",
    "social_cost_true_mean",
    "utility_min_min",
    "mt_floor_gap_mwh_mean",
    "mt_floor_gap_mwh_max",
    "postprocess_mt_slack_mwh_mean",
    "positive_adjustment_mwh_mean",
    "positive_adjustment_mwh_std",
    "line_violation_max_mw",
    "voltage_violation_max_pu",
    "power_balance_residual_max_mw",
    "grid_import_mwh_mean",
    "der_energy_mwh_mean",
]

CURVE_COLUMNS = [
    "method",
    "stage",
    "load_scale",
    "scenario_id",
    "feasible_rate",
    "feasible_flag",
    "procurement_cost",
    "info_rent",
    "social_cost_true",
    "mt_floor_gap_mwh",
    "postprocess_mt_slack_mwh",
    "positive_adjustment_mwh",
    "line_violation_max_mw",
    "voltage_violation_max_pu",
    "grid_import_mwh",
    "der_energy_mwh",
]


class FixedPostedPriceMechanism(torch.nn.Module):
    """Non-learned posted-price baseline with rho_i,t = ratio * pi_DA_t."""

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
        type_names = []
        for label, der_type in zip(labels, der_types):
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

        type_cap_ratio = type_cap_ratio or TYPE_CAP_RATIO
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
    if x is None:
        return None
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


def economics(net: dict, types_np: np.ndarray, x_np: np.ndarray,
              p_np: np.ndarray, P_np: np.ndarray) -> dict:
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float64)
    true_cost_i = true_cost_per_der_np(types_np, x_np)
    payment = p_np.sum(axis=1)
    true_cost = true_cost_i.sum(axis=1)
    grid_cost = (pi[None, :] * P_np).sum(axis=1)
    utility = p_np - true_cost_i
    return {
        "procurement_cost": float((payment + grid_cost).mean()),
        "social_cost_true": float((true_cost + grid_cost).mean()),
        "der_payment": float(payment.mean()),
        "grid_cost": float(grid_cost.mean()),
        "true_der_cost": float(true_cost.mean()),
        "info_rent": float((payment - true_cost).mean()),
        "utility_min": float(utility.min()),
        "utility_mean": float(utility.mean()),
        "der_energy_mwh": float(x_np.sum(axis=(1, 2)).mean()),
        "grid_import_mwh": float(P_np.sum(axis=1).mean()),
    }


def security_metrics(net: dict, x_np: np.ndarray, P_np: np.ndarray,
                     offer_cap_np: np.ndarray, ess_net_np: np.ndarray = None,
                     x_ref_np: np.ndarray = None, rho_np: np.ndarray = None,
                     mt_slack_np: np.ndarray = None, statuses: list = None,
                     tol: float = 1e-4) -> dict:
    B, T, N = x_np.shape
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
    non_mt_idx = np.asarray([i for i in range(N)
                             if i not in set(mt_idx.tolist())],
                            dtype=np.int64)

    if offer_cap_np is None:
        offer_cap_np = np.broadcast_to(x_bar[None, :, :], (B, T, N))
    if ess_net_np is None:
        ess_net_np = np.zeros((B, T, A_flow_ess.shape[1]), dtype=np.float64)

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
    mt_gap_by_sample = np.maximum(floor - mt_dispatch, 0.0).sum(axis=1)
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

    balance = x_np.sum(axis=2) + ess_net_np.sum(axis=2) + P_np - load[None, :]

    if x_ref_np is None:
        pos_adj = np.zeros_like(x_np)
        neg_adj = np.zeros_like(x_np)
    else:
        delta = x_np - x_ref_np
        pos_adj = np.maximum(delta, 0.0)
        neg_adj = np.maximum(-delta, 0.0)

    if mt_slack_np is None:
        mt_slack = np.zeros(B, dtype=np.float64)
    else:
        mt_slack = np.nansum(mt_slack_np, axis=1)

    flow_sample = flow_viol.max(axis=(1, 2))
    volt_sample = volt_viol.max(axis=(1, 2))
    cap_sample = cap_viol.max(axis=(1, 2))
    balance_sample = np.abs(balance).max(axis=1)
    sample_feasible = (
        (flow_sample <= tol)
        & (volt_sample <= tol)
        & (cap_sample <= tol)
        & (balance_sample <= tol)
        & (mt_gap_by_sample <= tol)
        & (mt_slack <= tol)
    )

    out = {
        "feasible_flag": int(bool(sample_feasible.all())),
        "feasible_rate": float(sample_feasible.mean()),
        "line_violation_max_mw": float(flow_viol.max()),
        "voltage_violation_max_pu": float(volt_viol.max()),
        "physical_cap_violation_max_mw": float(cap_viol.max()),
        "non_mt_offer_violation_max_mw": non_mt_offer_viol,
        "power_balance_residual_max_mw": float(np.abs(balance).max()),
        "mt_floor_gap_mwh": float(mt_gap_by_sample.mean()),
        "mt_floor_gap_max_mwh": float(mt_gap_by_sample.max()),
        "mt_offer_gap_mwh": float(mt_offer_gap_by_sample.mean()),
        "mt_security_uplift_mwh": float(mt_security_uplift.sum(axis=(1, 2)).mean()),
        "postprocess_mt_slack_mwh": float(mt_slack.mean()),
        "positive_adjustment_mwh": float(pos_adj.sum(axis=(1, 2)).mean()),
        "negative_adjustment_mwh": float(neg_adj.sum(axis=(1, 2)).mean()),
        "correction_l1_mwh": float((pos_adj + neg_adj).sum(axis=(1, 2)).mean()),
    }
    if rho_np is not None:
        out["positive_adjustment_payment"] = float(
            (rho_np * pos_adj).sum(axis=(1, 2)).mean())
    if statuses is not None:
        out["status_count"] = ";".join(
            f"{k}:{v}" for k, v in sorted(Counter(statuses).items()))
    return out


def apply_load_multiplier(net: dict, multiplier: np.ndarray) -> dict:
    """Return a scenario net with load profile and physical margins updated."""
    out = dict(net)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    load_profile = np.asarray(net["load_profile"], dtype=np.float64) * multiplier

    loads_base = np.asarray(net["loads"], dtype=np.float64)
    loads_q = _extract_loads_q(net)
    load_total_single = float(net["load_total"])
    load_multiplier = load_profile / load_total_single
    loads_profile = loads_base[None, :] * load_multiplier[:, None]
    loads_profile_q = loads_q[None, :] * load_multiplier[:, None]

    H = np.asarray(net["H"], dtype=np.float64)
    P_inj_base_profile = -loads_profile
    baseline_flow_profile = P_inj_base_profile @ H.T
    line_ratings = np.asarray(net["line_ratings"], dtype=np.float64)
    flow_margin_up_profile = line_ratings[None, :] - baseline_flow_profile
    flow_margin_dn_profile = line_ratings[None, :] + baseline_flow_profile

    S_V_P = np.asarray(net["S_V_P"], dtype=np.float64)
    S_V_Q = np.asarray(net["S_V_Q"], dtype=np.float64)
    Q_inj_base_profile = -loads_profile_q
    v_base_profile = np.ones((int(net["T"]), int(net["n_buses"])),
                             dtype=np.float64)
    v_base_profile += P_inj_base_profile @ S_V_P.T
    v_base_profile += Q_inj_base_profile @ S_V_Q.T
    v_base_profile[:, 0] = 1.0
    v_base_profile = np.clip(v_base_profile, net["v_min"], 1.02)

    out.update(dict(
        load_profile=load_profile,
        load_total_profile=load_profile,
        loads_profile=loads_profile,
        baseline_flow_profile=baseline_flow_profile,
        flow_margin_up_profile=flow_margin_up_profile,
        flow_margin_dn_profile=flow_margin_dn_profile,
        v_base_profile=v_base_profile,
        volt_margin_up_profile=net["v_max"] - v_base_profile,
        volt_margin_dn_profile=v_base_profile - net["v_min"],
    ))
    return out


def smooth_hourly_multiplier(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, (1, 1), mode="edge")
    return (
        0.25 * padded[:-2]
        + 0.50 * padded[1:-1]
        + 0.25 * padded[2:]
    )


def parse_band(spec: str):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Band must have format name:low:high, got {spec!r}")
    name, lo, hi = parts
    return name, float(lo), float(hi)


def build_scenarios(base_net: dict, args) -> list:
    rng = np.random.default_rng(args.scenario_seed)
    T = int(base_net["T"])
    scenarios = []
    for scale in args.scale_levels:
        multiplier = np.full(T, float(scale), dtype=np.float64)
        scenarios.append({
            "scenario_id": f"scale_{scale:g}",
            "scenario_group": "scale_sweep",
            "scenario_kind": "uniform_scale",
            "load_multiplier": multiplier,
        })

    for band_spec in args.uncertainty_bands:
        name, lo, hi = parse_band(band_spec)
        for k in range(int(args.random_scenarios_per_band)):
            multiplier = rng.uniform(lo, hi, size=T).astype(np.float64)
            if not args.disable_smoothing:
                multiplier = smooth_hourly_multiplier(multiplier)
                multiplier = np.clip(multiplier, lo, hi)
            scenarios.append({
                "scenario_id": f"{name}_{k:03d}",
                "scenario_group": f"random_{name}",
                "scenario_kind": "hourly_random_band",
                "load_multiplier": multiplier,
                "band_low": lo,
                "band_high": hi,
            })
    return scenarios


def scenario_metadata(scenario: dict, net: dict) -> dict:
    mult = np.asarray(scenario["load_multiplier"], dtype=np.float64)
    load = np.asarray(net["load_profile"], dtype=np.float64)
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_group": scenario["scenario_group"],
        "scenario_kind": scenario["scenario_kind"],
        "load_scale": float(mult.mean()),
        "load_multiplier_min": float(mult.min()),
        "load_multiplier_max": float(mult.max()),
        "load_peak_mw": float(load.max()),
        "load_mean_mw": float(load.mean()),
        "load_mwh": float(load.sum()),
        "band_low": scenario.get("band_low", ""),
        "band_high": scenario.get("band_high", ""),
    }


def solve_bid_opf(qp: JointQPMulti, bids_np: np.ndarray):
    B, N = bids_np.shape[:2]
    T = qp.T
    n_ess = int(qp.net["ess_params"]["n_ess"])
    x_out = np.zeros((B, T, N), dtype=np.float32)
    P_out = np.zeros((B, T), dtype=np.float32)
    ess_net_out = np.zeros((B, T, n_ess), dtype=np.float32)
    statuses = []
    for b in range(B):
        a = np.broadcast_to(bids_np[b, :, 0][None, :], (T, N))
        c = np.broadcast_to(bids_np[b, :, 1][None, :], (T, N))
        x_np, P_np, status = qp.solve(a, c)
        x_out[b] = x_np
        P_out[b] = P_np
        ess = qp.last_ess
        if ess is not None:
            ess_net_out[b] = ess["P_d"] - ess["P_c"]
        statuses.append(status)
    return (
        x_out.astype(np.float64),
        P_out.astype(np.float64),
        ess_net_out.astype(np.float64),
        statuses,
    )


def evaluate_posted_price_method(name: str, category: str, mech,
                                 net: dict, types: torch.Tensor,
                                 postprocessor: SecurityPostProcessor,
                                 scenario_meta: dict,
                                 eval_seed: int, checkpoint: str = "",
                                 run_dir: str = "") -> list:
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
    p_pre_np = to_numpy(p_pre)
    P_pre_np = to_numpy(P_pre)
    offer_np = to_numpy(offer_cap)

    rows = []
    pre = {}
    pre.update(scenario_meta)
    pre.update(economics(net, types_np, x_pre_np, p_pre_np, P_pre_np))
    pre.update(security_metrics(
        net, x_pre_np, P_pre_np, offer_np, ess_pre, rho_np=rho_np,
        statuses=["preliminary_dc3"] * x_pre_np.shape[0]))
    pre.update({
        "method": name,
        "category": category,
        "stage": "pre",
        "eval_seed": eval_seed,
        "checkpoint": checkpoint,
        "run_dir": run_dir,
    })
    rows.append(pre)

    post = postprocessor.process_batch(x_pre_np, P_pre_np, rho_np, offer_np)
    x_post_np = post.x.astype(np.float64)
    P_post_np = post.P_VPP.astype(np.float64)
    p_post_np = (rho_np * x_post_np).sum(axis=1)
    ess_post = None
    if post.P_d is not None and post.P_c is not None:
        ess_post = post.P_d.astype(np.float64) - post.P_c.astype(np.float64)
    post_row = {}
    post_row.update(scenario_meta)
    post_row.update(economics(net, types_np, x_post_np, p_post_np, P_post_np))
    post_row.update(security_metrics(
        net, x_post_np, P_post_np, offer_np, ess_post,
        x_ref_np=x_pre_np, rho_np=rho_np, mt_slack_np=post.mt_slack,
        statuses=post.status))
    post_row.update({
        "method": name,
        "category": category,
        "stage": "post",
        "eval_seed": eval_seed,
        "checkpoint": checkpoint,
        "run_dir": run_dir,
    })
    rows.append(post_row)
    return rows


def evaluate_social_optimum(net: dict, types: torch.Tensor,
                            scenario_meta: dict, eval_seed: int) -> dict:
    qp = JointQPMulti(net)
    types_np = to_numpy(types)
    x_np, P_np, ess_net_np, statuses = solve_bid_opf(qp, types_np)
    p_np = true_cost_per_der_np(types_np, x_np)
    offer = np.broadcast_to(
        np.asarray(net["x_bar_profile"], dtype=np.float64)[None, :, :],
        x_np.shape,
    )
    row = {}
    row.update(scenario_meta)
    row.update(economics(net, types_np, x_np, p_np, P_np))
    row.update(security_metrics(
        net, x_np, P_np, offer, ess_net_np=ess_net_np, statuses=statuses))
    row.update({
        "method": "constrained_social_opt",
        "category": "oracle",
        "stage": "feasible",
        "eval_seed": eval_seed,
    })
    return row


def evaluate_bid_opf(net: dict, types: torch.Tensor,
                     scenario_meta: dict, eval_seed: int) -> list:
    qp = JointQPMulti(net)
    types_np = to_numpy(types)
    x_np, P_np, ess_net_np, statuses = solve_bid_opf(qp, types_np)
    offer = np.broadcast_to(
        np.asarray(net["x_bar_profile"], dtype=np.float64)[None, :, :],
        x_np.shape,
    )
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float64)[None, :, None]
    settlements = [
        ("bid_dependent_opf_pay_as_bid",
         reported_cost_per_der_np(types_np, x_np),
         "pay_as_bid"),
        ("bid_dependent_opf_uniform_da",
         (pi * x_np).sum(axis=1),
         "uniform_da"),
    ]
    rows = []
    for method, p_np, settlement in settlements:
        row = {}
        row.update(scenario_meta)
        row.update(economics(net, types_np, x_np, p_np, P_np))
        row.update(security_metrics(
            net, x_np, P_np, offer, ess_net_np=ess_net_np, statuses=statuses))
        row.update({
            "method": method,
            "category": "bid_dependent_opf",
            "stage": "feasible",
            "settlement": settlement,
            "eval_seed": eval_seed,
        })
        rows.append(row)
    return rows


def aggregate_summary(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["stage"], row["scenario_group"])].append(row)

    def mean(group, key):
        return float(np.mean([float(r.get(key, 0.0)) for r in group]))

    def std(group, key):
        values = [float(r.get(key, 0.0)) for r in group]
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    def max_value(group, key):
        return float(max(float(r.get(key, 0.0)) for r in group))

    def min_value(group, key):
        return float(min(float(r.get(key, 0.0)) for r in group))

    out = []
    for (method, stage, scenario_group), group in sorted(grouped.items()):
        out.append({
            "method": method,
            "stage": stage,
            "scenario_group": scenario_group,
            "n_rows": len(group),
            "feasible_rate_mean": mean(group, "feasible_rate"),
            "feasible_all_count": int(sum(int(r.get("feasible_flag", 0))
                                          for r in group)),
            "procurement_cost_mean": mean(group, "procurement_cost"),
            "procurement_cost_std": std(group, "procurement_cost"),
            "info_rent_mean": mean(group, "info_rent"),
            "info_rent_std": std(group, "info_rent"),
            "social_cost_true_mean": mean(group, "social_cost_true"),
            "utility_min_min": min_value(group, "utility_min"),
            "mt_floor_gap_mwh_mean": mean(group, "mt_floor_gap_mwh"),
            "mt_floor_gap_mwh_max": max_value(group, "mt_floor_gap_mwh"),
            "postprocess_mt_slack_mwh_mean": mean(
                group, "postprocess_mt_slack_mwh"),
            "positive_adjustment_mwh_mean": mean(
                group, "positive_adjustment_mwh"),
            "positive_adjustment_mwh_std": std(
                group, "positive_adjustment_mwh"),
            "line_violation_max_mw": max_value(group, "line_violation_max_mw"),
            "voltage_violation_max_pu": max_value(
                group, "voltage_violation_max_pu"),
            "power_balance_residual_max_mw": max_value(
                group, "power_balance_residual_max_mw"),
            "grid_import_mwh_mean": mean(group, "grid_import_mwh"),
            "der_energy_mwh_mean": mean(group, "der_energy_mwh"),
        })
    return out


def deterministic_curve(rows: list) -> list:
    out = []
    for row in rows:
        if row.get("scenario_kind") != "uniform_scale":
            continue
        out.append({col: row.get(col, "") for col in CURVE_COLUMNS})
    return sorted(out, key=lambda r: (
        str(r["method"]), str(r["stage"]), float(r["load_scale"])))


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
    parser.add_argument("--eval-seeds", nargs="*", type=int,
                        default=[20260426])
    parser.add_argument("--scenario-seed", type=int, default=20260507)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--scale-levels", nargs="*", type=float,
                        default=[0.8, 0.9, 1.0, 1.1, 1.2])
    parser.add_argument("--uncertainty-bands", nargs="*", default=[
        "mild:0.95:1.05",
        "medium:0.90:1.10",
        "high:0.80:1.20",
    ])
    parser.add_argument("--random-scenarios-per-band", type=int, default=1)
    parser.add_argument("--disable-smoothing", action="store_true")
    parser.add_argument("--fixed-price-ratios", nargs="*", type=float,
                        default=[0.7])
    parser.add_argument("--skip-fixed-price", action="store_true")
    parser.add_argument("--skip-public-context-baseline", action="store_true")
    parser.add_argument("--skip-bid-opf-baseline", action="store_true")
    parser.add_argument("--skip-social-optimum", action="store_true")
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--feasibility-tol", type=float, default=1e-4)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = (os.path.abspath(args.run) if args.run
               else default_run_for_checkpoint(_ROOT, args.checkpoint))
    out_dir = args.out_dir or os.path.join(_THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    base_net = build_network_multi(
        constant_price=False,
        ctrl_min_ratio=args.ctrl_min_ratio,
    )
    scenarios = build_scenarios(base_net, args)
    print(f"Generated {len(scenarios)} load scenarios.")

    all_rows = []
    total = len(scenarios) * len(args.eval_seeds)
    counter = 0
    for scenario in scenarios:
        scenario_net = apply_load_multiplier(
            base_net, scenario["load_multiplier"])
        scenario_meta = scenario_metadata(scenario, scenario_net)
        source_types = classify_sources(scenario_net)
        postprocessor = SecurityPostProcessor(
            scenario_net,
            allow_mt_security_uplift=True,
            adjustment_weight=args.adjustment_weight,
            settlement_weight=args.settlement_weight,
            mt_slack_weight=args.mt_slack_weight,
        )

        for seed in args.eval_seeds:
            counter += 1
            print(
                f"Scenario {counter}/{total}: "
                f"{scenario_meta['scenario_id']} seed={seed}"
            )
            prior = DERTypePriorMulti(scenario_net)
            torch.manual_seed(int(seed))
            types = prior.sample(args.samples, device="cpu")

            if not args.skip_social_optimum:
                print("  evaluating constrained social optimum...")
                all_rows.append(evaluate_social_optimum(
                    scenario_net, types, scenario_meta, seed))

            if not args.skip_bid_opf_baseline:
                print("  evaluating bid-dependent OPF baselines...")
                all_rows.extend(evaluate_bid_opf(
                    scenario_net, types, scenario_meta, seed))

            if not args.skip_fixed_price:
                for ratio in args.fixed_price_ratios:
                    print(f"  evaluating fixed price ratio={ratio:.3f}...")
                    fixed = FixedPostedPriceMechanism(
                        scenario_net,
                        ratio=ratio,
                        pi_buyback_ratio=args.pi_buyback_ratio,
                        type_cap_ratio=TYPE_CAP_RATIO,
                    )
                    all_rows.extend(evaluate_posted_price_method(
                        f"fixed_price_ratio_{ratio:.2f}",
                        "fixed_price",
                        fixed,
                        scenario_net,
                        types,
                        postprocessor,
                        scenario_meta,
                        seed,
                    ))

            print("  evaluating learned peer posted price...")
            peer = load_mechanism(
                scenario_net, run_dir, args.checkpoint, args,
                use_peer_bid_context=True)
            all_rows.extend(evaluate_posted_price_method(
                f"learned_peer_{args.checkpoint}",
                "learned_posted_price",
                peer,
                scenario_net,
                types,
                postprocessor,
                scenario_meta,
                seed,
                checkpoint=args.checkpoint,
                run_dir=run_dir,
            ))

            if not args.skip_public_context_baseline:
                print("  evaluating public-only posted price...")
                public = load_mechanism(
                    scenario_net, run_dir, args.checkpoint, args,
                    use_peer_bid_context=False)
                all_rows.extend(evaluate_posted_price_method(
                    f"learned_public_only_{args.checkpoint}",
                    "learned_posted_price",
                    public,
                    scenario_net,
                    types,
                    postprocessor,
                    scenario_meta,
                    seed,
                    checkpoint=args.checkpoint,
                    run_dir=run_dir,
                ))

    summary_rows = aggregate_summary(all_rows)
    curve_rows = deterministic_curve(all_rows)

    detailed_path = os.path.join(out_dir, "robustness_detailed.csv")
    summary_path = os.path.join(out_dir, "robustness_summary.csv")
    summary_md_path = os.path.join(out_dir, "robustness_summary.md")
    curve_path = os.path.join(out_dir, "load_scale_curve.csv")
    config_path = os.path.join(out_dir, "robustness_config.json")
    write_csv(detailed_path, all_rows)
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_markdown(summary_md_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(curve_path, curve_rows, CURVE_COLUMNS)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["n_scenarios"] = len(scenarios)
        config["scenarios"] = [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in scenario.items()
            }
            for scenario in scenarios
        ]
        json.dump(config, f, indent=2)

    print(f"\nSaved detailed rows : {detailed_path}")
    print(f"Saved summary CSV   : {summary_path}")
    print(f"Saved summary MD    : {summary_md_path}")
    print(f"Saved scale curve   : {curve_path}")
    print(f"Saved config        : {config_path}")
    print("\nRobustness summary:")
    for row in summary_rows:
        print(
            f"  {row['method']:<42} {row['stage']:<8} "
            f"{row['scenario_group']:<14} "
            f"feas={fmt(row['feasible_rate_mean']):<8} "
            f"cost={fmt(row['procurement_cost_mean']):<10} "
            f"rent={fmt(row['info_rent_mean']):<10} "
            f"corr={fmt(row['positive_adjustment_mwh_mean'])}"
        )


if __name__ == "__main__":
    main()
