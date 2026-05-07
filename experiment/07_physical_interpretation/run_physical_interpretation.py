#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 7: Physical Sensitivity and Learned Topology Analysis
----------------------------------------------------------------
Connect learned posted-price components to physical network signals.

The script evaluates a trained mechanism, runs the security postprocess, and
compares learned price adders with:
  - feeder depth and impedance path length,
  - line-flow and voltage sensitivity matrices,
  - OPF postprocess dual-implied marginal security values,
  - MT-floor scarcity pressure and correction burden.
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

DER_COLUMNS = [
    "method",
    "checkpoint",
    "der_idx",
    "der_label",
    "source_type",
    "bus",
    "feeder_depth",
    "path_resistance",
    "path_reactance",
    "flow_sensitivity_l1",
    "flow_sensitivity_rating_norm",
    "voltage_sensitivity_l1",
    "voltage_sensitivity_max",
    "voltage_sensitivity_own_bus",
    "rho_total_mean",
    "rho_base_mean",
    "rho_type_mean",
    "rho_security_mean",
    "rho_security_abs_mean",
    "rho_security_main_mean",
    "rho_security_residual_mean",
    "rho_scarcity_mean",
    "rho_scarcity_abs_mean",
    "rho_scarcity_main_mean",
    "rho_scarcity_residual_mean",
    "rho_peer_bid_mean",
    "pre_energy_mwh",
    "post_energy_mwh",
    "offer_cap_mwh",
    "positive_adjustment_mwh",
    "negative_adjustment_mwh",
    "security_uplift_mwh",
    "dual_security_value_mean",
    "dual_security_abs_value_mean",
    "line_dual_abs_value_mean",
    "voltage_dual_abs_value_mean",
    "der_cap_dual_abs_mean",
    "mt_floor_dual_abs_mean_if_mt",
]

TIME_COLUMNS = [
    "method",
    "checkpoint",
    "hour",
    "load_mw",
    "pi_DA",
    "mt_floor_mw",
    "mt_offer_gap_mwh",
    "positive_adjustment_mwh",
    "MT_positive_adjustment_mwh",
    "rho_total_mean",
    "rho_base_mean",
    "rho_type_mean",
    "rho_security_mean",
    "rho_security_abs_mean",
    "rho_scarcity_mean",
    "rho_scarcity_abs_mean",
    "rho_peer_bid_mean",
    "MT_rho_total_mean",
    "MT_rho_scarcity_mean",
    "MT_rho_scarcity_abs_mean",
    "line_dual_abs_mean",
    "line_dual_abs_max",
    "voltage_dual_abs_mean",
    "voltage_dual_abs_max",
    "mt_floor_dual_abs_mean",
    "mt_floor_dual_abs_max",
    "balance_dual_abs_mean",
    "dual_security_abs_mean",
]

SUMMARY_COLUMNS = [
    "method",
    "checkpoint",
    "scope",
    "x_metric",
    "y_metric",
    "pearson",
    "spearman",
    "n",
    "interpretation",
]

TOPK_COLUMNS = [
    "method",
    "checkpoint",
    "scope",
    "learned_metric",
    "physical_metric",
    "k",
    "overlap_count",
    "overlap_fraction",
    "learned_top_labels",
    "physical_top_labels",
    "overlap_labels",
]


def default_run_for_checkpoints(root: str, checkpoints: list) -> str:
    runs_dir = os.path.join(root, "runs")
    candidates = [
        os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    candidates.sort(key=os.path.getmtime, reverse=True)
    for checkpoint in checkpoints:
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
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_corr(x, y, method: str = "pearson"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan, int(x.size)
    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def corr_row(method: str, checkpoint: str, scope: str, rows: list,
             x_metric: str, y_metric: str, interpretation: str) -> dict:
    x = [row.get(x_metric, np.nan) for row in rows]
    y = [row.get(y_metric, np.nan) for row in rows]
    pearson, n = safe_corr(x, y, method="pearson")
    spearman, _ = safe_corr(x, y, method="spearman")
    return {
        "method": method,
        "checkpoint": checkpoint,
        "scope": scope,
        "x_metric": x_metric,
        "y_metric": y_metric,
        "pearson": pearson,
        "spearman": spearman,
        "n": n,
        "interpretation": interpretation,
    }


def topk_indices(values, k: int):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(arr))[0]
    if finite.size == 0:
        return []
    k = min(int(k), int(finite.size))
    order = finite[np.argsort(arr[finite])[::-1]]
    return list(order[:k])


def topk_row(method: str, checkpoint: str, scope: str, rows: list,
             learned_metric: str, physical_metric: str, k: int) -> dict:
    labels = [row["der_label"] for row in rows]
    learned = [row.get(learned_metric, np.nan) for row in rows]
    physical = [row.get(physical_metric, np.nan) for row in rows]
    idx_learned = topk_indices(learned, k)
    idx_physical = topk_indices(physical, k)
    set_l = set(idx_learned)
    set_p = set(idx_physical)
    overlap = sorted(set_l & set_p)
    denom = max(1, min(int(k), len(rows)))
    return {
        "method": method,
        "checkpoint": checkpoint,
        "scope": scope,
        "learned_metric": learned_metric,
        "physical_metric": physical_metric,
        "k": denom,
        "overlap_count": len(overlap),
        "overlap_fraction": float(len(overlap) / denom),
        "learned_top_labels": ",".join(labels[i] for i in idx_learned),
        "physical_top_labels": ",".join(labels[i] for i in idx_physical),
        "overlap_labels": ",".join(labels[i] for i in overlap),
    }


def physical_sensitivity_rows(net: dict, source_types: list) -> list:
    labels = list(net["der_labels"])
    buses = list(net["der_bus"])
    path = np.asarray(net["path_matrix"], dtype=np.float64)
    line_R = np.asarray(net["line_R"], dtype=np.float64)
    line_X = np.asarray(net["line_X"], dtype=np.float64)
    A_flow = np.asarray(net["A_flow"], dtype=np.float64)
    A_volt = np.asarray(net["A_volt"], dtype=np.float64)
    ratings = np.asarray(net["line_ratings"], dtype=np.float64)

    rows = []
    for i, label in enumerate(labels):
        bus = int(buses[i])
        path_mask = path[bus]
        rows.append({
            "der_idx": i,
            "der_label": label,
            "source_type": source_types[i],
            "bus": bus + 1,
            "bus_zero_based": bus,
            "feeder_depth": float(path_mask.sum()),
            "path_resistance": float((path_mask * line_R).sum()),
            "path_reactance": float((path_mask * line_X).sum()),
            "flow_sensitivity_l1": float(np.abs(A_flow[:, i]).sum()),
            "flow_sensitivity_rating_norm": float(
                (np.abs(A_flow[:, i]) / np.maximum(ratings, 1e-9)).sum()),
            "voltage_sensitivity_l1": float(np.abs(A_volt[:, i]).sum()),
            "voltage_sensitivity_max": float(np.abs(A_volt[:, i]).max()),
            "voltage_sensitivity_own_bus": float(A_volt[bus, i]),
        })
    return rows


def component_array(price_components: dict, key: str, fallback_shape):
    value = price_components.get(key)
    if value is None:
        return np.zeros(fallback_shape, dtype=np.float64)
    return to_numpy(value).astype(np.float64)


def dual_implied_values(net: dict, post) -> dict:
    A_flow = np.asarray(net["A_flow"], dtype=np.float64)
    A_volt = np.asarray(net["A_volt"], dtype=np.float64)
    duals = post.duals

    flow_up = np.nan_to_num(np.asarray(duals["flow_up"], dtype=np.float64))
    flow_down = np.nan_to_num(np.asarray(duals["flow_down"], dtype=np.float64))
    voltage_up = np.nan_to_num(np.asarray(duals["voltage_up"], dtype=np.float64))
    voltage_down = np.nan_to_num(np.asarray(duals["voltage_down"], dtype=np.float64))

    flow_grad = np.einsum("btl,ln->btn", flow_up - flow_down, A_flow)
    voltage_grad = np.einsum("btm,mn->btn", voltage_up - voltage_down, A_volt)
    marginal_congestion = flow_grad + voltage_grad
    security_value = -marginal_congestion
    line_abs_value = np.einsum(
        "btl,ln->btn", np.abs(flow_up) + np.abs(flow_down), np.abs(A_flow))
    voltage_abs_value = np.einsum(
        "btm,mn->btn",
        np.abs(voltage_up) + np.abs(voltage_down),
        np.abs(A_volt),
    )
    return {
        "security_value": security_value,
        "security_abs_value": np.abs(security_value),
        "line_abs_value": line_abs_value,
        "voltage_abs_value": voltage_abs_value,
        "der_cap_dual_abs": np.abs(
            np.nan_to_num(np.asarray(duals["der_cap"], dtype=np.float64))),
        "mt_floor_dual_abs": np.abs(
            np.nan_to_num(np.asarray(duals["mt_floor"], dtype=np.float64))),
        "flow_dual_abs": np.abs(flow_up) + np.abs(flow_down),
        "voltage_dual_abs": np.abs(voltage_up) + np.abs(voltage_down),
        "balance_dual_abs": np.abs(
            np.nan_to_num(np.asarray(duals["balance"], dtype=np.float64))),
    }


def build_der_rows(method: str, checkpoint: str, net: dict, source_types: list,
                   physical_rows: list, arrays: dict, post) -> list:
    comps = arrays["components"]
    rho = arrays["rho"]
    x_pre = arrays["x_pre"]
    x_post = arrays["x_post"]
    offer = arrays["offer_cap"]
    pos_adj = np.maximum(x_post - x_pre, 0.0)
    neg_adj = np.maximum(x_pre - x_post, 0.0)
    uplift = np.maximum(x_post - offer, 0.0)
    dual_values = dual_implied_values(net, post)
    mt_set = set(np.asarray(net["mt_indices"], dtype=np.int64).tolist())

    rows = []
    for base in physical_rows:
        i = int(base["der_idx"])
        row = {
            "method": method,
            "checkpoint": checkpoint,
            **{k: v for k, v in base.items() if k != "bus_zero_based"},
            "rho_total_mean": float(rho[:, :, i].mean()),
            "rho_base_mean": float(comps["rho_base"][:, :, i].mean()),
            "rho_type_mean": float(comps["rho_type"][:, :, i].mean()),
            "rho_security_mean": float(comps["rho_security"][:, :, i].mean()),
            "rho_security_abs_mean": float(np.abs(
                comps["rho_security"][:, :, i]).mean()),
            "rho_security_main_mean": float(
                comps["rho_security_main"][:, :, i].mean()),
            "rho_security_residual_mean": float(
                comps["rho_security_residual"][:, :, i].mean()),
            "rho_scarcity_mean": float(comps["rho_scarcity"][:, :, i].mean()),
            "rho_scarcity_abs_mean": float(np.abs(
                comps["rho_scarcity"][:, :, i]).mean()),
            "rho_scarcity_main_mean": float(
                comps["rho_scarcity_main"][:, :, i].mean()),
            "rho_scarcity_residual_mean": float(
                comps["rho_scarcity_residual"][:, :, i].mean()),
            "rho_peer_bid_mean": float(comps["rho_peer_bid"][:, :, i].mean()),
            "pre_energy_mwh": float(x_pre[:, :, i].sum(axis=1).mean()),
            "post_energy_mwh": float(x_post[:, :, i].sum(axis=1).mean()),
            "offer_cap_mwh": float(offer[:, :, i].sum(axis=1).mean()),
            "positive_adjustment_mwh": float(pos_adj[:, :, i].sum(axis=1).mean()),
            "negative_adjustment_mwh": float(neg_adj[:, :, i].sum(axis=1).mean()),
            "security_uplift_mwh": float(uplift[:, :, i].sum(axis=1).mean()),
            "dual_security_value_mean": float(
                dual_values["security_value"][:, :, i].mean()),
            "dual_security_abs_value_mean": float(
                dual_values["security_abs_value"][:, :, i].mean()),
            "line_dual_abs_value_mean": float(
                dual_values["line_abs_value"][:, :, i].mean()),
            "voltage_dual_abs_value_mean": float(
                dual_values["voltage_abs_value"][:, :, i].mean()),
            "der_cap_dual_abs_mean": float(
                dual_values["der_cap_dual_abs"][:, :, i].mean()),
            "mt_floor_dual_abs_mean_if_mt": float(
                dual_values["mt_floor_dual_abs"].mean()) if i in mt_set else 0.0,
        }
        rows.append(row)
    return rows


def mean_component_by_type(arr: np.ndarray, mask: np.ndarray, use_abs=False):
    if not mask.any():
        return np.full(arr.shape[1], np.nan, dtype=np.float64)
    values = np.abs(arr[:, :, mask]) if use_abs else arr[:, :, mask]
    return values.mean(axis=(0, 2))


def build_time_rows(method: str, checkpoint: str, net: dict,
                    source_types: list, arrays: dict, post) -> list:
    comps = arrays["components"]
    offer = arrays["offer_cap"]
    x_pre = arrays["x_pre"]
    x_post = arrays["x_post"]
    pos_adj = np.maximum(x_post - x_pre, 0.0)
    dual_values = dual_implied_values(net, post)
    mt_idx = np.asarray(net["mt_indices"], dtype=np.int64)
    mt_mask = np.asarray([typ == "MT" for typ in source_types])
    load = np.asarray(net["load_profile"], dtype=np.float64)
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float64)
    floor = float(net["ctrl_min_ratio"]) * load
    mt_offer = offer[:, :, mt_idx].sum(axis=2)
    mt_offer_gap = np.maximum(floor[None, :] - mt_offer, 0.0)

    rows = []
    for t in range(int(net["T"])):
        rows.append({
            "method": method,
            "checkpoint": checkpoint,
            "hour": t,
            "load_mw": float(load[t]),
            "pi_DA": float(pi[t]),
            "mt_floor_mw": float(floor[t]),
            "mt_offer_gap_mwh": float(mt_offer_gap[:, t].mean()),
            "positive_adjustment_mwh": float(pos_adj[:, t, :].sum(axis=1).mean()),
            "MT_positive_adjustment_mwh": float(
                pos_adj[:, t, mt_mask].sum(axis=1).mean()),
            "rho_total_mean": float(comps["rho_total"][:, t, :].mean()),
            "rho_base_mean": float(comps["rho_base"][:, t, :].mean()),
            "rho_type_mean": float(comps["rho_type"][:, t, :].mean()),
            "rho_security_mean": float(comps["rho_security"][:, t, :].mean()),
            "rho_security_abs_mean": float(
                np.abs(comps["rho_security"][:, t, :]).mean()),
            "rho_scarcity_mean": float(comps["rho_scarcity"][:, t, :].mean()),
            "rho_scarcity_abs_mean": float(
                np.abs(comps["rho_scarcity"][:, t, :]).mean()),
            "rho_peer_bid_mean": float(comps["rho_peer_bid"][:, t, :].mean()),
            "MT_rho_total_mean": float(comps["rho_total"][:, t, mt_mask].mean()),
            "MT_rho_scarcity_mean": float(
                comps["rho_scarcity"][:, t, mt_mask].mean()),
            "MT_rho_scarcity_abs_mean": float(
                np.abs(comps["rho_scarcity"][:, t, mt_mask]).mean()),
            "line_dual_abs_mean": finite_abs_mean(
                dual_values["flow_dual_abs"][:, t, :]),
            "line_dual_abs_max": finite_abs_max(
                dual_values["flow_dual_abs"][:, t, :]),
            "voltage_dual_abs_mean": finite_abs_mean(
                dual_values["voltage_dual_abs"][:, t, :]),
            "voltage_dual_abs_max": finite_abs_max(
                dual_values["voltage_dual_abs"][:, t, :]),
            "mt_floor_dual_abs_mean": finite_abs_mean(
                dual_values["mt_floor_dual_abs"][:, t]),
            "mt_floor_dual_abs_max": finite_abs_max(
                dual_values["mt_floor_dual_abs"][:, t]),
            "balance_dual_abs_mean": finite_abs_mean(
                dual_values["balance_dual_abs"][:, t]),
            "dual_security_abs_mean": finite_abs_mean(
                dual_values["security_abs_value"][:, t, :]),
        })
    return rows


def build_alignment_summary(method: str, checkpoint: str,
                            der_rows: list, time_rows: list) -> list:
    rows = [
        corr_row(
            method, checkpoint, "DER",
            der_rows,
            "rho_security_abs_mean",
            "voltage_sensitivity_l1",
            "Security price magnitude vs voltage sensitivity.",
        ),
        corr_row(
            method, checkpoint, "DER",
            der_rows,
            "rho_security_abs_mean",
            "flow_sensitivity_rating_norm",
            "Security price magnitude vs rating-normalized flow sensitivity.",
        ),
        corr_row(
            method, checkpoint, "DER",
            der_rows,
            "rho_security_abs_mean",
            "dual_security_abs_value_mean",
            "Security price magnitude vs postprocess dual-implied security value.",
        ),
        corr_row(
            method, checkpoint, "DER",
            der_rows,
            "rho_scarcity_abs_mean",
            "security_uplift_mwh",
            "Scarcity price magnitude vs realized out-of-offer uplift.",
        ),
        corr_row(
            method, checkpoint, "DER",
            der_rows,
            "rho_scarcity_abs_mean",
            "mt_floor_dual_abs_mean_if_mt",
            "Scarcity price magnitude vs MT-floor dual pressure.",
        ),
        corr_row(
            method, checkpoint, "time",
            time_rows,
            "rho_base_mean",
            "load_mw",
            "Base component vs load level.",
        ),
        corr_row(
            method, checkpoint, "time",
            time_rows,
            "rho_base_mean",
            "pi_DA",
            "Base component vs day-ahead price.",
        ),
        corr_row(
            method, checkpoint, "time",
            time_rows,
            "MT_rho_scarcity_abs_mean",
            "mt_floor_dual_abs_mean",
            "MT scarcity component vs MT-floor dual.",
        ),
        corr_row(
            method, checkpoint, "time",
            time_rows,
            "MT_rho_scarcity_abs_mean",
            "MT_positive_adjustment_mwh",
            "MT scarcity component vs MT postprocess correction.",
        ),
        corr_row(
            method, checkpoint, "time",
            time_rows,
            "rho_security_abs_mean",
            "voltage_dual_abs_mean",
            "Security component vs voltage dual pressure.",
        ),
        corr_row(
            method, checkpoint, "time",
            time_rows,
            "rho_security_abs_mean",
            "line_dual_abs_mean",
            "Security component vs line dual pressure.",
        ),
    ]

    mt_rows = [row for row in der_rows if row["source_type"] == "MT"]
    if len(mt_rows) >= 3:
        rows.append(corr_row(
            method, checkpoint, "DER_MT",
            mt_rows,
            "rho_scarcity_abs_mean",
            "positive_adjustment_mwh",
            "Within-MT scarcity component vs MT positive correction.",
        ))
    return rows


def build_topk_summary(method: str, checkpoint: str,
                       der_rows: list, top_k: int) -> list:
    return [
        topk_row(
            method, checkpoint, "DER",
            der_rows, "rho_security_abs_mean", "voltage_sensitivity_l1", top_k),
        topk_row(
            method, checkpoint, "DER",
            der_rows, "rho_security_abs_mean",
            "flow_sensitivity_rating_norm", top_k),
        topk_row(
            method, checkpoint, "DER",
            der_rows, "rho_security_abs_mean",
            "dual_security_abs_value_mean", top_k),
        topk_row(
            method, checkpoint, "DER",
            der_rows, "rho_scarcity_abs_mean",
            "security_uplift_mwh", top_k),
    ]


def evaluate_method(method: str, checkpoint: str, mech: VPPMechanismMulti,
                    net: dict, types: torch.Tensor,
                    postprocessor: SecurityPostProcessor,
                    physical_rows: list, source_types: list) -> dict:
    with torch.no_grad():
        x_pre, rho, _, P_pre = mech(types)
        offer_cap = mech._last_offer_cap.detach().clone()
        price_components = getattr(mech, "_last_price_components_detached", {})

    rho_np = to_numpy(rho).astype(np.float64)
    x_pre_np = to_numpy(x_pre).astype(np.float64)
    P_pre_np = to_numpy(P_pre).astype(np.float64)
    offer_np = to_numpy(offer_cap).astype(np.float64)
    post = postprocessor.process_batch(x_pre_np, P_pre_np, rho_np, offer_np)

    fallback_shape = rho_np.shape
    comps = {
        key: component_array(price_components, key, fallback_shape)
        for key in PRICE_COMPONENT_KEYS
    }
    if not np.any(comps["rho_total"]):
        comps["rho_total"] = rho_np

    arrays = {
        "rho": rho_np,
        "x_pre": x_pre_np,
        "x_post": post.x.astype(np.float64),
        "offer_cap": offer_np,
        "components": comps,
    }
    der_rows = build_der_rows(
        method, checkpoint, net, source_types, physical_rows, arrays, post)
    time_rows = build_time_rows(
        method, checkpoint, net, source_types, arrays, post)
    summary_rows = build_alignment_summary(
        method, checkpoint, der_rows, time_rows)
    topk_rows = build_topk_summary(
        method, checkpoint, der_rows, top_k=3)

    status_counts = Counter(post.status)
    run_summary = {
        "method": method,
        "checkpoint": checkpoint,
        "postprocess_status": ";".join(
            f"{k}:{v}" for k, v in sorted(status_counts.items())),
        "samples": int(types.shape[0]),
        "rho_mean": float(rho_np.mean()),
        "rho_security_abs_mean": float(np.abs(comps["rho_security"]).mean()),
        "rho_scarcity_abs_mean": float(np.abs(comps["rho_scarcity"]).mean()),
        "positive_adjustment_mwh": float(
            np.maximum(post.x.astype(np.float64) - x_pre_np, 0.0
                       ).sum(axis=(1, 2)).mean()),
        "mt_slack_mwh": float(np.nansum(post.mt_slack, axis=1).mean()),
        "mt_floor_dual_abs_mean": finite_abs_mean(post.duals["mt_floor"]),
        "voltage_dual_abs_max": finite_abs_max(
            np.abs(post.duals["voltage_up"]) + np.abs(post.duals["voltage_down"])),
        "line_dual_abs_max": finite_abs_max(
            np.abs(post.duals["flow_up"]) + np.abs(post.duals["flow_down"])),
    }

    return {
        "der_rows": der_rows,
        "time_rows": time_rows,
        "summary_rows": summary_rows,
        "topk_rows": topk_rows,
        "run_summary": run_summary,
    }


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
        if np.isnan(float(value)):
            return ""
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


def maybe_write_plots(out_dir: str, der_rows: list, time_rows: list,
                      skip_plots: bool):
    if skip_plots or not der_rows or not time_rows:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return []

    paths = []
    first_method = der_rows[0]["method"]
    first_ckpt = der_rows[0]["checkpoint"]
    drows = [
        row for row in der_rows
        if row["method"] == first_method and row["checkpoint"] == first_ckpt
    ]
    trows = [
        row for row in time_rows
        if row["method"] == first_method and row["checkpoint"] == first_ckpt
    ]

    labels = [row["der_label"] for row in drows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11, 4))
    sec = np.asarray([row["rho_security_abs_mean"] for row in drows])
    volt = np.asarray([row["voltage_sensitivity_l1"] for row in drows])
    volt_norm = volt / max(float(volt.max()), 1e-9)
    sec_norm = sec / max(float(sec.max()), 1e-9)
    ax1.bar(x - 0.18, sec_norm, width=0.36, label="security price")
    ax1.bar(x + 0.18, volt_norm, width=0.36, label="voltage sensitivity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("normalized value")
    ax1.legend()
    ax1.set_title("DER-Level Security Price vs Voltage Sensitivity")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_security_component_by_der.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    hours = np.asarray([row["hour"] for row in trows])
    scarcity = np.asarray([row["MT_rho_scarcity_abs_mean"] for row in trows])
    mt_dual = np.asarray([row["mt_floor_dual_abs_mean"] for row in trows])
    mt_adj = np.asarray([row["MT_positive_adjustment_mwh"] for row in trows])
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hours, scarcity, label="MT scarcity price", linewidth=2)
    ax1.plot(hours, mt_dual, label="MT floor dual", linewidth=2)
    ax1.plot(hours, mt_adj, label="MT correction", linewidth=2)
    ax1.set_xlabel("hour")
    ax1.set_ylabel("raw value")
    ax1.legend()
    ax1.set_title("Time-Level Scarcity Component and MT Pressure")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_time_scarcity_alignment.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        [row["voltage_sensitivity_l1"] for row in drows],
        [row["rho_security_abs_mean"] for row in drows],
    )
    for row in drows:
        ax.annotate(
            row["der_label"],
            (row["voltage_sensitivity_l1"], row["rho_security_abs_mean"]),
            fontsize=7,
        )
    ax.set_xlabel("voltage sensitivity l1")
    ax.set_ylabel("security component abs mean")
    ax.set_title("Security Price vs Physical Sensitivity")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_security_physical_scatter.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=[
        "model_best_constr.pth",
    ])
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--ctrl-min-ratio", type=float, default=0.15)
    parser.add_argument("--pi-buyback-ratio", type=float, default=0.1)
    parser.add_argument("--peer-bid-scale", type=float, default=0.25)
    parser.add_argument("--price-arch", default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--include-public-context-baseline", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--adjustment-weight", type=float, default=1.0)
    parser.add_argument("--settlement-weight", type=float, default=1e-3)
    parser.add_argument("--mt-slack-weight", type=float, default=1e5)
    parser.add_argument("--skip-plots", action="store_true")
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
    source_types = classify_sources(net)
    physical_rows = physical_sensitivity_rows(net, source_types)
    prior = DERTypePriorMulti(net)
    torch.manual_seed(args.seed)
    types = prior.sample(args.samples, device="cpu")
    postprocessor = SecurityPostProcessor(
        net,
        allow_mt_security_uplift=True,
        adjustment_weight=args.adjustment_weight,
        settlement_weight=args.settlement_weight,
        mt_slack_weight=args.mt_slack_weight,
    )

    all_der_rows = []
    all_time_rows = []
    all_summary_rows = []
    all_topk_rows = []
    run_summaries = []

    for checkpoint in args.checkpoints:
        ckpt_path = os.path.join(run_dir, checkpoint)
        if not os.path.exists(ckpt_path):
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        print(f"Evaluating physical interpretation for {checkpoint}...")
        peer = load_mechanism(
            net, run_dir, checkpoint, args, use_peer_bid_context=True)
        result = evaluate_method(
            "learned_peer_posted_price",
            checkpoint,
            peer,
            net,
            types,
            postprocessor,
            physical_rows,
            source_types,
        )
        all_der_rows.extend(result["der_rows"])
        all_time_rows.extend(result["time_rows"])
        all_summary_rows.extend(result["summary_rows"])
        all_topk_rows.extend([
            {**row, "k": min(args.top_k, int(row["k"]))}
            for row in result["topk_rows"]
        ])
        run_summaries.append(result["run_summary"])

        if args.include_public_context_baseline:
            print(f"Evaluating public-context baseline for {checkpoint}...")
            public = load_mechanism(
                net, run_dir, checkpoint, args, use_peer_bid_context=False)
            result = evaluate_method(
                "learned_public_only_posted_price",
                checkpoint,
                public,
                net,
                types,
                postprocessor,
                physical_rows,
                source_types,
            )
            all_der_rows.extend(result["der_rows"])
            all_time_rows.extend(result["time_rows"])
            all_summary_rows.extend(result["summary_rows"])
            all_topk_rows.extend([
                {**row, "k": min(args.top_k, int(row["k"]))}
                for row in result["topk_rows"]
            ])
            run_summaries.append(result["run_summary"])

    # Recompute top-k rows with the requested k, preserving method/checkpoint.
    all_topk_rows = []
    for method in sorted({row["method"] for row in all_der_rows}):
        for checkpoint in sorted({
            row["checkpoint"] for row in all_der_rows
            if row["method"] == method
        }):
            rows = [
                row for row in all_der_rows
                if row["method"] == method and row["checkpoint"] == checkpoint
            ]
            all_topk_rows.extend(build_topk_summary(
                method, checkpoint, rows, args.top_k))

    der_path = os.path.join(out_dir, "physical_component_by_der.csv")
    time_path = os.path.join(out_dir, "physical_alignment_by_time.csv")
    summary_path = os.path.join(out_dir, "physical_alignment_summary.csv")
    summary_md_path = os.path.join(out_dir, "physical_alignment_summary.md")
    topk_path = os.path.join(out_dir, "topology_overlap_summary.csv")
    run_summary_path = os.path.join(out_dir, "physical_run_summary.csv")
    config_path = os.path.join(out_dir, "physical_interpretation_config.json")

    write_csv(der_path, all_der_rows, DER_COLUMNS)
    write_csv(time_path, all_time_rows, TIME_COLUMNS)
    write_csv(summary_path, all_summary_rows, SUMMARY_COLUMNS)
    write_markdown(summary_md_path, all_summary_rows, SUMMARY_COLUMNS)
    write_csv(topk_path, all_topk_rows, TOPK_COLUMNS)
    write_csv(run_summary_path, run_summaries)
    plot_paths = maybe_write_plots(out_dir, all_der_rows, all_time_rows,
                                   args.skip_plots)
    with open(config_path, "w") as f:
        config = vars(args).copy()
        config["run_dir"] = run_dir
        config["root"] = _ROOT
        config["plot_paths"] = plot_paths
        json.dump(config, f, indent=2)

    print(f"\nSaved DER component map : {der_path}")
    print(f"Saved time alignment    : {time_path}")
    print(f"Saved summary CSV       : {summary_path}")
    print(f"Saved summary MD        : {summary_md_path}")
    print(f"Saved top-k summary     : {topk_path}")
    print(f"Saved run summary       : {run_summary_path}")
    print(f"Saved config            : {config_path}")
    for path in plot_paths:
        print(f"Saved plot              : {path}")

    print("\nPhysical-alignment summary:")
    for row in all_summary_rows:
        print(
            f"  {row['method']:<34} {row['scope']:<7} "
            f"{row['x_metric']} vs {row['y_metric']} "
            f"pearson={fmt(row['pearson'])} "
            f"spearman={fmt(row['spearman'])}"
        )


if __name__ == "__main__":
    main()
