#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_security.py
-----------------------
Deterministic security correction for the posted-price procurement pipeline.

The learned mechanism produces a preliminary dispatch from posted prices and
DER offer caps.  This module solves a small 24h QP after that dispatch to
enforce physical constraints, especially the MT controllable-generation floor.

Settlement remains at the same posted price rho.  If the correction increases
MT output, the added MWh are paid at rho_MT; no new bid-dependent uplift price
is introduced.
"""

from dataclasses import dataclass, field
import os
import sys
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class SecurityPostprocessResult:
    x: np.ndarray                 # [B, T, N]
    P_VPP: np.ndarray             # [B, T]
    mt_slack: np.ndarray          # [B, T]
    status: list
    positive_adjustment: np.ndarray   # [B, T, N]
    P_c: np.ndarray = None        # [B, T, n_ess]
    P_d: np.ndarray = None        # [B, T, n_ess]
    SOC: np.ndarray = None        # [B, T+1, n_ess]
    duals: dict = field(default_factory=dict)
    correction_summary: dict = field(default_factory=dict)


class SecurityPostProcessor:
    """
    Fast QP-based post-processing layer.

    By default, non-MT DERs are capped by their accepted offer caps, while MTs
    may be increased up to their physical availability for security correction.
    This is the only out-of-offer adjustment and it is still settled at rho_MT.
    """

    def __init__(self, net_multi: dict, solver: str = None,
                 allow_mt_security_uplift: bool = True,
                 adjustment_weight: float = None,
                 settlement_weight: float = 1e-3,
                 mt_slack_weight: float = 1e5,
                 ess_degrad_cost: float = 5.0,
                 enable_ess_arbitrage: bool = True):
        if not CVXPY_AVAILABLE:
            raise RuntimeError("cvxpy is required for SecurityPostProcessor")

        self.net = net_multi
        self.T = int(net_multi["T"])
        self.N = int(net_multi["n_ders"])
        self.allow_mt_security_uplift = bool(allow_mt_security_uplift)
        # When True (default), let P_VPP and ESS be jointly optimised with the
        # full day-ahead grid cost term so ESS can perform price arbitrage.
        # When False, fall back to the legacy behaviour that pins P_VPP near
        # the mechanism's preliminary value (ESS sits idle).
        self.enable_ess_arbitrage = bool(enable_ess_arbitrage)
        if adjustment_weight is None:
            # Arbitrage mode needs a stronger DER anchor (||x - x_pre||²)
            # because grid_cost (full weight) would otherwise drag DER
            # dispatch up to displace P_VPP directly. Empirically 1e3 keeps
            # the per-cell deviation below 0.05 MW while leaving ESS free.
            adjustment_weight = 1000.0 if enable_ess_arbitrage else 1.0
        self.adjustment_weight = float(adjustment_weight)
        self.settlement_weight = float(settlement_weight)
        self.mt_slack_weight = float(mt_slack_weight)
        self.ess_degrad_cost = float(ess_degrad_cost)

        if solver is None:
            for cand in ("CLARABEL", "ECOS", "SCS"):
                if cand in cp.installed_solvers():
                    solver = cand
                    break
        self.solver = solver

        self.load_profile = np.asarray(net_multi["load_profile"], dtype=np.float64)
        self.pi_DA_profile = np.asarray(net_multi["pi_DA_profile"], dtype=np.float64)
        self.x_bar_profile = np.asarray(net_multi["x_bar_profile"], dtype=np.float64)
        self.A_flow = np.asarray(net_multi["A_flow"], dtype=np.float64)
        self.A_volt = np.asarray(net_multi["A_volt"], dtype=np.float64)
        self.flow_up = np.asarray(net_multi["flow_margin_up_profile"], dtype=np.float64)
        self.flow_dn = np.asarray(net_multi["flow_margin_dn_profile"], dtype=np.float64)
        self.volt_up = np.asarray(net_multi["volt_margin_up_profile"], dtype=np.float64)
        self.volt_dn = np.asarray(net_multi["volt_margin_dn_profile"], dtype=np.float64)
        self.ctrl_min_ratio = float(net_multi.get("ctrl_min_ratio", 0.0))
        self.mt_indices = np.asarray(net_multi["mt_indices"], dtype=np.int64)

        ess = net_multi["ess_params"]
        self.n_ess = int(ess["n_ess"])
        self.ess_pmax = np.asarray(ess["ess_power_max"], dtype=np.float64)
        self.ess_cap = np.asarray(ess["ess_capacity"], dtype=np.float64)
        self.ess_eta_c = np.asarray(ess["ess_eta_c"], dtype=np.float64)
        self.ess_eta_d = np.asarray(ess["ess_eta_d"], dtype=np.float64)
        self.ess_soc_max = np.asarray(ess["ess_soc_max_pu"], dtype=np.float64) * self.ess_cap
        self.ess_soc_min = np.asarray(ess["ess_soc_min_pu"], dtype=np.float64) * self.ess_cap
        self.ess_soc_init = np.asarray(ess["ess_soc_init_pu"], dtype=np.float64) * self.ess_cap
        self.A_flow_ess = np.asarray(net_multi["A_flow_ess"], dtype=np.float64)
        self.A_volt_ess = np.asarray(net_multi["A_volt_ess"], dtype=np.float64)

        labels = net_multi.get("der_labels", [f"DER_{i}" for i in range(self.N)])
        der_types = net_multi.get("der_type", ["DER"] * self.N)
        self.source_types = [
            self._classify_source_type(label, der_type)
            for label, der_type in zip(labels, der_types)
        ]
        ordered = ["PV", "WT", "DG", "MT", "DR"]
        self.summary_types = [typ for typ in ordered if typ in self.source_types]

        self._build_problem()

    @staticmethod
    def _classify_source_type(label: str, der_type: str) -> str:
        if str(label).startswith("PV"):
            return "PV"
        if str(label).startswith("WT"):
            return "WT"
        if str(label).startswith("MT"):
            return "MT"
        if der_type == "DR":
            return "DR"
        return "DG"

    def _build_problem(self):
        T, N, n_ess = self.T, self.N, self.n_ess

        x = cp.Variable((T, N), nonneg=True)
        P_VPP = cp.Variable(T, nonneg=True)
        P_c = cp.Variable((T, n_ess), nonneg=True)
        P_d = cp.Variable((T, n_ess), nonneg=True)
        SOC = cp.Variable((T + 1, n_ess))
        mt_slack = cp.Variable(T, nonneg=True)

        x_pre = cp.Parameter((T, N))
        pvpp_pre = cp.Parameter(T)
        x_cap = cp.Parameter((T, N), nonneg=True)
        rho = cp.Parameter((T, N), nonneg=True)
        mt_aggr_cap = cp.Parameter(T, nonneg=True)

        ess_deg = self.ess_degrad_cost * cp.sum(P_c + P_d)
        slack_penalty = self.mt_slack_weight * cp.sum_squares(mt_slack)

        if self.enable_ess_arbitrage:
            # New objective: keep DER dispatch close to the mechanism's
            # preliminary x_pre, but let P_VPP and ESS freely arbitrage
            # against the (public) day-ahead price. No DER cost information
            # is used.
            adjustment = cp.sum_squares(x - x_pre)
            grid_cost = self.pi_DA_profile @ P_VPP
            obj = cp.Minimize(
                self.adjustment_weight * adjustment
                + grid_cost
                + ess_deg
                + slack_penalty
            )
        else:
            # Legacy behaviour: pin both x and P_VPP to their pre values.
            adjustment = (cp.sum_squares(x - x_pre)
                          + cp.sum_squares(P_VPP - pvpp_pre))
            settlement = (cp.sum(cp.multiply(rho, x))
                          + self.pi_DA_profile @ P_VPP)
            obj = cp.Minimize(
                self.adjustment_weight * adjustment
                + self.settlement_weight * settlement
                + ess_deg
                + slack_penalty
            )

        cons = []
        dual_cons = {
            "balance": [],
            "flow_up": [],
            "flow_down": [],
            "voltage_up": [],
            "voltage_down": [],
            "der_cap": [],
            "mt_floor": [],
            "ess_charge_cap": [],
            "ess_discharge_cap": [],
        }
        for t in range(T):
            ess_net_t = cp.sum(P_d[t, :]) - cp.sum(P_c[t, :])
            c = (P_VPP[t] + cp.sum(x[t, :]) + ess_net_t
                 == float(self.load_profile[t]))
            cons.append(c)
            dual_cons["balance"].append(c)

            flow_der = self.A_flow @ x[t, :]
            flow_ess = self.A_flow_ess @ (P_d[t, :] - P_c[t, :])
            c = flow_der + flow_ess <= self.flow_up[t]
            cons.append(c)
            dual_cons["flow_up"].append(c)
            c = flow_der + flow_ess >= -self.flow_dn[t]
            cons.append(c)
            dual_cons["flow_down"].append(c)

            volt_der = self.A_volt @ x[t, :]
            volt_ess = self.A_volt_ess @ (P_d[t, :] - P_c[t, :])
            c = volt_der + volt_ess <= self.volt_up[t]
            cons.append(c)
            dual_cons["voltage_up"].append(c)
            c = volt_der + volt_ess >= -self.volt_dn[t]
            cons.append(c)
            dual_cons["voltage_down"].append(c)

            c = x[t, :] <= x_cap[t, :]
            cons.append(c)
            dual_cons["der_cap"].append(c)

            mt_floor = self.ctrl_min_ratio * float(self.load_profile[t])
            c = cp.sum(x[t, self.mt_indices]) + mt_slack[t] >= mt_floor
            cons.append(c)
            dual_cons["mt_floor"].append(c)

            # Aggregate MT cap: MTs cannot exceed max(offer_sum, floor).
            # Prevents the grid-cost objective from over-dispatching MTs
            # beyond what security requires.
            c = cp.sum(x[t, self.mt_indices]) <= mt_aggr_cap[t]
            cons.append(c)

            c = P_c[t, :] <= self.ess_pmax
            cons.append(c)
            dual_cons["ess_charge_cap"].append(c)
            c = P_d[t, :] <= self.ess_pmax
            cons.append(c)
            dual_cons["ess_discharge_cap"].append(c)

        for j in range(n_ess):
            cons.append(SOC[0, j] == float(self.ess_soc_init[j]))
            for t in range(T):
                cons.append(SOC[t + 1, j] == SOC[t, j]
                            + float(self.ess_eta_c[j]) * P_c[t, j]
                            - P_d[t, j] / float(self.ess_eta_d[j]))
                cons.append(SOC[t + 1, j] >= float(self.ess_soc_min[j]))
                cons.append(SOC[t + 1, j] <= float(self.ess_soc_max[j]))
            cons.append(SOC[T, j] >= float(self.ess_soc_init[j]))

        self._x = x
        self._P_VPP = P_VPP
        self._P_c = P_c
        self._P_d = P_d
        self._SOC = SOC
        self._mt_slack = mt_slack
        self._x_pre = x_pre
        self._pvpp_pre = pvpp_pre
        self._x_cap = x_cap
        self._rho = rho
        self._mt_aggr_cap = mt_aggr_cap
        self._dual_cons = dual_cons
        self._prob = cp.Problem(obj, cons)

    def _effective_cap(self, offer_cap: np.ndarray) -> np.ndarray:
        """Per-DER per-hour upper bound used by the QP.

        Non-MT DERs are always capped at their accepted offer cap (preserves
        IR / IC, since the mechanism's rho was learned against these caps).
        MT DERs are individually allowed up to their physical availability,
        but an extra aggregate constraint in ``_build_problem`` caps the
        sum of MT dispatch at ``max(floor, sum(offer_cap_MT))`` so the
        ESS-arbitrage objective cannot over-dispatch MTs beyond what
        security requires.
        """
        cap = np.minimum(np.maximum(offer_cap, 0.0), self.x_bar_profile)
        if not self.allow_mt_security_uplift:
            return cap
        cap = cap.copy()
        cap[:, self.mt_indices] = self.x_bar_profile[:, self.mt_indices]
        return cap

    def _mt_aggregate_cap(self, offer_cap: np.ndarray) -> np.ndarray:
        """Per-hour cap on the *sum* of MT dispatch.

        The QP can lift MTs up to their offer cap freely, and additionally
        up to whatever is needed to satisfy the local controllable floor.
        It cannot lift them higher than that just to displace P_VPP.
        """
        offer_cap = np.maximum(np.asarray(offer_cap, dtype=np.float64), 0.0)
        mt_offer_sum_t = offer_cap[:, self.mt_indices].sum(axis=1)
        floor_t = self.ctrl_min_ratio * self.load_profile
        return np.maximum(mt_offer_sum_t, floor_t)

    def _dual_specs(self) -> dict:
        return {
            "balance": (self.T,),
            "flow_up": (self.T, self.A_flow.shape[0]),
            "flow_down": (self.T, self.A_flow.shape[0]),
            "voltage_up": (self.T, self.A_volt.shape[0]),
            "voltage_down": (self.T, self.A_volt.shape[0]),
            "der_cap": (self.T, self.N),
            "mt_floor": (self.T,),
            "ess_charge_cap": (self.T, self.n_ess),
            "ess_discharge_cap": (self.T, self.n_ess),
        }

    def _empty_duals(self) -> dict:
        return {
            name: np.full(shape, np.nan, dtype=np.float64)
            for name, shape in self._dual_specs().items()
        }

    def _collect_duals(self) -> dict:
        duals = self._empty_duals()
        for name, constraints in self._dual_cons.items():
            out = duals[name]
            for t, constr in enumerate(constraints):
                value = constr.dual_value
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape == ():
                    out[t] = float(arr)
                else:
                    out[t] = arr
        return duals

    @staticmethod
    def _nan_abs_mean(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or np.isnan(arr).all():
            return 0.0
        return float(np.nanmean(np.abs(arr)))

    @staticmethod
    def _nan_abs_max(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or np.isnan(arr).all():
            return 0.0
        return float(np.nanmax(np.abs(arr)))

    def _build_correction_summary(self, x_pre: np.ndarray, x_post: np.ndarray,
                                  P_VPP_pre: np.ndarray, P_VPP_post: np.ndarray,
                                  rho: np.ndarray, offer_cap: np.ndarray,
                                  mt_slack: np.ndarray, duals: dict) -> dict:
        delta_x = x_post - x_pre
        pos_adj = np.maximum(delta_x, 0.0)
        neg_adj = np.maximum(-delta_x, 0.0)
        delta_pvpp = P_VPP_post - P_VPP_pre

        by_time = {
            "positive_adjustment_mwh": pos_adj.sum(axis=2).mean(axis=0),
            "negative_adjustment_mwh": neg_adj.sum(axis=2).mean(axis=0),
            "net_der_adjustment_mwh": delta_x.sum(axis=2).mean(axis=0),
            "pvpp_adjustment_mwh": delta_pvpp.mean(axis=0),
            "mt_slack_mwh": np.nanmean(mt_slack, axis=0),
        }
        if duals:
            by_time.update({
                "balance_dual_abs_mean": np.nanmean(np.abs(duals["balance"]), axis=0),
                "mt_floor_dual_abs_mean": np.nanmean(np.abs(duals["mt_floor"]), axis=0),
                "flow_dual_abs_mean": np.nanmean(
                    np.abs(duals["flow_up"]) + np.abs(duals["flow_down"]),
                    axis=(0, 2)),
                "voltage_dual_abs_mean": np.nanmean(
                    np.abs(duals["voltage_up"]) + np.abs(duals["voltage_down"]),
                    axis=(0, 2)),
            })

        by_type = {}
        for typ in self.summary_types:
            mask = np.asarray([src == typ for src in self.source_types])
            if not mask.any():
                continue
            pos_typ = pos_adj[:, :, mask]
            neg_typ = neg_adj[:, :, mask]
            delta_typ = delta_x[:, :, mask]
            rho_typ = rho[:, :, mask]
            offer_typ = offer_cap[:, :, mask]
            post_typ = x_post[:, :, mask]
            security_uplift = np.maximum(post_typ - offer_typ, 0.0)
            by_type[typ] = {
                "positive_adjustment_mwh": float(pos_typ.sum(axis=(1, 2)).mean()),
                "negative_adjustment_mwh": float(neg_typ.sum(axis=(1, 2)).mean()),
                "net_adjustment_mwh": float(delta_typ.sum(axis=(1, 2)).mean()),
                "additional_payment": float((rho_typ * pos_typ).sum(axis=(1, 2)).mean()),
                "security_uplift_mwh": float(security_uplift.sum(axis=(1, 2)).mean()),
                "security_uplift_payment": float(
                    (rho_typ * security_uplift).sum(axis=(1, 2)).mean()),
            }

        line_duals = np.concatenate(
            [duals["flow_up"].reshape(duals["flow_up"].shape[0], -1),
             duals["flow_down"].reshape(duals["flow_down"].shape[0], -1)],
            axis=1,
        ) if duals else np.array([])
        voltage_duals = np.concatenate(
            [duals["voltage_up"].reshape(duals["voltage_up"].shape[0], -1),
             duals["voltage_down"].reshape(duals["voltage_down"].shape[0], -1)],
            axis=1,
        ) if duals else np.array([])

        totals = {
            "positive_adjustment_mwh": float(pos_adj.sum(axis=(1, 2)).mean()),
            "negative_adjustment_mwh": float(neg_adj.sum(axis=(1, 2)).mean()),
            "pvpp_adjustment_l1_mwh": float(np.abs(delta_pvpp).sum(axis=1).mean()),
            "additional_payment": float((rho * pos_adj).sum(axis=(1, 2)).mean()),
            "mt_slack_mwh": float(np.nansum(mt_slack, axis=1).mean()),
            "balance_dual_abs_mean": self._nan_abs_mean(duals.get("balance", [])),
            "mt_floor_dual_abs_mean": self._nan_abs_mean(duals.get("mt_floor", [])),
            "mt_floor_dual_abs_max": self._nan_abs_max(duals.get("mt_floor", [])),
            "line_dual_abs_max": self._nan_abs_max(line_duals),
            "voltage_dual_abs_max": self._nan_abs_max(voltage_duals),
            "der_cap_dual_abs_max": self._nan_abs_max(duals.get("der_cap", [])),
        }
        return {
            "totals": totals,
            "by_time": by_time,
            "by_type": by_type,
        }

    def solve_one(self, x_pre: np.ndarray, P_VPP_pre: np.ndarray,
                  rho: np.ndarray, offer_cap: np.ndarray,
                  return_duals: bool = False):
        x_pre = np.asarray(x_pre, dtype=np.float64)
        P_VPP_pre = np.asarray(P_VPP_pre, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        cap = self._effective_cap(np.asarray(offer_cap, dtype=np.float64))

        self._x_pre.value = x_pre
        self._pvpp_pre.value = P_VPP_pre
        self._rho.value = np.maximum(rho, 0.0)
        self._x_cap.value = cap
        self._mt_aggr_cap.value = self._mt_aggregate_cap(offer_cap)

        try:
            self._prob.solve(solver=self.solver, warm_start=True)
        except Exception:
            fallback = "ECOS" if "ECOS" in cp.installed_solvers() else "SCS"
            self._prob.solve(solver=fallback, warm_start=True)

        if self._x.value is None:
            result = (x_pre, P_VPP_pre, np.full(self.T, np.nan), "fallback")
            if return_duals:
                return (*result, self._empty_duals())
            return result

        result = (self._x.value.astype(np.float64),
                  self._P_VPP.value.astype(np.float64),
                  self._mt_slack.value.astype(np.float64),
                  self._prob.status)
        if return_duals:
            return (*result, self._collect_duals())
        return result

    def process_batch(self, x_pre: np.ndarray, P_VPP_pre: np.ndarray,
                      rho: np.ndarray, offer_cap: np.ndarray) -> SecurityPostprocessResult:
        x_pre = np.asarray(x_pre, dtype=np.float64)
        P_VPP_pre = np.asarray(P_VPP_pre, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        offer_cap = np.asarray(offer_cap, dtype=np.float64)

        B = x_pre.shape[0]
        x_out = np.zeros_like(x_pre, dtype=np.float64)
        pvpp_out = np.zeros_like(P_VPP_pre, dtype=np.float64)
        slack_out = np.zeros((B, self.T), dtype=np.float64)
        pc_out = np.zeros((B, self.T, self.n_ess), dtype=np.float64)
        pd_out = np.zeros((B, self.T, self.n_ess), dtype=np.float64)
        soc_out = np.zeros((B, self.T + 1, self.n_ess), dtype=np.float64)
        statuses = []
        dual_out = None

        for b in range(B):
            x_b, pvpp_b, slack_b, status, dual_b = self.solve_one(
                x_pre[b], P_VPP_pre[b], rho[b], offer_cap[b],
                return_duals=True)
            x_out[b] = x_b
            pvpp_out[b] = pvpp_b
            slack_out[b] = slack_b
            if status == "fallback" or self._P_c.value is None:
                pc_out[b] = np.nan
                pd_out[b] = np.nan
                soc_out[b] = np.nan
            else:
                pc_out[b] = self._P_c.value.astype(np.float64)
                pd_out[b] = self._P_d.value.astype(np.float64)
                soc_out[b] = self._SOC.value.astype(np.float64)
            statuses.append(status)
            if dual_out is None:
                dual_out = {
                    name: np.zeros((B,) + arr.shape, dtype=np.float64)
                    for name, arr in dual_b.items()
                }
            for name, arr in dual_b.items():
                dual_out[name][b] = arr

        pos_adj = np.maximum(x_out - x_pre, 0.0)
        if dual_out is None:
            dual_out = {
                name: np.zeros((B,) + shape, dtype=np.float64)
                for name, shape in self._dual_specs().items()
            }
        summary = self._build_correction_summary(
            x_pre=x_pre,
            x_post=x_out,
            P_VPP_pre=P_VPP_pre,
            P_VPP_post=pvpp_out,
            rho=rho,
            offer_cap=offer_cap,
            mt_slack=slack_out,
            duals=dual_out,
        )
        return SecurityPostprocessResult(
            x=x_out.astype(np.float32),
            P_VPP=pvpp_out.astype(np.float32),
            mt_slack=slack_out.astype(np.float32),
            status=statuses,
            positive_adjustment=pos_adj.astype(np.float32),
            P_c=pc_out.astype(np.float32),
            P_d=pd_out.astype(np.float32),
            SOC=soc_out.astype(np.float32),
            duals={k: v.astype(np.float32) for k, v in dual_out.items()},
            correction_summary=summary,
        )


if __name__ == "__main__":
    _THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_THIS, "..", "network"))
    from vpp_network_multi import build_network_multi

    net = build_network_multi(constant_price=False, ctrl_min_ratio=0.15)
    pp = SecurityPostProcessor(net)
    print(f"SecurityPostProcessor ready: T={pp.T}, N={pp.N}, solver={pp.solver}")
