#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_common_multi.py
------------------------
Shared infrastructure for multi-period baselines.

Provides:
  * JointQPMulti        — Joint 24h QP with ESS and DR cap
  * solve_joint_qp(...) — high-level helper used by all baselines
  * evaluate_baseline   — unified eval loop returning a metrics dict
  * compute_regret_multi— 1-shot adversarial misreport regret estimate

Constraints (matching DC3OPFLayerMulti):
  - Per-t power balance  Σ_i x_{i,t} + Σ_j(P_d_{j,t} - P_c_{j,t}) + P_VPP_t == load[t]
  - Per-t line / voltage limits (including ESS bus injection)
  - Per-t capacity         0 <= x_{i,t} <= x_bar_profile[t, i]
  - (DR removed from this scenario)
  - ESS power limits       0 <= P_c_{j,t}, P_d_{j,t} <= ess_power_max[j]
  - ESS SOC dynamics       SOC_{j,t+1} = SOC_{j,t} + η_c P_c_{j,t} - P_d_{j,t}/η_d
  - ESS SOC bounds         soc_min_j <= SOC_{j,t} <= soc_max_j
  - ESS degradation cost   in objective: Σ_t Σ_j degrad * (P_c + P_d)

Mechanism interface (matches VPPMechanismMulti.forward):
  forward(bids: [B, N, 2]) -> (x [B,T,N], price_or_None [B,N], p [B,N], P_VPP [B,T])
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("[baseline_common_multi] WARNING: cvxpy not installed.")


# ===========================================================================
# Joint 24h QP (Phase 1a)
# ===========================================================================

class JointQPMulti:
    """
    Reusable CVXPY problem for the joint 24h dispatch **with ESS**.

    The objective is parameterised by per-(t,i) linear+quadratic cost
    coefficients so the same compiled problem can serve every baseline:

        min   Σ_t pi_DA_t * P_VPP_t
              + Σ_t Σ_i [ a_param_{i,t} * x_{i,t}^2 + b_param_{i,t} * x_{i,t} ]
              + Σ_t Σ_j degrad_j * (P_c_{j,t} + P_d_{j,t})

    ESS variables: P_c [T, n_ess], P_d [T, n_ess], SOC [T+1, n_ess]
    """

    def __init__(self, net_multi: dict, solver: str = None,
                 ess_degrad_cost: float = 5.0):
        if not CVXPY_AVAILABLE:
            raise RuntimeError("cvxpy required for JointQPMulti")
        self.net = net_multi
        self.T = int(net_multi["T"])
        self.N = int(net_multi["n_ders"])

        # Pick solver: prefer Clarabel (faster on QP), fall back to ECOS/SCS
        if solver is None:
            for cand in ("CLARABEL", "ECOS", "SCS"):
                if cand in cp.installed_solvers():
                    solver = cand
                    break
        self.solver = solver

        # Static numpy arrays
        self.load_profile  = np.asarray(net_multi["load_profile"],  dtype=np.float64)
        self.pi_DA_profile = np.asarray(net_multi["pi_DA_profile"], dtype=np.float64)
        self.x_bar_profile = np.asarray(net_multi["x_bar_profile"], dtype=np.float64)
        self.A_flow        = np.asarray(net_multi["A_flow"],        dtype=np.float64)
        self.A_volt        = np.asarray(net_multi["A_volt"],        dtype=np.float64)
        self.flow_up       = np.asarray(net_multi["flow_margin_up_profile"], dtype=np.float64)
        self.flow_dn       = np.asarray(net_multi["flow_margin_dn_profile"], dtype=np.float64)
        self.volt_up       = np.asarray(net_multi["volt_margin_up_profile"], dtype=np.float64)
        self.volt_dn       = np.asarray(net_multi["volt_margin_dn_profile"], dtype=np.float64)

        # Controllable generation constraint (local MT >= ratio × load)
        self.ctrl_min_ratio = net_multi.get("ctrl_min_ratio", 0.35)
        self.mt_qp_indices  = np.asarray(net_multi["mt_indices"], dtype=np.int64)

        # ESS parameters
        ess = net_multi["ess_params"]
        self.n_ess      = int(ess["n_ess"])
        self.ess_pmax   = np.asarray(ess["ess_power_max"], dtype=np.float64)
        self.ess_cap    = np.asarray(ess["ess_capacity"],   dtype=np.float64)
        self.ess_eta_c  = np.asarray(ess["ess_eta_c"],      dtype=np.float64)
        self.ess_eta_d  = np.asarray(ess["ess_eta_d"],      dtype=np.float64)
        self.ess_soc_max = np.asarray(ess["ess_soc_max_pu"], dtype=np.float64) * self.ess_cap
        self.ess_soc_min = np.asarray(ess["ess_soc_min_pu"], dtype=np.float64) * self.ess_cap
        self.ess_soc_init = np.asarray(ess["ess_soc_init_pu"], dtype=np.float64) * self.ess_cap
        self.ess_degrad  = ess_degrad_cost

        self.A_flow_ess = np.asarray(net_multi["A_flow_ess"], dtype=np.float64)
        self.A_volt_ess = np.asarray(net_multi["A_volt_ess"], dtype=np.float64)

        self._build_problem()

    def _build_problem(self):
        T, N, n_ess = self.T, self.N, self.n_ess

        x     = cp.Variable((T, N), nonneg=True)
        P_VPP = cp.Variable(T,       nonneg=True)
        P_c   = cp.Variable((T, n_ess), nonneg=True)
        P_d   = cp.Variable((T, n_ess), nonneg=True)
        SOC   = cp.Variable((T + 1, n_ess))

        a_par = cp.Parameter((T, N), nonneg=True)
        b_par = cp.Parameter((T, N))

        # Objective: DER cost + grid cost + ESS degradation
        quad = cp.sum(cp.multiply(a_par, cp.square(x)))
        lin  = cp.sum(cp.multiply(b_par, x))
        grid = self.pi_DA_profile @ P_VPP
        ess_deg = self.ess_degrad * cp.sum(P_c + P_d)

        obj = cp.Minimize(grid + quad + lin + ess_deg)

        cons = []
        for t in range(T):
            # Power balance: DER + ESS net + P_VPP = load
            ess_net_t = cp.sum(P_d[t, :]) - cp.sum(P_c[t, :])
            cons.append(P_VPP[t] + cp.sum(x[t, :]) + ess_net_t
                        == float(self.load_profile[t]))

            # Line flow limits (DER + ESS injection)
            flow_der = self.A_flow @ x[t, :]
            flow_ess = self.A_flow_ess @ (P_d[t, :] - P_c[t, :])
            cons.append(flow_der + flow_ess <=  self.flow_up[t])
            cons.append(flow_der + flow_ess >= -self.flow_dn[t])

            # Voltage limits (DER + ESS injection)
            volt_der = self.A_volt @ x[t, :]
            volt_ess = self.A_volt_ess @ (P_d[t, :] - P_c[t, :])
            cons.append(volt_der + volt_ess <=  self.volt_up[t])
            cons.append(volt_der + volt_ess >= -self.volt_dn[t])

            # DER capacity
            cons.append(x[t, :] <= self.x_bar_profile[t])

            # ESS power limits
            for j in range(n_ess):
                cons.append(P_c[t, j] <= float(self.ess_pmax[j]))
                cons.append(P_d[t, j] <= float(self.ess_pmax[j]))

        # Local controllable constraint: MT >= ctrl_min_ratio * load
        mt_idx = self.mt_qp_indices.tolist()
        for t in range(T):
            mt_floor = float(self.ctrl_min_ratio * self.load_profile[t])
            cons.append(cp.sum(x[t, mt_idx]) >= mt_floor)

        # ESS SOC dynamics and bounds
        for j in range(n_ess):
            cons.append(SOC[0, j] == float(self.ess_soc_init[j]))
            for t in range(T):
                cons.append(SOC[t + 1, j] == SOC[t, j]
                            + float(self.ess_eta_c[j]) * P_c[t, j]
                            - P_d[t, j] / float(self.ess_eta_d[j]))
                cons.append(SOC[t + 1, j] >= float(self.ess_soc_min[j]))
                cons.append(SOC[t + 1, j] <= float(self.ess_soc_max[j]))
            # Terminal constraint: ESS must return to initial SOC
            # (sustainable daily cycling, no one-shot energy dumping)
            cons.append(SOC[T, j] >= float(self.ess_soc_init[j]))

        self._x      = x
        self._P_VPP  = P_VPP
        self._P_c    = P_c
        self._P_d    = P_d
        self._SOC    = SOC
        self._a_par  = a_par
        self._b_par  = b_par
        self._prob   = cp.Problem(obj, cons)

    def solve(self, a_param: np.ndarray, b_param: np.ndarray):
        """
        Parameters
        ----------
        a_param : [T, N] non-negative
        b_param : [T, N]

        Returns
        -------
        x_np    : [T, N] float64
        pvpp_np : [T]    float64
        status  : str
        """
        a_param = np.clip(np.asarray(a_param, dtype=np.float64), 0.0, None)
        b_param = np.asarray(b_param, dtype=np.float64)
        self._a_par.value = a_param
        self._b_par.value = b_param
        try:
            self._prob.solve(solver=self.solver, warm_start=True)
        except Exception as e:
            print(f"[JointQPMulti] solver {self.solver} failed: {e}; retrying ECOS")
            self._prob.solve(solver=cp.ECOS)

        if self._x.value is None:
            x_np    = np.minimum(self.x_bar_profile, self.load_profile[:, None] / max(self.N, 1))
            pvpp_np = np.maximum(self.load_profile - x_np.sum(axis=1), 0.0)
            return x_np, pvpp_np, "fallback"

        return (self._x.value.astype(np.float64),
                self._P_VPP.value.astype(np.float64),
                self._prob.status)

    @property
    def last_ess(self):
        """Return ESS solution from the last solve() call, or None."""
        if self._P_c.value is None:
            return None
        return dict(
            P_c=self._P_c.value.astype(np.float64),
            P_d=self._P_d.value.astype(np.float64),
            SOC=self._SOC.value.astype(np.float64),
        )


def solve_joint_qp_batch(qp: JointQPMulti,
                         a_per_sample: np.ndarray,
                         b_per_sample: np.ndarray):
    """
    Solve the joint QP for a batch of samples.

    a_per_sample : [B, N] (time-invariant — bids are 2D)
    b_per_sample : [B, N]
    Returns
        x_batch  : [B, T, N]
        pvpp     : [B, T]
    """
    B, N = a_per_sample.shape
    T = qp.T
    x_out    = np.zeros((B, T, N), dtype=np.float32)
    pvpp_out = np.zeros((B, T),    dtype=np.float32)

    for s in range(B):
        a_tile = np.broadcast_to(a_per_sample[s][None, :], (T, N))
        b_tile = np.broadcast_to(b_per_sample[s][None, :], (T, N))
        x_np, pvpp_np, _ = qp.solve(a_tile, b_tile)
        x_out[s]    = x_np
        pvpp_out[s] = pvpp_np
    return x_out, pvpp_out


# ===========================================================================
# Cost / utility helpers (matching VPPMechanismMulti)
# ===========================================================================

def true_cost_total(types: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Σ_t (a_i x_{i,t}^2 + b_i x_{i,t})  →  [B, N]
    types : [B, N, 2]   x : [B, T, N]
    """
    a = types[..., 0].unsqueeze(1)        # [B, 1, N]
    b = types[..., 1].unsqueeze(1)        # [B, 1, N]
    return (a * x ** 2 + b * x).sum(dim=1)


def utility(types: torch.Tensor, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return p - true_cost_total(types, x)


def system_cost(net_multi: dict, p: torch.Tensor, P_VPP: torch.Tensor) -> torch.Tensor:
    pi_DA = torch.tensor(net_multi["pi_DA_profile"], dtype=torch.float32,
                         device=P_VPP.device)
    der_payment = p.sum(dim=-1)                                    # [B]
    grid_cost   = (pi_DA.unsqueeze(0) * P_VPP).sum(dim=-1)         # [B]
    return (der_payment + grid_cost).mean()


# ===========================================================================
# Regret estimator (1-shot misreport via grid sweep on bid scaling r)
# ===========================================================================

def compute_regret_multi(mech, types: torch.Tensor,
                         r_grid: list = (0.8, 0.9, 1.0, 1.1, 1.25, 1.5),
                         device: str = "cpu") -> torch.Tensor:
    """
    Estimate per-DER regret via a uniform-scaling sweep.

    For each DER i, we sweep r over r_grid and replace bids[:, i, :] *= r,
    computing the resulting utility u_i. Regret_i = max_r relu(u_i(r) - u_i(1)).

    This is a coarse but cheap estimator; for tighter bounds use the
    gradient-ascent inner loop from the trainer. For headline numbers in
    Exp2 / Exp3 this is sufficient.

    Returns : [N]  per-DER mean regret across the batch.
    """
    types = types.to(device)
    B, N, _ = types.shape

    # Truthful utility (baseline)
    with torch.no_grad():
        x_t, _, p_t, _ = mech(types)
    u_true = utility(types, x_t, p_t)                              # [B, N]

    rgt_per_der = torch.zeros(N, device=device)

    for i in range(N):
        best_gain_i = torch.zeros(B, device=device)
        for r in r_grid:
            if abs(r - 1.0) < 1e-6:
                continue
            bids = types.clone()
            bids[:, i, :] = bids[:, i, :] * r
            with torch.no_grad():
                x_m, _, p_m, _ = mech(bids)
            u_m = utility(types, x_m, p_m)                         # [B, N]
            gain_i = torch.relu(u_m[:, i] - u_true[:, i])          # [B]
            best_gain_i = torch.maximum(best_gain_i, gain_i)
        rgt_per_der[i] = best_gain_i.mean()

    return rgt_per_der


# ===========================================================================
# Unified evaluate_baseline
# ===========================================================================

def evaluate_baseline(mech, types: torch.Tensor,
                      net_multi: dict,
                      compute_regret: bool = True,
                      device: str = "cpu") -> dict:
    """
    Returns a dict of headline metrics for one mechanism on one type set.

    Keys:
      sys_cost          : true social cost under TRUTHFUL bidding
      procurement       : Σ_i p_i  (total payment to DERs), mean over batch
      info_rent         : Σ_i (p_i - true_cost_i),         mean over batch
      vpp_budget        : sys_cost (alias kept for plot legends)
      ir_violation_rate : fraction of (sample, der) with u_i < -tol
      ir_violation_max  : worst u_i across the batch
      regret_per_der    : [N] (only if compute_regret=True)
      regret_mean       : scalar mean across DERs
      regret_max        : scalar max  across DERs
      eval_time         : seconds spent solving + evaluating
    """
    types = types.to(device)
    t0 = time.time()
    with torch.no_grad():
        x, _, p, P_VPP = mech(types)

    cost_per_der = true_cost_total(types, x)                       # [B, N]
    u_per_der    = p - cost_per_der                                # [B, N]

    sc        = system_cost(net_multi, p, P_VPP).item()
    procure   = p.sum(dim=-1).mean().item()
    info_rent = (p - cost_per_der).sum(dim=-1).mean().item()

    ir_tol = 1e-4
    ir_viol_mask = (u_per_der < -ir_tol)
    ir_rate = ir_viol_mask.float().mean().item()
    ir_max  = (-u_per_der).max().item()

    out = dict(
        sys_cost          = sc,
        procurement       = procure,
        info_rent         = info_rent,
        vpp_budget        = sc,
        ir_violation_rate = ir_rate,
        ir_violation_max  = ir_max,
    )

    if compute_regret:
        rgt = compute_regret_multi(mech, types, device=device)     # [N]
        out["regret_per_der"] = rgt.cpu().numpy()
        out["regret_mean"]    = float(rgt.mean().item())
        out["regret_max"]     = float(rgt.max().item())

    out["eval_time"] = time.time() - t0
    return out


# ===========================================================================
# Sanity check
# ===========================================================================
if __name__ == "__main__":
    _THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_THIS, ".."))
    from network.vpp_network_multi import build_network_multi

    print("=== JointQPMulti Sanity Check (with ESS) ===")
    net = build_network_multi(constant_price=False)
    qp = JointQPMulti(net)
    print(f"  Solver        : {qp.solver}")
    print(f"  vars          : x[T={qp.T}, N={qp.N}], P_VPP[T], "
          f"P_c/P_d[T, {qp.n_ess}], SOC[T+1, {qp.n_ess}]")
    print(f"  Total cons    : {len(qp._prob.constraints)}")

    # Solve once with uniform costs
    a0 = np.full((qp.T, qp.N), 1.0, dtype=np.float64)
    b0 = np.full((qp.T, qp.N), 10.0, dtype=np.float64)
    t0 = time.time()
    x_np, pvpp_np, status = qp.solve(a0, b0)
    dt = time.time() - t0
    print(f"  solve status  : {status}    time={dt*1000:.1f}ms")

    # Power balance (including ESS)
    ess = qp.last_ess
    if ess is not None:
        ess_net = ess["P_d"].sum(axis=1) - ess["P_c"].sum(axis=1)   # [T]
        bal = x_np.sum(axis=1) + ess_net + pvpp_np - qp.load_profile
        print(f"  max balance   : {np.abs(bal).max():.2e}")
        print(f"  ESS P_d total : {ess['P_d'].sum(axis=0).round(4)} MWh")
        print(f"  ESS P_c total : {ess['P_c'].sum(axis=0).round(4)} MWh")
        print(f"  ESS SOC range : [{ess['SOC'].min(axis=0).round(4)}, "
              f"{ess['SOC'].max(axis=0).round(4)}]")
    else:
        bal = x_np.sum(axis=1) + pvpp_np - qp.load_profile
        print(f"  max balance   : {np.abs(bal).max():.2e}")
        print("  ESS: solver returned None")

    print(f"  N DERs          : {qp.N} (no DR in this scenario)")
