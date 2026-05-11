#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpp_network_multi.py
--------------------
Multi-period (24h day-ahead) version of the VPP distribution network.

Extends the local single-period IEEE 33-bus network in
  network/vpp_network.py
to include:
  - Time-varying load profile (load_profile [T])
  - Time-varying DER availability (x_bar_profile [T, N])
    - PV scaled by pv_factor[t]
    - WT scaled by wt_factor[t]
    - MT/DR capacity is time-invariant (equipment-level, not weather-driven)
  - Time-varying day-ahead grid price (pi_DA_profile [T])
  - ESS parameters

Topology (buses, lines, DER-to-bus mapping, impedances) is IDENTICAL to the
single-period network — only time-varying quantities are added.

Returned `net_multi` dict contains:
  * All single-period fields (topology, impedances, DER metadata, cost bounds)
  * Per-timestep quantities:
      T                          : int
      load_profile               [T]
      x_bar_profile              [T, N]
      pi_DA_profile              [T]
      flow_margin_up_profile     [T, n_lines]
      flow_margin_dn_profile     [T, n_lines]
      volt_margin_up_profile     [T, n_buses]
      volt_margin_dn_profile     [T, n_buses]
      v_base_profile             [T, n_buses]
  * ESS parameters:
      ess_params                 dict
"""

import os
import sys
import numpy as np
import torch

# Import single-period network builder (do NOT copy/fork topology — reuse)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from .vpp_network import build_33bus_network as _build_single_period_network  # noqa: E402
    from .vpp_network import _radial_path_matrix                                  # noqa: E402
except ImportError:
    sys.path.insert(0, _THIS_DIR)
    from vpp_network import build_33bus_network as _build_single_period_network   # noqa: E402
    from vpp_network import _radial_path_matrix                                   # noqa: E402

# Data module
_DATA_DIR = os.path.join(_THIS_DIR, "..", "data")
sys.path.insert(0, _DATA_DIR)
from liu_profiles import load_24h_profiles   # noqa: E402


# ---------------------------------------------------------------------------
# Multi-period network builder
# ---------------------------------------------------------------------------

def _filter_out_dr(base: dict) -> dict:
    """Remove DR-type DERs from the single-period network dict.

    Keeps only DG and MT, reducing N from 16 to 12.  All DER-indexed
    arrays are sliced accordingly.  Topology (buses, lines, impedances)
    is unchanged.
    """
    der_type = base["der_type"]
    keep = np.array([i for i, t in enumerate(der_type) if t != "DR"])
    N_new = len(keep)

    base = dict(base)  # shallow copy
    base["n_ders"]     = N_new
    base["der_bus"]    = [base["der_bus"][i] for i in keep]
    base["der_type"]   = [base["der_type"][i] for i in keep]
    base["der_labels"] = [base["der_labels"][i] for i in keep]
    base["x_bar"]      = base["x_bar"][keep]
    base["der_pf"]     = base["der_pf"][keep]
    base["tan_pf"]     = base["tan_pf"][keep]
    base["mc_a_lo"]    = base["mc_a_lo"][keep]
    base["mc_a_hi"]    = base["mc_a_hi"][keep]
    base["mc_b_lo"]    = base["mc_b_lo"][keep]
    base["mc_b_hi"]    = base["mc_b_hi"][keep]
    base["B_der"]      = base["B_der"][:, keep]
    base["A_flow"]     = base["A_flow"][:, keep]
    base["A_volt"]     = base["A_volt"][:, keep]
    return base


def build_network_multi(base_load_mw: float = None,
                        pi_DA_baseline: float = None,
                        constant_price: bool = True,
                        include_dr: bool = False,
                        pv_scale: float = 4.8,
                        mt_scale: float = 2.0,
                        ctrl_min_ratio: float = 0.15) -> dict:
    """
    Build the 24h multi-period VPP network dict.

    Parameters
    ----------
    base_load_mw : float or None
        If provided, scales the load profile so its PEAK equals this value.
    pi_DA_baseline : float or None
        Mean grid price over 24h. If None, uses single-period pi_DA.
    constant_price : bool
        If True, pi_DA_profile is flat at pi_DA_baseline.
    include_dr : bool
        If False (default), DR-type DERs are removed. N = 12 (8 DG + 4 MT).
    pv_scale : float
        Multiplier for PV nameplate capacity (default 4.8 → PV ≈ 85% load).
    mt_scale : float
        Multiplier for MT nameplate capacity (default 2.0).
    ctrl_min_ratio : float
        Minimum local controllable (MT) generation ratio per timestep (default 0.15).
        Controllable = MT + P_VPP (grid import). Stored in net dict for
        use by OPF layer and baselines.

    Returns
    -------
    net_multi : dict
    """
    base = _build_single_period_network()
    if not include_dr:
        base = _filter_out_dr(base)

    n_buses = base["n_buses"]
    n_lines = base["n_lines"]
    n_ders  = base["n_ders"]

    if base_load_mw is None:
        base_load_mw = float(base["load_total"])
    if pi_DA_baseline is None:
        pi_DA_baseline = float(base["pi_DA"])

    # ------------------------------------------------------------------
    # Time-varying profiles
    # ------------------------------------------------------------------
    profiles = load_24h_profiles(
        base_load_mw=base_load_mw,
        pi_DA_baseline=pi_DA_baseline,
        constant_price=constant_price,
    )
    T              = profiles["T"]
    load_profile   = profiles["load_profile"]       # [T]
    pv_factor      = profiles["pv_factor"]           # [T]
    wt_factor      = profiles["wt_factor"]           # [T]
    pi_DA_profile  = profiles["pi_DA_profile"]       # [T]

    # ------------------------------------------------------------------
    # x_bar_profile [T, N]: per-timestep per-DER availability
    # ------------------------------------------------------------------
    der_type = base["der_type"]
    x_bar    = base["x_bar"].copy()                  # [N]

    # Scale PV and MT nameplate capacities
    der_labels = base["der_labels"]
    for i, lbl in enumerate(der_labels):
        if lbl.startswith("PV"):
            x_bar[i] *= pv_scale
        elif lbl.startswith("MT"):
            x_bar[i] *= mt_scale

    x_bar_profile = np.zeros((T, n_ders), dtype=np.float64)
    for i, (lbl, dtype) in enumerate(zip(der_labels, der_type)):
        if lbl.startswith("PV"):
            x_bar_profile[:, i] = x_bar[i] * pv_factor
        elif lbl.startswith("WT"):
            x_bar_profile[:, i] = x_bar[i] * wt_factor
        else:
            # MT: capacity is time-invariant (equipment-level)
            x_bar_profile[:, i] = x_bar[i]

    # ------------------------------------------------------------------
    # Per-timestep network margins
    # Baseline flow depends on per-t load (since load drives baseline
    # injection at each bus).
    # ------------------------------------------------------------------
    loads_base    = base["loads"]                   # [n_buses], nominal load per bus
    loads_base_q  = _extract_loads_q(base)          # [n_buses], reactive baseline
    load_total_single = float(base["load_total"])   # scalar

    # Per-t load distribution: keep the spatial shape, scale by time multiplier
    # multiplier_t = load_profile[t] / load_total_single
    load_multiplier = load_profile / load_total_single    # [T]
    loads_profile   = loads_base[None, :] * load_multiplier[:, None]   # [T, n_buses]
    loads_profile_q = loads_base_q[None, :] * load_multiplier[:, None] # [T, n_buses]

    H = base["H"]   # [n_lines, n_buses]

    # Per-t baseline flow: H @ (-loads_t)
    P_inj_base_profile = -loads_profile                                  # [T, n_buses]
    baseline_flow_profile = P_inj_base_profile @ H.T                     # [T, n_lines]

    line_ratings = base["line_ratings"]                                  # [n_lines]

    # Per-t flow margins
    flow_margin_up_profile = line_ratings[None, :] - baseline_flow_profile
    flow_margin_dn_profile = line_ratings[None, :] + baseline_flow_profile

    # Per-t baseline voltage
    S_V_P = base["S_V_P"]                                                # [n_buses, n_buses]
    S_V_Q = base["S_V_Q"]                                                # [n_buses, n_buses]
    Q_inj_base_profile = -loads_profile_q                                # [T, n_buses]

    # v_base_t = 1 + S_V_P @ P_inj_t + S_V_Q @ Q_inj_t
    v_base_profile = np.ones((T, n_buses), dtype=np.float64)
    v_base_profile += P_inj_base_profile @ S_V_P.T
    v_base_profile += Q_inj_base_profile @ S_V_Q.T
    # Clamp substation and enforce v_min..v_max
    v_base_profile[:, 0] = 1.0
    v_base_profile = np.clip(v_base_profile, base["v_min"], 1.02)

    # Per-t voltage margins
    volt_margin_up_profile = base["v_max"] - v_base_profile
    volt_margin_dn_profile = v_base_profile - base["v_min"]

    # ------------------------------------------------------------------
    # ESS parameters
    # ------------------------------------------------------------------
    ess_params    = profiles["ess_params"]

    mt_indices = np.array([i for i, t_ in enumerate(der_type) if t_ == "MT"],
                          dtype=np.int64)
    dr_indices = np.array([i for i, t_ in enumerate(der_type) if t_ == "DR"],
                          dtype=np.int64)

    # ESS network sensitivity matrices (Phase 2)
    # ESS units inject/withdraw active power at their buses, affecting line
    # flows and voltages.  Unity power factor assumed for ESS inverters.
    ess_buses_arr = np.asarray(ess_params["ess_buses"], dtype=np.int64)
    A_flow_ess = H[:, ess_buses_arr]            # [n_lines, n_ess]
    A_volt_ess = S_V_P[:, ess_buses_arr]        # [n_buses, n_ess]

    # ------------------------------------------------------------------
    # Assemble net_multi dict: inherit single-period + add time-varying
    # ------------------------------------------------------------------
    net_multi = dict(base)   # shallow copy — all single-period fields included

    # Identify renewable (non-controllable) DER indices for controllable constraint
    re_indices = np.array([i for i, lbl in enumerate(der_labels)
                           if lbl.startswith("PV") or lbl.startswith("WT")],
                          dtype=np.int64)

    net_multi.update(dict(
        # Time dimension
        T                       = T,
        # Time-varying profiles
        load_profile            = load_profile,          # [T]
        pv_factor               = pv_factor,             # [T]
        wt_factor               = wt_factor,             # [T]
        x_bar                   = x_bar,                 # [N] (scaled PV)
        x_bar_profile           = x_bar_profile,         # [T, N]
        pi_DA_profile           = pi_DA_profile,         # [T]
        # Per-timestep network margins
        loads_profile           = loads_profile,         # [T, n_buses]
        baseline_flow_profile   = baseline_flow_profile, # [T, n_lines]
        flow_margin_up_profile  = flow_margin_up_profile,
        flow_margin_dn_profile  = flow_margin_dn_profile,
        v_base_profile          = v_base_profile,        # [T, n_buses]
        volt_margin_up_profile  = volt_margin_up_profile,
        volt_margin_dn_profile  = volt_margin_dn_profile,
        # Scenario parameters
        pv_scale                = pv_scale,
        ctrl_min_ratio          = ctrl_min_ratio,        # min controllable gen ratio
        re_indices              = re_indices,             # renewable DER indices
        mt_indices              = mt_indices,
        dr_indices              = dr_indices,
        # ESS
        ess_params              = ess_params,
        A_flow_ess              = A_flow_ess,            # [n_lines, n_ess]
        A_volt_ess              = A_volt_ess,            # [n_buses, n_ess]
    ))

    return net_multi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_loads_q(base: dict) -> np.ndarray:
    """
    Re-derive the per-bus reactive load from the single-period network dict.
    The single-period builder doesn't expose `loads_q` directly, so we
    reconstruct it from the voltage-sensitivity baseline.

    In practice, we hardcode the known IEEE 33-bus Q values (matching the
    single-period builder), since these are canonical benchmark numbers.
    """
    loads_q = np.array([
        0.000, 0.060, 0.040, 0.080, 0.030, 0.020,   # buses 0-5
        0.100, 0.100, 0.020, 0.020, 0.030,           # buses 6-10
        0.035, 0.035, 0.080, 0.010, 0.020,           # buses 11-15
        0.020, 0.040, 0.040, 0.040, 0.040,           # buses 16-20
        0.040, 0.050, 0.200, 0.200, 0.025,           # buses 21-25
        0.025, 0.020, 0.070, 0.600, 0.070,           # buses 26-30
        0.100, 0.040,                                 # buses 31-32
    ], dtype=np.float64)
    return loads_q


# ---------------------------------------------------------------------------
# Multi-period feasibility checker (wraps per-timestep single-period checker)
# ---------------------------------------------------------------------------

class VPPNetworkMulti:
    """
    Per-timestep feasibility check for multi-period dispatch.
    Accepts x of shape [B, T, N], returns max violations over B and T.
    """

    def __init__(self, net_multi: dict):
        self.net = net_multi
        self.T   = net_multi["T"]
        self.N   = net_multi["n_ders"]

        # Register as float32 tensors
        self.x_bar_profile = torch.tensor(
            net_multi["x_bar_profile"], dtype=torch.float32)               # [T, N]
        self.A_flow        = torch.tensor(
            net_multi["A_flow"], dtype=torch.float32)                      # [n_lines, N]
        self.A_volt        = torch.tensor(
            net_multi["A_volt"], dtype=torch.float32)                      # [n_buses, N]
        self.baseline_flow_profile = torch.tensor(
            net_multi["baseline_flow_profile"], dtype=torch.float32)       # [T, n_lines]
        self.v_base_profile = torch.tensor(
            net_multi["v_base_profile"], dtype=torch.float32)              # [T, n_buses]
        self.line_ratings  = torch.tensor(
            net_multi["line_ratings"], dtype=torch.float32)                # [n_lines]
        self.v_max = float(net_multi["v_max"])
        self.v_min = float(net_multi["v_min"])

    def line_flows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Total active power flow at each hour and each line.
        x : [B, T, N]  → [B, T, n_lines]
        """
        # x @ A_flow.T : [B, T, n_lines]
        incr = x @ self.A_flow.T
        return self.baseline_flow_profile.unsqueeze(0) + incr

    def line_limit_violation(self, x: torch.Tensor) -> torch.Tensor:
        """Max thermal violation across B x T x lines. Returns scalar-per-sample [B]."""
        f = self.line_flows(x)
        viol = torch.relu(f.abs() - self.line_ratings.view(1, 1, -1))
        return viol.amax(dim=(-2, -1))

    def voltage_at_buses(self, x: torch.Tensor) -> torch.Tensor:
        """
        v_t = v_base_t + A_volt @ x_t (linearised DistFlow).
        x : [B, T, N] → [B, T, n_buses]
        """
        incr = x @ self.A_volt.T
        return self.v_base_profile.unsqueeze(0) + incr

    def voltage_violation(self, x: torch.Tensor) -> torch.Tensor:
        v = self.voltage_at_buses(x)
        viol_hi = torch.relu(v - self.v_max)
        viol_lo = torch.relu(self.v_min - v)
        return torch.maximum(viol_hi.amax(dim=(-2, -1)),
                             viol_lo.amax(dim=(-2, -1)))

    def capacity_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        x_i,t in [0, x_bar_profile_{t, i}].
        x : [B, T, N] → scalar-per-sample [B]
        """
        viol_hi = torch.relu(x - self.x_bar_profile.unsqueeze(0))
        viol_lo = torch.relu(-x)
        return torch.maximum(viol_hi.amax(dim=(-2, -1)),
                             viol_lo.amax(dim=(-2, -1)))

    def is_feasible(self, x: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
        return (self.line_limit_violation(x) < tol) & \
               (self.voltage_violation(x)    < tol) & \
               (self.capacity_violation(x)   < tol)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = build_network_multi(constant_price=True)

    print("=== Multi-period VPP Network (Phase 1a) ===")
    print(f"  T              : {net['T']}")
    print(f"  n_buses        : {net['n_buses']}")
    print(f"  n_lines        : {net['n_lines']}")
    print(f"  n_ders         : {net['n_ders']}")
    print(f"  DER labels     : {net['der_labels']}")
    print(f"  pi_DA_profile  : min={net['pi_DA_profile'].min():.2f}  "
          f"mean={net['pi_DA_profile'].mean():.2f}  "
          f"max={net['pi_DA_profile'].max():.2f}")
    print(f"  load_profile   : min={net['load_profile'].min():.3f}  "
          f"mean={net['load_profile'].mean():.3f}  "
          f"max={net['load_profile'].max():.3f} MW")
    print()

    print("  x_bar_profile shape:", net['x_bar_profile'].shape)
    print(f"    (PV_4   at t=12): {net['x_bar_profile'][12, 0]:.4f}  "
          f"(nominal={net['x_bar'][0]:.4f})")
    print(f"    (PV_4   at t=2 ): {net['x_bar_profile'][2, 0]:.4f}  (night)")
    print(f"    (WT_17  at t=12): {net['x_bar_profile'][12, 4]:.4f}  "
          f"(nominal={net['x_bar'][4]:.4f})")
    print(f"    (MT_10  at t=12): {net['x_bar_profile'][12, 8]:.4f}  "
          f"(nominal={net['x_bar'][8]:.4f})  [time-invariant]")
    print()

    print("  Per-timestep network margins (sanity):")
    print(f"    flow_margin_up   shape: {net['flow_margin_up_profile'].shape}")
    print(f"    volt_margin_up   shape: {net['volt_margin_up_profile'].shape}")
    print(f"    v_base_profile   shape: {net['v_base_profile'].shape}")
    print(f"    min flow_margin_up over all t: "
          f"{net['flow_margin_up_profile'].min():.4f}")
    print(f"    max volt_margin_up over all t: "
          f"{net['volt_margin_up_profile'].max():.4f}")
    print()

    print("  DER indices:")
    print(f"    mt_indices    : {net['mt_indices']}")
    print(f"    dr_indices    : {net['dr_indices']}")
    print()

    print(f"  ess_params    : {net['ess_params']}")
    print()

    # Feasibility test
    vpp = VPPNetworkMulti(net)
    B, T, N = 2, net['T'], net['n_ders']
    x_test = torch.tensor(net['x_bar_profile'], dtype=torch.float32).unsqueeze(0).expand(B, -1, -1) * 0.5
    print("  --- Test dispatch at 50% of x_bar_profile ---")
    print(f"  x shape                    : {tuple(x_test.shape)}")
    print(f"  line_limit_violation [B]   : "
          f"{vpp.line_limit_violation(x_test).numpy().round(6)}")
    print(f"  voltage_violation    [B]   : "
          f"{vpp.voltage_violation(x_test).numpy().round(6)}")
    print(f"  capacity_violation   [B]   : "
          f"{vpp.capacity_violation(x_test).numpy().round(6)}")
    print(f"  is_feasible          [B]   : "
          f"{vpp.is_feasible(x_test).numpy()}")
