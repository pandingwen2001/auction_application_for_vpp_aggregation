#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpp_network.py
--------------
Distribution network topology and feasibility checks for the VPP
distribution-level market (Scene 3, no agent layer).

Scenario:
  - VPP operates a single-period distribution-level electricity market
  - N heterogeneous DERs (MT, DG, DR) submit bids directly to the VPP
  - VPP clears the market via QP, respecting physical network constraints
  - Each DER may strategically inflate its bid (bounded rationality)
  - No temporal coupling: single-period only (no ESS SOC, no ramp, no DR cumulative)

Network: condensed 8-bus radial feeder (inspired by IEEE 33-bus)
  Bus 0 (substation/slack)
   └─ line 0 ─ Bus 1
                ├─ line 1 ─ Bus 2
                │            ├─ line 3 ─ Bus 4  [MT_0, DG_0]
                │            └─ line 4 ─ Bus 5  [DR_0]
                └─ line 2 ─ Bus 3
                             ├─ line 5 ─ Bus 6  [MT_1, DG_1]
                             └─ line 6 ─ Bus 7  [DR_1]

Physical constraints modelled:
  1. Active power balance  (equality, with P_VPP as slack)
  2. Line thermal limits   (|S_l| ≤ S_max, approximated via active power flow)
  3. Bus voltage limits    (V_min ≤ V_j ≤ V_max, linearised DistFlow)
  4. DER capacity bounds   (0 ≤ x_i ≤ x_bar_i)

Reactive power:
  - Each DER operates at a fixed power factor (grid-code mandated)
  - Q_i = P_i * tan(arccos(pf_i))
  - Reactive contribution is folded into the voltage sensitivity matrix A_volt
  - Reactive power balance is NOT enforced as a separate constraint
    (the substation absorbs reactive mismatch, consistent with standard
     linearised DistFlow assumptions used in Liu et al. 2024 and
     Heydarian-Forushani et al. 2022)
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _radial_path_matrix(n_buses, n_lines, parent_bus, parent_line):
    """
    Binary matrix [n_buses, n_lines].
    path[k, l] = 1 if line l lies on the path from root (bus 0) to bus k.
    """
    path = np.zeros((n_buses, n_lines), dtype=np.float64)
    for k in range(n_buses):
        b = k
        while parent_bus[b] >= 0:
            path[k, parent_line[b]] = 1.0
            b = parent_bus[b]
    return path


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------

def build_network():
    """
    Build the 8-bus radial distribution network dict.

    DER layout (6 DERs total, heterogeneous types):
      DER 0 : MT  at bus 4    quadratic cost, moderate
      DER 1 : DG  at bus 4    near-zero marginal cost (PV-like)
      DER 2 : DR  at bus 5    quadratic cost, higher
      DER 3 : MT  at bus 6    quadratic cost, moderate
      DER 4 : DG  at bus 6    near-zero marginal cost
      DER 5 : DR  at bus 7    quadratic cost, higher

    Returns
    -------
    net : dict  (see inline comments for each key)
    """

    n_buses = 8
    n_lines = 7
    n_ders  = 6

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    # parent_bus[k]  = parent bus of bus k  (-1 for root bus 0)
    # parent_line[k] = line index connecting bus k to its parent
    parent_bus  = [-1,  0,  1,  1,  2,  2,  3,  3]
    parent_line = [-1,  0,  1,  2,  3,  4,  5,  6]

    # DER-to-bus assignment
    der_bus  = [4, 4, 5, 6, 6, 7]
    der_type = ['MT', 'DG', 'DR', 'MT', 'DG', 'DR']

    # ------------------------------------------------------------------
    # DER parameters
    # ------------------------------------------------------------------
    # Maximum output (MW) — public knowledge
    x_bar = np.array([0.50, 0.30, 0.25,
                       0.50, 0.30, 0.25])

    # Fixed power factor per DER (grid-code mandated, public)
    #   MT : pf = 0.90  (inductive load, produces reactive)
    #   DG : pf = 1.00  (unity, inverter-based)
    #   DR : pf = 0.95
    der_pf = np.array([0.90, 1.00, 0.95,
                        0.90, 1.00, 0.95])
    tan_pf = np.tan(np.arccos(np.clip(der_pf, 1e-9, 1.0)))

    # ------------------------------------------------------------------
    # Private cost bounds  (used by DERTypePrior for type sampling)
    # C_i(x_i) = a_i * x_i^2 + b_i * x_i
    # ------------------------------------------------------------------
    #   MT : moderate quadratic + linear cost  (gas microturbine)
    #   DG : near-zero (PV marginal cost ≈ 0)
    #   DR : moderate (user discomfort cost)
    #
    # Design principle: most DERs should be cheaper than pi_DA (=0.25)
    # at low-to-mid output, but may exceed pi_DA at full capacity.
    # This creates economic incentive to dispatch DERs AND strategic space.
    #
    #   MC(x) = 2*a*x + b
    #   MT: MC(0)=[0.02,0.08], MC(xbar)=[0.12,0.38] → competitive at low x
    #   DG: MC(0)=[0.00,0.01], MC(xbar)=[0.006,0.04] → always cheap
    #   DR: MC(0)=[0.03,0.10], MC(xbar)=[0.13,0.40] → competitive at low x
    mc_a_lo = np.array([0.10, 0.01, 0.10,   0.10, 0.01, 0.10])
    mc_a_hi = np.array([0.30, 0.05, 0.30,   0.30, 0.05, 0.30])
    mc_b_lo = np.array([0.02, 0.00, 0.03,   0.02, 0.00, 0.03])
    mc_b_hi = np.array([0.08, 0.01, 0.10,   0.08, 0.01, 0.10])

    # ------------------------------------------------------------------
    # Network parameters
    # ------------------------------------------------------------------
    # Line resistance R and reactance X (pu, 1 MVA base, 11 kV)
    # End-of-feeder lines (3-6) are longer → higher R, X
    line_R = np.array([0.010, 0.015, 0.015,
                        0.020, 0.020, 0.020, 0.020])
    line_X = np.array([0.030, 0.040, 0.040,
                        0.055, 0.055, 0.055, 0.055])

    # Thermal ratings (MVA ≈ MW at near-unity pf)
    line_ratings = np.array([1.50, 1.00, 1.00,
                               0.60, 0.60, 0.60, 0.60])

    # Fixed load per bus (MW)
    loads = np.array([0.00, 0.10, 0.15, 0.15,
                       0.20, 0.20, 0.20, 0.20])
    load_total = float(loads.sum())   # 1.20 MW

    # Baseline voltage (pu) — pre-dispatch operating point
    v_base = np.array([1.000, 0.995, 0.990, 0.990,
                        0.983, 0.985, 0.983, 0.985])
    v_min  = 0.95    # lower voltage limit (pu)
    v_max  = 1.05    # upper voltage limit (pu)

    # Day-ahead market clearing price ($/kWh = 250 $/MWh)
    # Set higher than most DER marginal costs so dispatching DERs is
    # economically rational for the VPP — the core incentive for
    # strategic bidding by DERs.
    pi_DA = 0.25

    # ------------------------------------------------------------------
    # Derived matrices
    # ------------------------------------------------------------------

    # Path matrix [n_buses, n_lines]: path[k,l]=1 if line l on path 0→k
    path_matrix = _radial_path_matrix(
        n_buses, n_lines, parent_bus, parent_line)

    # PTDF-like matrix H [n_lines, n_buses]: H[l,k]=1 if bus k downstream of line l
    H = path_matrix.T.copy()

    # DER-to-bus incidence B_der [n_buses, N]
    B_der = np.zeros((n_buses, n_ders), dtype=np.float64)
    for i, bus in enumerate(der_bus):
        B_der[bus, i] = 1.0

    # Active power flow sensitivity per DER: A_flow [n_lines, N]
    # delta_f_l = sum_i A_flow[l,i] * x_i
    A_flow = H @ B_der

    # Baseline net injection per bus (no DER dispatch at baseline)
    P_inj_base = -loads.copy()

    # Baseline line flows (MW)
    baseline_flow = H @ P_inj_base

    # Thermal flow margins for incremental DER dispatch
    # total_flow_l = baseline_flow_l + A_flow[l,:] @ x
    # constraint: -line_ratings <= total_flow <= line_ratings
    flow_margin_up = line_ratings - baseline_flow   # [n_lines]
    flow_margin_dn = line_ratings + baseline_flow   # [n_lines]

    # ------------------------------------------------------------------
    # Voltage sensitivity (linearised DistFlow)
    # ------------------------------------------------------------------
    # S_V_P[k,j] = dV_k / dP_j  (voltage rise at bus k per MW at bus j)
    # S_V_Q[k,j] = dV_k / dQ_j  (voltage rise at bus k per MVar at bus j)
    #
    # Derivation: for a radial feeder,
    #   dV_k = sum_{l on path 0→k} (R_l * dP_l + X_l * dQ_l) / V_rated
    # where dP_l = sum_{j downstream of l} dP_j  (captured by path_matrix)
    #
    # S_V_P = (path_matrix * R) @ H    [n_buses, n_buses]
    # S_V_Q = (path_matrix * X) @ H    [n_buses, n_buses]
    S_V_P = (path_matrix * line_R) @ H   # [n_buses, n_buses]
    S_V_Q = (path_matrix * line_X) @ H   # [n_buses, n_buses]

    # Combined voltage sensitivity per DER [n_buses, N]
    # dV_k / dx_i = S_V_P[k, bus_i] + S_V_Q[k, bus_i] * tan_pf_i
    # (reactive injection Q_i = x_i * tan_pf_i folded in automatically)
    A_volt_P = S_V_P @ B_der
    A_volt_Q = S_V_Q @ B_der
    A_volt   = A_volt_P + A_volt_Q * tan_pf[np.newaxis, :]   # [n_buses, N]

    # Voltage margins
    volt_margin_up = v_max - v_base   # [n_buses]
    volt_margin_dn = v_base - v_min   # [n_buses]

    return dict(
        # Dimensions
        n_buses      = n_buses,
        n_lines      = n_lines,
        n_ders       = n_ders,
        # DER metadata (public)
        der_bus      = der_bus,
        der_type     = der_type,
        x_bar        = x_bar,
        der_pf       = der_pf,
        tan_pf       = tan_pf,
        # Private cost bounds (for type sampling)
        mc_a_lo      = mc_a_lo,
        mc_a_hi      = mc_a_hi,
        mc_b_lo      = mc_b_lo,
        mc_b_hi      = mc_b_hi,
        # Network data
        loads        = loads,
        load_total   = load_total,
        line_R       = line_R,
        line_X       = line_X,
        line_ratings = line_ratings,
        v_base       = v_base,
        v_min        = v_min,
        v_max        = v_max,
        pi_DA        = pi_DA,
        # Derived matrices
        H                = H,
        B_der            = B_der,
        path_matrix      = path_matrix,
        A_flow           = A_flow,
        baseline_flow    = baseline_flow,
        flow_margin_up   = flow_margin_up,
        flow_margin_dn   = flow_margin_dn,
        S_V_P            = S_V_P,
        S_V_Q            = S_V_Q,
        A_volt           = A_volt,
        volt_margin_up   = volt_margin_up,
        volt_margin_dn   = volt_margin_dn,
    )


# ---------------------------------------------------------------------------
# IEEE 33-bus network builder
# ---------------------------------------------------------------------------

def build_33bus_network():
    """
    Build the IEEE 33-bus radial distribution network dict.

    Based on Liu et al. 2024 parameters.
    33 buses, 32 lines, 16 DERs (8 DG + 4 MT + 4 DR, no ESS).
    Single period, no temporal coupling.

    DER layout (16 DERs total):
      DER  0 : PV  at bus  3   linear cost
      DER  1 : PV  at bus  6   linear cost
      DER  2 : PV  at bus 10   linear cost
      DER  3 : PV  at bus 12   linear cost
      DER  4 : WT  at bus 16   linear cost
      DER  5 : WT  at bus 20   linear cost
      DER  6 : WT  at bus 22   linear cost
      DER  7 : WT  at bus 31   linear cost
      DER  8 : MT  at bus  9   quadratic cost
      DER  9 : MT  at bus 18   quadratic cost
      DER 10 : MT  at bus 23   quadratic cost
      DER 11 : MT  at bus 29   quadratic cost
      DER 12 : DR  at bus  1   quadratic cost
      DER 13 : DR  at bus 14   quadratic cost
      DER 14 : DR  at bus 26   quadratic cost
      DER 15 : DR  at bus 30   quadratic cost

    Returns
    -------
    net : dict  (same keys as build_network(), plus 'der_labels')
    """

    n_buses = 33
    n_lines = 32
    n_ders  = 16

    # ------------------------------------------------------------------
    # Topology (0-indexed, radial tree)
    # ------------------------------------------------------------------
    parent_bus  = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                   9, 10, 11, 12, 13, 14, 15, 16,
                   1, 18, 19, 20,
                   2, 22, 23,
                   5, 25, 26, 27, 28, 29, 30, 31]
    parent_line = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                   9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20,
                   21, 22, 23,
                   24, 25, 26, 27, 28, 29, 30, 31]

    # DER-to-bus assignment (0-indexed bus)
    # Order: 8 DG (PV/WT), 4 MT, 4 DR
    der_bus  = [3, 6, 10, 12, 16, 20, 22, 31,   # DG (PV/WT)
                9, 18, 23, 29,                    # MT
                1, 14, 26, 30]                    # DR
    der_type = ['DG', 'DG', 'DG', 'DG', 'DG', 'DG', 'DG', 'DG',
                'MT', 'MT', 'MT', 'MT',
                'DR', 'DR', 'DR', 'DR']

    # Human-readable labels (1-indexed bus numbers for readability)
    der_labels = ['PV_4', 'PV_7', 'PV_11', 'PV_13',
                  'WT_17', 'WT_21', 'WT_23', 'WT_32',
                  'MT_10', 'MT_19', 'MT_24', 'MT_30',
                  'DR_2', 'DR_15', 'DR_27', 'DR_31']

    # ------------------------------------------------------------------
    # DER parameters
    # ------------------------------------------------------------------
    # Maximum output (MW) — public knowledge
    # DG: PV_0=0.300, PV_1=0.150, PV_2=0.150, PV_3=0.100,
    #     WT_0=0.800, WT_1=0.800, WT_2=0.400, WT_3=0.600
    # MT: 0.180, 0.200, 0.200, 0.160
    # DR: 0.280, 0.260, 0.200, 0.300
    x_bar = np.array([0.300, 0.150, 0.150, 0.100, 0.800, 0.800, 0.400, 0.600,
                      0.180, 0.200, 0.200, 0.160,
                      0.280, 0.260, 0.200, 0.300])

    # Fixed power factor per DER (grid-code mandated, public)
    der_pf = np.array([0.90, 0.85, 0.85, 0.90, 0.95, 0.85, 0.90, 0.95,   # DG
                       0.95, 0.95, 0.95, 0.95,                             # MT
                       0.90, 0.85, 0.90, 0.90])                            # DR
    tan_pf = np.tan(np.arccos(np.clip(der_pf, 1e-9, 1.0)))

    # ------------------------------------------------------------------
    # Nominal cost coefficients (converted to MW units)
    # C_i(x) = a_i * x^2 + b_i * x,  x in MW
    #
    # DG (linear): a=0, b = paper_cost_coeff * 1000
    # MT (quadratic): a = paper_a * 1e6, b = paper_b * 1e3
    # DR (quadratic): a = paper_a * 1e6, b = paper_b * 1e3
    # ------------------------------------------------------------------
    # Nominal b for DG ($/MWh):
    #   PV_0: 0.0026*1000=2.6, PV_1: 0.0030*1000=3.0,
    #   PV_2: 0.0024*1000=2.4, PV_3: 0.0028*1000=2.8,
    #   WT_0: 0.0032*1000=3.2, WT_1: 0.0038*1000=3.8,
    #   WT_2: 0.0040*1000=4.0, WT_3: 0.0036*1000=3.6
    dg_b_nom = np.array([2.6, 3.0, 2.4, 2.8, 3.2, 3.8, 4.0, 3.6])

    # Nominal a, b for MT ($/MW^2h, $/MWh):
    #   MT_0: a=7e-5*1e6=70, b=0.034*1e3=34
    #   MT_1: a=6.8e-5*1e6=68, b=0.036*1e3=36
    #   MT_2: a=7.4e-5*1e6=74, b=0.035*1e3=35
    #   MT_3: a=7e-5*1e6=70, b=0.035*1e3=35
    mt_a_nom = np.array([70.0, 68.0, 74.0, 70.0])
    mt_b_nom = np.array([34.0, 36.0, 35.0, 35.0])

    # Nominal a, b for DR:
    #   DR_0: a=9e-5*1e6=90, b=0.039*1e3=39
    #   DR_1: a=9e-5*1e6=90, b=0.040*1e3=40
    #   DR_2: a=8.5e-5*1e6=85, b=0.037*1e3=37
    #   DR_3: a=9e-5*1e6=90, b=0.042*1e3=42
    dr_a_nom = np.array([90.0, 90.0, 85.0, 90.0])
    dr_b_nom = np.array([39.0, 40.0, 37.0, 42.0])

    # ------------------------------------------------------------------
    # Private cost bounds (for type sampling, +/- 30%)
    # DG: a_lo=0, a_hi=1.0 (small quadratic), b = nom +/- 30%
    # MT/DR: a,b = nom +/- 30%
    # ------------------------------------------------------------------
    mc_a_lo = np.concatenate([
        np.zeros(8),                    # DG: a_lo = 0
        0.7 * mt_a_nom,                 # MT
        0.7 * dr_a_nom,                 # DR
    ])
    mc_a_hi = np.concatenate([
        np.ones(8) * 1.0,               # DG: a_hi = 1.0
        1.3 * mt_a_nom,                 # MT
        1.3 * dr_a_nom,                 # DR
    ])
    mc_b_lo = np.concatenate([
        0.7 * dg_b_nom,                 # DG
        0.7 * mt_b_nom,                 # MT
        0.7 * dr_b_nom,                 # DR
    ])
    mc_b_hi = np.concatenate([
        1.3 * dg_b_nom,                 # DG
        1.3 * mt_b_nom,                 # MT
        1.3 * dr_b_nom,                 # DR
    ])

    # ------------------------------------------------------------------
    # Network parameters
    # ------------------------------------------------------------------
    V_base_kV = 12.66
    V_base_sq = V_base_kV ** 2   # 160.2756

    # Line impedance in ohm (32 lines)
    # Order: 0→1, 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→10,
    #        10→11, 11→12, 12→13, 13→14, 14→15, 15→16, 16→17,
    #        1→18, 18→19, 19→20, 20→21,
    #        2→22, 22→23, 23→24,
    #        5→25, 25→26, 26→27, 27→28, 28→29, 29→30, 30→31, 31→32
    R_ohm = np.array([
        0.0922, 0.4930, 0.3660, 0.3811, 0.8190,   # lines 0-4
        0.1872, 1.7114, 1.0300, 1.0440, 0.1966,   # lines 5-9
        0.3744, 1.4680, 0.5416, 0.5910, 0.7463,   # lines 10-14
        1.2890, 0.7320,                             # lines 15-16
        0.1640, 1.5042, 0.4095, 0.7089,            # lines 17-20
        0.4512, 0.8980, 0.8960,                     # lines 21-23
        0.2030, 0.2842, 1.0590, 0.8042,            # lines 24-27
        0.5075, 0.9744, 0.3105, 0.3410,            # lines 28-31
    ])
    X_ohm = np.array([
        0.0477, 0.2511, 0.1864, 0.1941, 0.7070,   # lines 0-4
        0.6188, 1.2351, 0.7400, 0.7400, 0.0650,   # lines 5-9
        0.1238, 1.1550, 0.7129, 0.5260, 0.5450,   # lines 10-14
        1.7210, 0.5740,                             # lines 15-16
        0.1565, 1.3554, 0.4784, 0.9373,            # lines 17-20
        0.3083, 0.7091, 0.7011,                     # lines 21-23
        0.1034, 0.1447, 0.9337, 0.7006,            # lines 24-27
        0.2585, 0.9630, 0.3619, 0.5302,            # lines 28-31
    ])

    # Convert to effective pu-like units: R_eff = R_ohm / V_base^2
    line_R = R_ohm / V_base_sq
    line_X = X_ohm / V_base_sq

    # Thermal ratings (MVA ≈ MW)
    # Main feeder (lines 0-4): 5 MVA, others: 3 MVA
    line_ratings = np.array([
        5.0, 5.0, 5.0, 5.0, 5.0,                   # lines 0-4 (main feeder)
        3.0, 3.0, 3.0, 3.0, 3.0,                   # lines 5-9
        3.0, 3.0, 3.0, 3.0, 3.0,                   # lines 10-14
        3.0, 3.0,                                   # lines 15-16
        3.0, 3.0, 3.0, 3.0,                         # lines 17-20
        3.0, 3.0, 3.0,                              # lines 21-23
        3.0, 3.0, 3.0, 3.0,                         # lines 24-27
        3.0, 3.0, 3.0, 3.0,                         # lines 28-31
    ])

    # Fixed active load per bus (MW) — converted from kW
    loads = np.array([
        0.000, 0.100, 0.090, 0.120, 0.060, 0.060,   # buses 0-5
        0.200, 0.200, 0.060, 0.060, 0.045,           # buses 6-10
        0.060, 0.060, 0.120, 0.060, 0.060,           # buses 11-15
        0.060, 0.090, 0.090, 0.090, 0.090,           # buses 16-20
        0.090, 0.090, 0.420, 0.420, 0.060,           # buses 21-25
        0.060, 0.060, 0.120, 0.200, 0.150,           # buses 26-30
        0.210, 0.060,                                 # buses 31-32
    ])
    load_total = float(loads.sum())

    # Fixed reactive load per bus (MVar) — from IEEE 33-bus standard data
    loads_q = np.array([
        0.000, 0.060, 0.040, 0.080, 0.030, 0.020,   # buses 0-5
        0.100, 0.100, 0.020, 0.020, 0.030,           # buses 6-10
        0.035, 0.035, 0.080, 0.010, 0.020,           # buses 11-15
        0.020, 0.040, 0.040, 0.040, 0.040,           # buses 16-20
        0.040, 0.050, 0.200, 0.200, 0.025,           # buses 21-25
        0.025, 0.020, 0.070, 0.600, 0.070,           # buses 26-30
        0.100, 0.040,                                 # buses 31-32
    ])

    v_min  = 0.95
    v_max  = 1.05

    # Day-ahead market clearing price ($/MWh)
    # Representative peak hour price from Liu et al. 2024
    pi_DA = 47.9

    # ------------------------------------------------------------------
    # Derived matrices
    # ------------------------------------------------------------------

    # Path matrix [n_buses, n_lines]
    path_matrix = _radial_path_matrix(
        n_buses, n_lines, parent_bus, parent_line)

    # PTDF-like matrix H [n_lines, n_buses]
    H = path_matrix.T.copy()

    # DER-to-bus incidence B_der [n_buses, N]
    B_der = np.zeros((n_buses, n_ders), dtype=np.float64)
    for i, bus in enumerate(der_bus):
        B_der[bus, i] = 1.0

    # Active power flow sensitivity per DER: A_flow [n_lines, N]
    A_flow = H @ B_der

    # Baseline net injection per bus (no DER dispatch at baseline)
    P_inj_base = -loads.copy()

    # Baseline line flows (MW)
    baseline_flow = H @ P_inj_base

    # Thermal flow margins
    flow_margin_up = line_ratings - baseline_flow
    flow_margin_dn = line_ratings + baseline_flow

    # ------------------------------------------------------------------
    # Voltage sensitivity (linearised DistFlow)
    # ------------------------------------------------------------------
    S_V_P = (path_matrix * line_R) @ H   # [n_buses, n_buses]
    S_V_Q = (path_matrix * line_X) @ H   # [n_buses, n_buses]

    # ------------------------------------------------------------------
    # Baseline voltage: compute voltage drop from loads (no DER)
    # ------------------------------------------------------------------
    # V_k = V_sub - sum_{l on path} (R_l * P_l + X_l * Q_l)
    # where P_l, Q_l are line flows due to loads only (negative injection)
    # Using DistFlow: dV_k = S_V_P[k,:] @ P_inj + S_V_Q[k,:] @ Q_inj
    Q_inj_base = -loads_q.copy()
    v_base = np.ones(n_buses, dtype=np.float64)
    v_base += S_V_P @ P_inj_base + S_V_Q @ Q_inj_base  # voltage DROP from loads
    # Clamp: substation = 1.0, others ≥ v_min (OPF needs non-negative margins)
    v_base[0] = 1.0
    v_base = np.clip(v_base, v_min, 1.02)

    # Combined voltage sensitivity per DER [n_buses, N]
    A_volt_P = S_V_P @ B_der
    A_volt_Q = S_V_Q @ B_der
    A_volt   = A_volt_P + A_volt_Q * tan_pf[np.newaxis, :]

    # Voltage margins (relative to baseline operating point)
    volt_margin_up = v_max - v_base   # more room since v_base < 1.0
    volt_margin_dn = v_base - v_min

    return dict(
        # Dimensions
        n_buses      = n_buses,
        n_lines      = n_lines,
        n_ders       = n_ders,
        # DER metadata (public)
        der_bus      = der_bus,
        der_type     = der_type,
        der_labels   = der_labels,
        x_bar        = x_bar,
        der_pf       = der_pf,
        tan_pf       = tan_pf,
        # Private cost bounds (for type sampling)
        mc_a_lo      = mc_a_lo,
        mc_a_hi      = mc_a_hi,
        mc_b_lo      = mc_b_lo,
        mc_b_hi      = mc_b_hi,
        # Network data
        loads        = loads,
        load_total   = load_total,
        line_R       = line_R,
        line_X       = line_X,
        line_ratings = line_ratings,
        v_base       = v_base,
        v_min        = v_min,
        v_max        = v_max,
        pi_DA        = pi_DA,
        # Derived matrices
        H                = H,
        B_der            = B_der,
        path_matrix      = path_matrix,
        A_flow           = A_flow,
        baseline_flow    = baseline_flow,
        flow_margin_up   = flow_margin_up,
        flow_margin_dn   = flow_margin_dn,
        S_V_P            = S_V_P,
        S_V_Q            = S_V_Q,
        A_volt           = A_volt,
        volt_margin_up   = volt_margin_up,
        volt_margin_dn   = volt_margin_dn,
    )


# ---------------------------------------------------------------------------
# Feasibility checker
# ---------------------------------------------------------------------------

class VPPNetwork:
    """
    Wraps the network dict and exposes feasibility checks.
    All tensor methods accept batched input [B, N].
    """

    def __init__(self, net: dict):
        self.net        = net
        self.N          = net["n_ders"]
        self.n_buses    = net["n_buses"]
        self.n_lines    = net["n_lines"]
        self.load_total = float(net["load_total"])

        # Register as float32 tensors
        self.x_bar         = torch.tensor(net["x_bar"],          dtype=torch.float32)
        self.line_ratings  = torch.tensor(net["line_ratings"],    dtype=torch.float32)
        self.A_flow        = torch.tensor(net["A_flow"],          dtype=torch.float32)
        self.baseline_flow = torch.tensor(net["baseline_flow"],   dtype=torch.float32)
        self.A_volt        = torch.tensor(net["A_volt"],          dtype=torch.float32)
        self.v_base        = torch.tensor(net["v_base"],          dtype=torch.float32)
        self.v_max         = float(net["v_max"])
        self.v_min         = float(net["v_min"])

    # ------------------------------------------------------------------
    # Line flows
    # ------------------------------------------------------------------

    def line_flows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Total active power flows: baseline + DER injection contribution.
        x : [B, N] → [B, n_lines]
        """
        return self.baseline_flow.unsqueeze(0) + x @ self.A_flow.T

    def line_limit_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Max thermal violation across lines: max_l (|f_l| - rating_l)+
        x : [B, N] → [B]
        """
        f    = self.line_flows(x)
        viol = torch.relu(f.abs() - self.line_ratings.unsqueeze(0))
        return viol.max(dim=-1).values

    # ------------------------------------------------------------------
    # Voltages
    # ------------------------------------------------------------------

    def voltage_at_buses(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bus voltages: v_base + A_volt @ x  (linearised DistFlow).
        A_volt already includes both P and Q contributions (fixed pf).
        x : [B, N] → [B, n_buses]
        """
        return self.v_base.unsqueeze(0) + x @ self.A_volt.T

    def voltage_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Max voltage violation: max(overvoltage, undervoltage)+
        x : [B, N] → [B]
        """
        v       = self.voltage_at_buses(x)
        viol_hi = torch.relu(v - self.v_max)
        viol_lo = torch.relu(self.v_min - v)
        return torch.max(viol_hi.max(dim=-1).values,
                         viol_lo.max(dim=-1).values)

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def capacity_violation(self, x: torch.Tensor) -> torch.Tensor:
        """x_i > x_bar_i or x_i < 0. Returns max violation [B]."""
        viol_hi = torch.relu(x  - self.x_bar.unsqueeze(0))
        viol_lo = torch.relu(-x)
        return torch.max(viol_hi.max(dim=-1).values,
                         viol_lo.max(dim=-1).values)

    # ------------------------------------------------------------------
    # DLMP  (analytical, post-hoc)
    # ------------------------------------------------------------------

    def compute_dlmp(self, x: torch.Tensor,
                     lambda_p: torch.Tensor) -> torch.Tensor:
        """
        Compute per-DER DLMP (active power component only):
            DLMP_i = lambda_p * (1 + LF_i)
        where LF_i is the linearised loss factor at DER i's bus.

        This is the settlement price used as baseline (Liu et al. 2024).

        lambda_p : [B]    system marginal price (dual of power balance)
        x        : [B, N] dispatch (used for loss factor computation)
        Returns  : [B, N]
        """
        # Loss factor: sum over buses of voltage sensitivity / mean v_base
        # LF_i = sum_k (S_V_P[k, bus_i]) / V_rated  (simplified, ignores Q)
        A_volt_np = self.net["A_volt"]                        # [n_buses, N]
        lf = torch.tensor(
            A_volt_np.sum(axis=0) / float(self.net["v_base"].mean()),
            dtype=torch.float32, device=x.device)             # [N]
        return lambda_p.unsqueeze(-1) * (1.0 + lf.unsqueeze(0))

    # ------------------------------------------------------------------
    # Overall feasibility
    # ------------------------------------------------------------------

    def is_feasible(self, x: torch.Tensor,
                    tol: float = 1e-3) -> torch.Tensor:
        """Bool [B]: True iff all constraints satisfied within tol."""
        return (self.line_limit_violation(x) < tol) & \
               (self.voltage_violation(x)    < tol) & \
               (self.capacity_violation(x)   < tol)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarise(self, x: torch.Tensor):
        """Print feasibility summary for a single dispatch x [N]."""
        assert x.ndim == 1
        x_ = x.unsqueeze(0)
        print(f"  Dispatch x (MW)  : {x.numpy().round(4)}")
        print(f"  Sum x            : {x.sum().item():.4f}  "
              f"(load_total={self.load_total:.4f})")
        print(f"  DER types        : {self.net['der_type']}")
        print()

        f  = self.line_flows(x_).squeeze(0).numpy()
        ll = self.line_limit_violation(x_).item()
        print(f"  Line flows (MW)  : {f.round(4)}")
        print(f"  Line ratings     : {self.net['line_ratings']}")
        print(f"  Line viol        : {ll:.6f}")
        print()

        v  = self.voltage_at_buses(x_).squeeze(0).numpy()
        vv = self.voltage_violation(x_).item()
        print(f"  Bus voltages (pu): {v.round(4)}")
        print(f"  Limits           : [{self.v_min:.2f}, {self.v_max:.2f}]")
        print(f"  Voltage viol     : {vv:.6f}")
        print()

        cv = self.capacity_violation(x_).item()
        print(f"  Capacity viol    : {cv:.6f}")
        print(f"  Feasible?        : {self.is_feasible(x_).item()}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = build_network()
    vpp = VPPNetwork(net)

    print("=== 8-bus VPP Distribution Market Network ===")
    print(f"  N DERs       : {net['n_ders']}")
    print(f"  DER types    : {net['der_type']}")
    print(f"  DER buses    : {net['der_bus']}")
    print(f"  x_bar (MW)   : {net['x_bar']}")
    print(f"  load_total   : {net['load_total']:.3f} MW")
    print(f"  pi_DA        : {net['pi_DA']:.4f} $/kWh")
    print()
    print(f"  Baseline flow: {net['baseline_flow'].round(4)}")
    print(f"  flow_margin_up: {net['flow_margin_up'].round(4)}")
    print(f"  volt_margin_up: {net['volt_margin_up'].round(4)}")
    print(f"  A_volt (bus4) : {net['A_volt'][4].round(5)}")
    print(f"  A_volt (bus6) : {net['A_volt'][6].round(5)}")
    print()

    x1 = torch.tensor([0.20, 0.15, 0.10, 0.20, 0.15, 0.10])
    print("--- Test 1: moderate dispatch ---")
    vpp.summarise(x1)
    print()

    x2 = torch.tensor([0.50, 0.30, 0.25, 0.02, 0.02, 0.02])
    print("--- Test 2: Agent A feeder heavily loaded ---")
    vpp.summarise(x2)
    print()

    x3 = torch.tensor([0.60, 0.30, 0.25, 0.20, 0.15, 0.10])
    print("--- Test 3: MT_0 over capacity ---")
    vpp.summarise(x3)
