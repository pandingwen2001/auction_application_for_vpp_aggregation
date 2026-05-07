#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liu_profiles.py
---------------
24h day-ahead profiles for the multi-period VPP mechanism.

Source: Liu et al. 2024 "Additional Document.pdf"
  - Table II    : MT parameters (ramp, rated power)
  - Table III   : DR parameters (daily energy cap, adjustable time window)
  - Table IV    : ESS parameters (2 units at nodes 16, 29)
  - Table V     : Day-ahead electricity prices pi_DA (24 hourly values)
  - Fig. 1      : Scaling curve for fixed loads (0.98-1.12 p.u.)

PV / WT hourly availability factors are NOT given in the Additional Document
(only the constant capacity-factor alpha is). We keep them as synthetic
typical shapes (bell curve for PV, nocturnal-biased for WT) and mark them
as synthetic. This is fine for Phase 1a/1b where we only need a realistic
day-shape, not Liu-specific renewable curves.

Convention:
  T = 24 hours, index t=0 is 00:00, t=23 is 23:00
"""

import numpy as np


T_HOURS = 24


# ===========================================================================
# Table V: Day-ahead electricity prices ($/MWh), 24 hourly values
# ===========================================================================
# Exactly the numbers from the PDF Table V (time slots 1..24 -> index 0..23).
PI_DA_LIU = np.array([
    32.4, 32.0, 32.4, 32.7, 32.7, 34.0,   #  1- 6
    40.0, 45.9, 47.8, 47.9, 47.7, 43.0,   #  7-12
    41.2, 40.5, 39.9, 40.5, 42.6, 46.5,   # 13-18
    49.4, 49.7, 45.7, 43.4, 40.4, 35.3,   # 19-24
], dtype=np.float64)


# ===========================================================================
# Fig. 1: Scaling curve for fixed loads (per-unit, ~0.98 .. 1.12)
# ===========================================================================
# Read off the plot in the Additional Document, sampled at each integer hour.
# Shape: small dip around 4am, rapid ramp to plateau at 8-11am (~1.12),
# slow decline through the afternoon, mild bump in the evening, back to
# ~1.02 at t=24.
LOAD_SCALING_LIU = np.array([
    1.000, 0.990, 0.985, 0.982, 0.980, 0.988,   #  0- 5
    1.010, 1.060, 1.105, 1.115, 1.118, 1.115,   #  6-11
    1.105, 1.095, 1.088, 1.085, 1.082, 1.078,   # 12-17
    1.075, 1.070, 1.060, 1.050, 1.035, 1.020,   # 18-23
], dtype=np.float64)


# ===========================================================================
# Synthetic PV / WT per-unit availability (Liu PDF does not give hourly shape)
# ===========================================================================

def _typical_pv_factor() -> np.ndarray:
    """PV per-unit availability: 0 at night, bell curve during daytime."""
    hours = np.arange(T_HOURS)
    f = np.exp(-((hours - 12.0) ** 2) / (2.0 * 3.0**2))
    f[hours < 6] = 0.0
    f[hours > 19] = 0.0
    if f.max() > 0:
        f = f / f.max()
    return f.astype(np.float64)


def _typical_wt_factor() -> np.ndarray:
    """WT per-unit availability: nocturnal-biased with moderate fluctuation."""
    hours = np.arange(T_HOURS)
    base = 0.5 + 0.25 * np.cos((hours - 3.0) / 24.0 * 2 * np.pi)
    pattern = np.array([
        0.90, 0.88, 0.85, 0.92, 0.95, 0.88,
        0.80, 0.72, 0.65, 0.60, 0.55, 0.50,
        0.48, 0.45, 0.50, 0.55, 0.62, 0.70,
        0.78, 0.85, 0.88, 0.90, 0.92, 0.91,
    ])
    return np.clip(base * pattern, 0.0, 1.0).astype(np.float64)


# ===========================================================================
# Table II: Microturbine ramp limits (kW/h -> MW/h)
# ===========================================================================
# Order matches vpp_network_multi.py's MT ordering: nodes [10, 19, 24, 30]
MT_RAMP = np.array([60.0, 100.0, 50.0, 50.0], dtype=np.float64) / 1000.0  # MW/h


# ===========================================================================
# Table III: Demand Response cumulative energy caps (kWh -> MWh)
# ===========================================================================
# Order matches vpp_network_multi.py's DR ordering: nodes [2, 15, 27, 31]
DR_ENERGY_MAX = np.array([280.0, 260.0, 200.0, 300.0], dtype=np.float64) / 1000.0  # MWh

# Adjustable time windows (start_hour_inclusive, end_hour_exclusive), same order.
# Liu Table III: DR at node 2 → 08-17, node 15 → 08-22, node 27 → 16-24,
# node 31 → 00-24 (always on).
DR_TIME_WINDOWS = [
    (8, 17),
    (8, 22),
    (16, 24),
    (0, 24),
]


def dr_availability_mask() -> np.ndarray:
    """
    Return a [T, N_dr] 0/1 mask of which hours each DR is allowed to dispatch.
    Multi-period correction layer should zero out x_{i,t} outside its window.
    """
    mask = np.zeros((T_HOURS, len(DR_TIME_WINDOWS)), dtype=np.float64)
    for j, (s, e) in enumerate(DR_TIME_WINDOWS):
        mask[s:e, j] = 1.0
    return mask


# ===========================================================================
# Table IV: Energy Storage System parameters (2 units)
# ===========================================================================
# Liu Table IV: nodes 16 and 29 (1-indexed). Code uses 0-indexed → buses 15, 28.
# P_max = 100 kW, Q_ES = 400 kWh, E_max=1.0 p.u., E_min=0.2 p.u.,
# eta = 0.95, cost $0.050 / $0.054 per kWh.
ESS_PARAMS = dict(
    n_ess          = 2,
    ess_buses      = np.array([15, 28], dtype=np.int64),
    ess_power_max  = np.array([0.100, 0.100], dtype=np.float64),   # MW
    ess_capacity   = np.array([0.400, 0.400], dtype=np.float64),   # MWh (Q_ES)
    ess_soc_max_pu = np.array([1.00, 1.00], dtype=np.float64),     # p.u. of Q_ES
    ess_soc_min_pu = np.array([0.20, 0.20], dtype=np.float64),     # p.u. of Q_ES
    ess_eta_c      = np.array([0.95, 0.95], dtype=np.float64),
    ess_eta_d      = np.array([0.95, 0.95], dtype=np.float64),
    ess_cost       = np.array([50.0, 54.0], dtype=np.float64),     # $/MWh
    ess_soc_init_pu= np.array([0.60, 0.60], dtype=np.float64),     # p.u. start
)


# ===========================================================================
# Public API
# ===========================================================================

def load_24h_profiles(base_load_mw: float = 3.715,
                      pi_DA_baseline: float = None,
                      constant_price: bool = False) -> dict:
    """
    Return the complete 24h profile bundle, aligned to Liu 2024
    Additional Document.pdf.

    Parameters
    ----------
    base_load_mw : float
        Nominal aggregate feeder load at scaling = 1.0. Defaults to the
        single-period 33-bus IEEE baseline (3.715 MW) so the day-integrated
        load is consistent with the single-period setup.
    constant_price : bool
        If True, overrides pi_DA_profile to a flat mean-value curve
        (recommended only for Phase 1a training stability debugging).

    Returns
    -------
    dict with keys:
        T                : int
        load_profile     : [T]       MW      (= base_load_mw * LOAD_SCALING_LIU)
        load_scaling     : [T]       p.u.    (raw Fig. 1 curve)
        pv_factor        : [T]       0..1    (synthetic)
        wt_factor        : [T]       0..1    (synthetic)
        pi_DA_profile    : [T]       $/MWh   (Table V)
        ess_params       : dict              (Table IV, 2 units)
        mt_ramp          : [N_mt]    MW/h    (Table II)
        dr_energy_max    : [N_dr]    MWh     (Table III, S_DR)
        dr_time_windows  : list              (Table III, adjustable time)
        dr_avail_mask    : [T, N_dr] 0/1     (derived from dr_time_windows)
    """
    load_scaling = LOAD_SCALING_LIU.copy()
    load_profile = (load_scaling * base_load_mw).astype(np.float64)
    pv_factor    = _typical_pv_factor()
    wt_factor    = _typical_wt_factor()

    pi_DA_profile = PI_DA_LIU.copy()
    if pi_DA_baseline is not None:
        # Rescale the Liu curve so its mean matches the requested baseline,
        # preserving the Table V hourly shape.
        pi_DA_profile = pi_DA_profile * (pi_DA_baseline / pi_DA_profile.mean())
    if constant_price:
        pi_DA_profile = np.full(T_HOURS, pi_DA_profile.mean(), dtype=np.float64)

    return dict(
        T                = T_HOURS,
        load_profile     = load_profile,
        load_scaling     = load_scaling,
        pv_factor        = pv_factor,
        wt_factor        = wt_factor,
        pi_DA_profile    = pi_DA_profile,
        ess_params       = ESS_PARAMS,
        mt_ramp          = MT_RAMP,
        dr_energy_max    = DR_ENERGY_MAX,
        dr_time_windows  = DR_TIME_WINDOWS,
        dr_avail_mask    = dr_availability_mask(),
    )


# ===========================================================================
# Sanity check
# ===========================================================================
if __name__ == "__main__":
    p = load_24h_profiles(constant_price=False)

    print("=== Liu 2024 24h Profiles (aligned to Additional Document.pdf) ===")
    print(f"  T                 : {p['T']}")
    print(f"  load_profile (MW) : min={p['load_profile'].min():.3f}  "
          f"mean={p['load_profile'].mean():.3f}  "
          f"max={p['load_profile'].max():.3f}")
    print(f"  pv_factor         : min={p['pv_factor'].min():.3f}  "
          f"mean={p['pv_factor'].mean():.3f}  "
          f"max={p['pv_factor'].max():.3f}   [synthetic]")
    print(f"  wt_factor         : min={p['wt_factor'].min():.3f}  "
          f"mean={p['wt_factor'].mean():.3f}  "
          f"max={p['wt_factor'].max():.3f}   [synthetic]")
    print(f"  pi_DA_profile     : min={p['pi_DA_profile'].min():.2f}  "
          f"mean={p['pi_DA_profile'].mean():.2f}  "
          f"max={p['pi_DA_profile'].max():.2f}   [Table V]")
    print()
    print("  Hourly profile:")
    for t in range(T_HOURS):
        print(f"    t={t:2d}  load={p['load_profile'][t]:.3f}  "
              f"pv={p['pv_factor'][t]:.3f}  wt={p['wt_factor'][t]:.3f}  "
              f"pi_DA={p['pi_DA_profile'][t]:5.2f}")
    print()
    print(f"  MT ramp   (MW/h) : {p['mt_ramp']}")
    print(f"  DR E_max  (MWh)  : {p['dr_energy_max']}")
    print(f"  DR windows       : {p['dr_time_windows']}")
    print(f"  DR avail hours/DR: {p['dr_avail_mask'].sum(axis=0)}")
    print()
    print(f"  ESS n_ess        : {p['ess_params']['n_ess']}")
    print(f"  ESS buses        : {p['ess_params']['ess_buses']}")
    print(f"  ESS P_max  (MW)  : {p['ess_params']['ess_power_max']}")
    print(f"  ESS Q_ES (MWh)   : {p['ess_params']['ess_capacity']}")
    print(f"  ESS SoC  (p.u.)  : [{p['ess_params']['ess_soc_min_pu']}, "
          f"{p['ess_params']['ess_soc_max_pu']}]")
    print(f"  ESS cost ($/MWh) : {p['ess_params']['ess_cost']}")
