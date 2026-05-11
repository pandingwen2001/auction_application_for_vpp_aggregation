#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ercot_profiles.py
-----------------
ERCOT 2023-derived 24h scenario profiles. Replaces the single-day Liu profile
with one of 24 representative days (first Wednesday and first Saturday of
each month) drawn from real ERCOT hourly data.

Data source: data/ercot_2023_typical.npz, produced by fetch_ercot_typical_days.py
  - Load: EIA-930 Balancing Authority hourly demand for ERCO
  - DAM SPP at HB_HOUSTON: gridstatus.Ercot.get_dam_spp(2023)
  - Solar / Wind capacity factors: EIA-930 net generation / 2023 annual peak

Calibration to Liu reference:
  1. Load:   each day rescaled so the 24-day mean equals base_load_mw.
             Each day retains its own intraday shape AND cross-day magnitude
             (winter days are smaller, summer days larger).
  2. pi_DA:  globally clipped at `clip_factor * pi_DA_baseline` (default 3.0×
             ≈ $123/MWh), then rescaled so the 24-day mean equals pi_DA_baseline.
             Removes pathological scarcity tails but keeps day-to-day variation.
  3. PV CF:  multiplied by alpha_PV so that the 24-day aggregate
             (PV energy / load energy) ratio matches Liu's single-day ratio.
  4. WT CF:  same calibration as PV.

DR availability, MT ramp, ESS parameters are unchanged from Liu (these are
equipment-level, not weather-driven).

Public API:
  num_scenarios()             -> int (= 24)
  scenario_date(idx)          -> "YYYY-MM-DD"
  load_24h_profiles_ercot(...)-> same dict shape as liu_profiles.load_24h_profiles
"""

import os
import numpy as np

from liu_profiles import (                         # noqa: E402
    LOAD_SCALING_LIU, PI_DA_LIU,
    _typical_pv_factor, _typical_wt_factor,
    MT_RAMP, DR_ENERGY_MAX, DR_TIME_WINDOWS,
    ESS_PARAMS, dr_availability_mask, T_HOURS,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NPZ_PATH = os.path.join(_THIS_DIR, "ercot_2023_typical.npz")

_CACHE = {}


def _load_npz():
    if "data" in _CACHE:
        return _CACHE["data"]
    if not os.path.exists(_NPZ_PATH):
        raise FileNotFoundError(
            f"ERCOT scenarios NPZ not found at {_NPZ_PATH}. "
            f"Run data/fetch_ercot_typical_days.py first.")
    z = np.load(_NPZ_PATH)
    _CACHE["data"] = dict(
        dates=z["dates"],
        load_MW=z["load_MW"].astype(np.float64),
        pi_DA_USDperMWh=z["pi_DA_USDperMWh"].astype(np.float64),
        pv_cf=z["pv_cf"].astype(np.float64),
        wt_cf=z["wt_cf"].astype(np.float64),
    )
    return _CACHE["data"]


def num_scenarios() -> int:
    return int(_load_npz()["load_MW"].shape[0])


def scenario_date(idx: int) -> str:
    return str(_load_npz()["dates"][idx])


def _compute_calibration(pi_DA_baseline: float,
                         clip_factor: float = 3.0) -> dict:
    """
    Compute scenario-independent calibration coefficients.
    Returns:
      load_scale          : MW per (raw_MW / mean_ercot_load_MW)  = base_load_mw
                            (caller multiplies by base_load_mw)
      pi_clip             : USD/MWh, the absolute clip threshold
      pi_post_clip_mean   : USD/MWh, mean after clip (for rescale)
      alpha_PV, alpha_WT  : multipliers applied to pv_cf / wt_cf
    """
    z = _load_npz()
    ercot_load_mean = float(z["load_MW"].mean())          # ~50 GW
    ercot_pv_mean   = float(z["pv_cf"].mean())
    ercot_wt_mean   = float(z["wt_cf"].mean())

    pi_clip = float(clip_factor) * float(pi_DA_baseline)
    pi_clipped = np.minimum(z["pi_DA_USDperMWh"], pi_clip)
    pi_post_clip_mean = float(pi_clipped.mean())

    liu_pv_mean = float(_typical_pv_factor().mean())
    liu_wt_mean = float(_typical_wt_factor().mean())
    liu_load_mean = float(LOAD_SCALING_LIU.mean())

    # Calibrate PV/WT so 24-day aggregate (PV energy / load energy)
    # matches Liu's single-day ratio. See derivation in docstring.
    alpha_PV = liu_pv_mean / (liu_load_mean * ercot_pv_mean)
    alpha_WT = liu_wt_mean / (liu_load_mean * ercot_wt_mean)

    return dict(
        ercot_load_mean=ercot_load_mean,
        pi_clip=pi_clip,
        pi_post_clip_mean=pi_post_clip_mean,
        alpha_PV=alpha_PV,
        alpha_WT=alpha_WT,
    )


def load_24h_profiles_ercot(scenario_idx: int,
                            base_load_mw: float = 3.715,
                            pi_DA_baseline: float = None,
                            pi_clip_factor: float = 3.0) -> dict:
    """
    Return the 24h profile bundle for ERCOT scenario `scenario_idx` in the
    same dict shape that `liu_profiles.load_24h_profiles` returns.

    Parameters
    ----------
    scenario_idx : int in [0, 23]
    base_load_mw : float
        Distribution-network nameplate load. The day's hourly load is
        scaled so that the across-24-scenarios mean equals this value.
    pi_DA_baseline : float or None
        Target across-24-scenarios mean for pi_DA, in $/MWh. If None,
        uses Liu's mean PI_DA_LIU.mean() ≈ $40.98.
    pi_clip_factor : float
        Clip raw ERCOT pi_DA at clip_factor * pi_DA_baseline before rescaling.
        Default 3.0 ⇒ scarcity capped at ~3x normal price. Increase to keep
        more tail; decrease to compress further.

    Returns
    -------
    dict, same keys as liu_profiles.load_24h_profiles, plus:
      scenario_idx, scenario_date — for traceability/wandb tagging.
    """
    z = _load_npz()
    if not (0 <= scenario_idx < z["load_MW"].shape[0]):
        raise IndexError(
            f"scenario_idx={scenario_idx} out of range "
            f"[0, {z['load_MW'].shape[0]})")

    if pi_DA_baseline is None:
        pi_DA_baseline = float(PI_DA_LIU.mean())

    cal = _compute_calibration(pi_DA_baseline=pi_DA_baseline,
                               clip_factor=pi_clip_factor)

    # ---- Load: rescale so 24-day mean == base_load_mw ----------------
    raw_load = z["load_MW"][scenario_idx]                  # [T]
    load_profile = (raw_load / cal["ercot_load_mean"]) * base_load_mw

    # ---- pi_DA: clip globally, then rescale to pi_DA_baseline --------
    raw_pi = z["pi_DA_USDperMWh"][scenario_idx]            # [T]
    pi_clipped = np.minimum(raw_pi, cal["pi_clip"])
    pi_DA_profile = (pi_clipped / cal["pi_post_clip_mean"]) * pi_DA_baseline

    # ---- PV / WT: apply alpha so aggregate PV-to-load ratio matches Liu
    pv_factor = (cal["alpha_PV"] * z["pv_cf"][scenario_idx]).astype(np.float64)
    wt_factor = (cal["alpha_WT"] * z["wt_cf"][scenario_idx]).astype(np.float64)
    # alpha_PV is ~1.06, alpha_WT ~0.89; pv_factor may slightly exceed 1.0 on
    # exceptional days. Clamping is intentionally NOT done so daily ratios
    # vary naturally (sunny days stay sunny).

    return dict(
        T                = T_HOURS,
        load_profile     = load_profile.astype(np.float64),
        load_scaling     = (load_profile / base_load_mw).astype(np.float64),
        pv_factor        = pv_factor,
        wt_factor        = wt_factor,
        pi_DA_profile    = pi_DA_profile.astype(np.float64),
        ess_params       = ESS_PARAMS,
        mt_ramp          = MT_RAMP,
        dr_energy_max    = DR_ENERGY_MAX,
        dr_time_windows  = DR_TIME_WINDOWS,
        dr_avail_mask    = dr_availability_mask(),
        # ERCOT-specific metadata
        scenario_idx     = int(scenario_idx),
        scenario_date    = scenario_date(scenario_idx),
        calibration      = cal,
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loaded {num_scenarios()} ERCOT scenarios.")
    print()
    base_load = 3.715
    pi_base = float(PI_DA_LIU.mean())

    p0 = load_24h_profiles_ercot(0, base_load_mw=base_load,
                                 pi_DA_baseline=pi_base)
    print(f"Scenario 0  date={p0['scenario_date']}")
    print(f"  load_profile  : min={p0['load_profile'].min():.3f}  "
          f"mean={p0['load_profile'].mean():.3f}  "
          f"max={p0['load_profile'].max():.3f}  MW")
    print(f"  pi_DA_profile : min={p0['pi_DA_profile'].min():.2f}  "
          f"mean={p0['pi_DA_profile'].mean():.2f}  "
          f"max={p0['pi_DA_profile'].max():.2f}  $/MWh")
    print(f"  pv_factor     : min={p0['pv_factor'].min():.3f}  "
          f"mean={p0['pv_factor'].mean():.3f}  "
          f"max={p0['pv_factor'].max():.3f}")
    print(f"  wt_factor     : min={p0['wt_factor'].min():.3f}  "
          f"mean={p0['wt_factor'].mean():.3f}  "
          f"max={p0['wt_factor'].max():.3f}")
    print()

    # Aggregate stats across all 24
    loads, pis, pvs, wts = [], [], [], []
    for idx in range(num_scenarios()):
        p = load_24h_profiles_ercot(idx, base_load_mw=base_load,
                                    pi_DA_baseline=pi_base)
        loads.append(p["load_profile"])
        pis.append(p["pi_DA_profile"])
        pvs.append(p["pv_factor"])
        wts.append(p["wt_factor"])
    loads = np.stack(loads); pis = np.stack(pis)
    pvs   = np.stack(pvs);   wts = np.stack(wts)

    print(f"Across 24 scenarios:")
    print(f"  daily-mean load (MW)        : min={loads.mean(axis=1).min():.3f}  "
          f"mean={loads.mean():.3f}  max={loads.mean(axis=1).max():.3f}")
    print(f"  daily-mean pi_DA ($/MWh)    : min={pis.mean(axis=1).min():.2f}  "
          f"mean={pis.mean():.2f}  max={pis.mean(axis=1).max():.2f}")
    print(f"  daily-mean PV cf             : min={pvs.mean(axis=1).min():.3f}  "
          f"mean={pvs.mean():.3f}  max={pvs.mean(axis=1).max():.3f}")
    print(f"  daily-mean WT cf             : min={wts.mean(axis=1).min():.3f}  "
          f"mean={wts.mean():.3f}  max={wts.mean(axis=1).max():.3f}")
    print()
    print(f"  Liu reference: load_mean={base_load*LOAD_SCALING_LIU.mean():.3f}  "
          f"pi_mean={pi_base:.2f}  "
          f"pv_mean={_typical_pv_factor().mean():.3f}  "
          f"wt_mean={_typical_wt_factor().mean():.3f}")
