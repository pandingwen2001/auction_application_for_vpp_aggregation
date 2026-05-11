"""
fetch_ercot_typical_days.py
---------------------------
Pulls 2023 ERCOT data and constructs 24 typical days (first Wednesday and
first Saturday of each month).

Sources:
  * EIA-930 Balancing Authority 6-month CSVs  → hourly load, solar gen, wind gen
        https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_*.csv
  * gridstatus.Ercot.get_dam_spp(year=2023)   → DAM hourly settlement-point price
        filtered to HB_HOUSTON

Outputs:
  data/ercot_2023_typical.npz with keys
      dates           : [24] str array of YYYY-MM-DD
      load_MW         : [24, 24] hourly system load (MW)
      pi_DA_USDperMWh : [24, 24] hourly DAM SPP at HB_HOUSTON ($/MWh)
      pv_cf           : [24, 24] solar capacity factor in [0, 1]
      wt_cf           : [24, 24] wind  capacity factor in [0, 1]

The capacity factor is normalised by the annual peak generation of each
technology, so it is a "fraction of best-observed-hour" interpretation
(close to true CF, robust against installed-capacity drift over the year).
"""
import os
import sys
import urllib.request
import datetime
import numpy as np
import pandas as pd

import gridstatus


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_THIS_DIR, "_eia_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. EIA-930: load + solar + wind for all of 2023
# ---------------------------------------------------------------------------

EIA_URLS = {
    "h1": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
          "EIA930_BALANCE_2023_Jan_Jun.csv",
    "h2": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
          "EIA930_BALANCE_2023_Jul_Dec.csv",
}


def _download_eia_csv(url: str) -> str:
    fname = url.rsplit("/", 1)[-1]
    cache_path = os.path.join(_CACHE_DIR, fname)
    if os.path.exists(cache_path):
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"  [cache] {fname}  ({size_mb:.1f} MB)")
        return cache_path
    print(f"  [download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    with open(cache_path, "wb") as f:
        f.write(data)
    print(f"             saved {len(data) / 1e6:.1f} MB")
    return cache_path


def load_eia930_ercot_2023() -> pd.DataFrame:
    """Return a DataFrame indexed by hourly local time with columns
       ['load_MW', 'solar_MW', 'wind_MW'] for ERCO in 2023."""
    print("Fetching EIA-930 BA balance files for 2023...")
    parts = []
    for half, url in EIA_URLS.items():
        path = _download_eia_csv(url)
        df = pd.read_csv(path, low_memory=False, thousands=",")
        df = df[df["Balancing Authority"] == "ERCO"].copy()
        df["ts"] = pd.to_datetime(df["Local Time at End of Hour"],
                                  format="%m/%d/%Y %I:%M:%S %p")
        # EIA labels hours as "end-of-hour"; shift back to start-of-hour
        df["hour_start"] = df["ts"] - pd.Timedelta(hours=1)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("hour_start").reset_index(drop=True)
    out = pd.DataFrame({
        "hour_start": df["hour_start"],
        "load_MW":  pd.to_numeric(df["Demand (MW) (Adjusted)"],
                                  errors="coerce"),
        "solar_MW": pd.to_numeric(
            df["Net Generation (MW) from Solar (Adjusted)"], errors="coerce"),
        "wind_MW":  pd.to_numeric(
            df["Net Generation (MW) from Wind (Adjusted)"], errors="coerce"),
    }).set_index("hour_start")
    out = out.loc["2023-01-01":"2023-12-31"]
    print(f"  ERCOT 2023 rows: {len(out)}  "
          f"(NaN: load={out['load_MW'].isna().sum()}, "
          f"solar={out['solar_MW'].isna().sum()}, "
          f"wind={out['wind_MW'].isna().sum()})")
    return out


# ---------------------------------------------------------------------------
# 2. gridstatus: DAM SPP at HB_HOUSTON for 2023
# ---------------------------------------------------------------------------

def load_dam_spp_houston_2023() -> pd.Series:
    print("Fetching ERCOT DAM SPP (full 2023, all hubs) via gridstatus...")
    cache_path = os.path.join(_CACHE_DIR, "ercot_dam_spp_hb_houston_2023.parquet")
    if os.path.exists(cache_path):
        print(f"  [cache] {cache_path}")
        s = pd.read_parquet(cache_path)["SPP"]
        s.index = pd.to_datetime(s.index)
        return s

    iso = gridstatus.Ercot()
    spp = iso.get_dam_spp(year=2023, verbose=False)
    spp = spp[spp["Location"] == "HB_HOUSTON"].copy()
    print(f"  HB_HOUSTON rows: {len(spp)}")
    spp["hour_start"] = pd.to_datetime(spp["Interval Start"]).dt.tz_localize(None)
    s = spp.set_index("hour_start")["SPP"].sort_index()
    s.to_frame().to_parquet(cache_path)
    return s


# ---------------------------------------------------------------------------
# 3. Pick 24 typical dates: first Wednesday + first Saturday of each month
# ---------------------------------------------------------------------------

def first_weekday_of_month(year: int, month: int, weekday: int) -> datetime.date:
    """weekday: Monday=0 .. Sunday=6"""
    d = datetime.date(year, month, 1)
    while d.weekday() != weekday:
        d += datetime.timedelta(days=1)
    return d


def typical_dates_2023() -> list:
    dates = []
    for m in range(1, 13):
        dates.append(first_weekday_of_month(2023, m, weekday=2))  # Wednesday
        dates.append(first_weekday_of_month(2023, m, weekday=5))  # Saturday
    return sorted(dates)


# ---------------------------------------------------------------------------
# 4. Slice and assemble [24 days, 24 hours] arrays
# ---------------------------------------------------------------------------

def slice_day(series: pd.Series, date: datetime.date) -> np.ndarray:
    start = pd.Timestamp(date)
    end = start + pd.Timedelta(hours=23)
    arr = series.loc[start:end].values
    if len(arr) != 24:
        raise RuntimeError(
            f"date {date}: expected 24 hours, got {len(arr)}. "
            f"Check DST or missing data.")
    return arr.astype(np.float64)


def build_typical_days_npz(out_path: str = None) -> dict:
    eia = load_eia930_ercot_2023()
    spp = load_dam_spp_houston_2023()
    dates = typical_dates_2023()

    print(f"\nSlicing {len(dates)} typical days:")
    for d in dates:
        print(f"  {d}  ({d.strftime('%a')})")

    load_MW = np.stack([slice_day(eia["load_MW"], d)  for d in dates])
    solar   = np.stack([slice_day(eia["solar_MW"], d) for d in dates])
    wind    = np.stack([slice_day(eia["wind_MW"], d)  for d in dates])
    pi_DA   = np.stack([slice_day(spp, d)             for d in dates])

    # Normalise generation to 0~1 capacity factor by the 2023 annual peak.
    # This is robust against minor capacity additions during the year and
    # avoids the need for separate installed-capacity bookkeeping.
    solar_peak_2023 = float(np.nanmax(eia["solar_MW"].values))
    wind_peak_2023  = float(np.nanmax(eia["wind_MW"].values))
    print(f"\n  solar 2023 peak: {solar_peak_2023:.0f} MW")
    print(f"  wind  2023 peak: {wind_peak_2023:.0f} MW")
    pv_cf = np.clip(solar / solar_peak_2023, 0.0, 1.0)
    wt_cf = np.clip(wind  / wind_peak_2023,  0.0, 1.0)

    if out_path is None:
        out_path = os.path.join(_THIS_DIR, "ercot_2023_typical.npz")
    np.savez_compressed(
        out_path,
        dates=np.array([d.isoformat() for d in dates]),
        load_MW=load_MW,
        pi_DA_USDperMWh=pi_DA,
        pv_cf=pv_cf,
        wt_cf=wt_cf,
        solar_peak_MW_2023=solar_peak_2023,
        wind_peak_MW_2023=wind_peak_2023,
    )
    print(f"\nWrote {out_path}")
    print(f"  load_MW         : shape={load_MW.shape}  "
          f"min={load_MW.min():.0f}  max={load_MW.max():.0f}")
    print(f"  pi_DA_USDperMWh : shape={pi_DA.shape}  "
          f"min={pi_DA.min():.2f}  max={pi_DA.max():.2f}")
    print(f"  pv_cf           : shape={pv_cf.shape}  "
          f"min={pv_cf.min():.3f}  max={pv_cf.max():.3f}  "
          f"mean={pv_cf.mean():.3f}")
    print(f"  wt_cf           : shape={wt_cf.shape}  "
          f"min={wt_cf.min():.3f}  max={wt_cf.max():.3f}  "
          f"mean={wt_cf.mean():.3f}")
    return dict(load_MW=load_MW, pi_DA=pi_DA, pv_cf=pv_cf, wt_cf=wt_cf,
                dates=dates)


if __name__ == "__main__":
    build_typical_days_npz()
