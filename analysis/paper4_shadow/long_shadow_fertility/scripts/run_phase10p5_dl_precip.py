"""Phase 10 Pillar #5 channel decomposition: Distributed-lag with
precipitation (p_growing) as the shock variable.

Mirrors run_phase10p5_dl_spei.py but uses x='p_growing'.

Models:
  M1: p_growing DL + full BASE_CONTROLS (including vol_p_10y — note that
      unlike SPEI, precipitation is NOT a composite of vol_p_10y, so
      including the volatility control is not double-counting)
  M2: p_growing DL + BASE_CONTROLS without vol_p_10y (robustness check,
      parallel to SPEI treatment)
  M3: Temperature comparator on identical sample (same controls as M1)
  M4: SPEI comparator on identical sample (same controls as M1 minus vol_p_10y
      to match SPEI's preferred specification)

All models restricted to the joint SPEI+T+P+wage sample for comparability.

Output: phase10p5_dl_precip.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

REAL_WAGE_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10p5_dl_precip.json"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

BASE_CONTROLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought",
    "vol_t_10y", "vol_p_10y",
]

BASE_CONTROLS_NO_PVOL = [c for c in BASE_CONTROLS if c != "vol_p_10y"]


def _add_lags(
    df: pd.DataFrame, var: str, lags: int, unit_col: str = "iso3"
) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values([unit_col, "year"]).copy()
    lag_cols = []
    for k in range(lags + 1):
        col = f"{var}_lag{k}"
        df[col] = df.groupby(unit_col)[var].shift(k)
        lag_cols.append(col)
    return df, lag_cols


def _fit_dl(
    df: pd.DataFrame,
    y: str,
    x: str,
    lags: int,
    controls: list[str],
    unit_col: str = "iso3",
) -> dict:
    df_lag, lag_cols = _add_lags(df, x, lags, unit_col)
    keep = [y, unit_col, "year"] + lag_cols + controls
    keep = [c for c in keep if c in df_lag.columns]
    sub = df_lag[keep].dropna()

    unit_dums = pd.get_dummies(sub[unit_col], drop_first=True, dtype=float)
    year_dums = pd.get_dummies(sub["year"].astype(int), drop_first=True, dtype=float)
    year_dums.columns = [f"y_{c}" for c in year_dums.columns]

    ctrl_present = [c for c in controls if c in sub.columns]
    X = sm.add_constant(pd.concat([
        sub[lag_cols].astype(float),
        unit_dums,
        year_dums,
        sub[ctrl_present].astype(float),
    ], axis=1))
    cluster = sub[unit_col].astype("category").cat.codes.to_numpy()
    res = sm.OLS(sub[y].astype(float).to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )

    # DL irf rows
    rows = []
    for k, col in enumerate(lag_cols):
        idx = 1 + k
        b = float(res.params[idx])
        s = float(res.bse[idx])
        rows.append({"lag": k, "beta": b, "se": s,
                     "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    lag_idx = np.arange(1, 1 + len(lag_cols))
    cum_b = float(res.params[lag_idx].sum())
    cov_block = res.cov_params()[np.ix_(lag_idx, lag_idx)]
    cum_se = float(np.sqrt(np.ones(len(lag_idx)) @ cov_block @ np.ones(len(lag_idx))))
    rows.append({"lag": "cumulative", "beta": cum_b, "se": cum_se,
                 "ci_low": cum_b - 1.96 * cum_se, "ci_high": cum_b + 1.96 * cum_se})

    return {
        "irf": rows,
        "cumulative_beta": cum_b,
        "cumulative_se": cum_se,
        "n_obs": int(sub.shape[0]),
        "n_units": int(sub[unit_col].nunique()),
    }


def run() -> None:
    print("Loading panel...")
    df = assemble_panel_multi()
    print(f"  Panel shape: {df.shape}")

    rw = pd.read_parquet(REAL_WAGE_PATH)[["iso3", "year", "log_real_wage"]]
    df = df.merge(rw, on=["iso3", "year"], how="left")

    # Restrict to the joint sample where all three shock variables are available
    joint_cols = ["log_cbr", "p_growing", "spei_growing", "t_growing", "log_real_wage"]
    df_joint = df.dropna(subset=joint_cols)
    print(f"  Joint sample (all shock vars + wage): {len(df_joint)} rows, "
          f"{df_joint['iso3'].nunique()} countries")

    results: dict = {}

    # --- M1: Precipitation DL + full controls ---
    print("\nFitting M1: precipitation DL + full controls...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m1_precip_full"] = _fit_dl(
            df_joint, y="log_cbr", x="p_growing", lags=3,
            controls=BASE_CONTROLS,
        )
    cum1 = results["m1_precip_full"]["cumulative_beta"]
    cum1_se = results["m1_precip_full"]["cumulative_se"]
    print(f"  Cumulative P beta: {cum1:.4f}  (SE={cum1_se:.4f}, t={cum1/cum1_se:.2f})")
    print(f"  N_obs={results['m1_precip_full']['n_obs']}, N_units={results['m1_precip_full']['n_units']}")

    # --- M2: Precipitation DL without vol_p_10y (robustness) ---
    print("\nFitting M2: precipitation DL — no vol_p_10y (robustness)...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m2_precip_no_pvol"] = _fit_dl(
            df_joint, y="log_cbr", x="p_growing", lags=3,
            controls=BASE_CONTROLS_NO_PVOL,
        )
    cum2 = results["m2_precip_no_pvol"]["cumulative_beta"]
    cum2_se = results["m2_precip_no_pvol"]["cumulative_se"]
    print(f"  Cumulative P beta: {cum2:.4f}  (SE={cum2_se:.4f}, t={cum2/cum2_se:.2f})")

    # --- M3: Temperature comparator on same joint sample ---
    print("\nFitting M3: temperature DL comparator (same joint sample)...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m3_temperature_comparator"] = _fit_dl(
            df_joint, y="log_cbr", x="t_growing", lags=3,
            controls=BASE_CONTROLS,
        )
    cum3 = results["m3_temperature_comparator"]["cumulative_beta"]
    cum3_se = results["m3_temperature_comparator"]["cumulative_se"]
    print(f"  Cumulative T beta: {cum3:.4f}  (SE={cum3_se:.4f}, t={cum3/cum3_se:.2f})")

    # --- M4: SPEI comparator (no vol_p_10y, matching SPEI preferred spec) ---
    print("\nFitting M4: SPEI DL comparator (same joint sample, no vol_p_10y)...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m4_spei_comparator"] = _fit_dl(
            df_joint, y="log_cbr", x="spei_growing", lags=3,
            controls=BASE_CONTROLS_NO_PVOL,
        )
    cum4 = results["m4_spei_comparator"]["cumulative_beta"]
    cum4_se = results["m4_spei_comparator"]["cumulative_se"]
    print(f"  Cumulative SPEI beta: {cum4:.4f}  (SE={cum4_se:.4f}, t={cum4/cum4_se:.2f})")

    # --- Serialise ---
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    results["meta"] = {
        "note": (
            "All models restricted to the joint sample where p_growing, spei_growing, "
            "t_growing, and log_real_wage are all non-null (ModE-RA coverage 1421-2008). "
            "M1 uses full BASE_CONTROLS including vol_p_10y. M2 drops vol_p_10y. "
            "M3 = temperature DL on same sample. M4 = SPEI DL without vol_p_10y "
            "(matching the SPEI-preferred specification). "
            "Country + year fixed effects, CGM cluster-robust SEs."
        ),
    }
    OUT.write_text(json.dumps(_clean(results), indent=2))
    print(f"\nWrote {OUT}")

    print("\n=== SUMMARY ===")
    print(f"M1 Precipitation (full controls): cum_beta = {cum1:.4f}  SE={cum1_se:.4f}")
    print(f"M2 Precipitation (no vol_p_10y):  cum_beta = {cum2:.4f}  SE={cum2_se:.4f}")
    print(f"M3 Temperature comparator:         cum_beta = {cum3:.4f}  SE={cum3_se:.4f}")
    print(f"M4 SPEI comparator (no vol_p_10y): cum_beta = {cum4:.4f}  SE={cum4_se:.4f}")

    irf1 = [r for r in results["m1_precip_full"]["irf"] if isinstance(r["lag"], int)]
    print("IRF by lag (M1 precipitation):")
    for r in irf1:
        print(f"  lag {r['lag']}: beta={r['beta']:.4f}  SE={r['se']:.4f}")


if __name__ == "__main__":
    run()
