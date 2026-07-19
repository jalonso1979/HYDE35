"""Phase 10 Pillar #5: Distributed-lag regressions with SPEI as shock.

Mirrors run_phase10_uncertainty_dl.py but uses x='spei_growing' instead of
x='t_growing'.

Control choice rationale
------------------------
SPEI already integrates the precipitation channel (P - PET), so the
precipitation-volatility control vol_p_10y would partially double-count the
water-supply component.  We therefore run two specifications:

  M_spei_full:   SPEI DL + full BASE_CONTROLS (includes vol_p_10y as robustness)
  M_spei_nopvol: SPEI DL + BASE_CONTROLS minus vol_p_10y (cleaner, avoids
                 double-counting the precipitation channel)

We also run the temperature comparator (same controls, x=t_growing) on the
identical SPEI sample for a direct comparison.

Output: phase10p5_dl_spei.json
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
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10p5_dl_spei.json"
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
    # Only keep controls that exist in df
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
    col_names = list(X.columns)

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

    # Restrict to SPEI-available rows so all models are on the same sample
    df_spei = df.dropna(subset=["spei_growing"])
    print(f"  SPEI-available rows: {len(df_spei)}")

    results: dict = {}

    # --- M1: SPEI DL + full controls (including vol_p_10y as robustness) ---
    print("\nFitting M1: SPEI DL + full controls (incl. vol_p_10y)...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m1_spei_full"] = _fit_dl(
            df_spei, y="log_cbr", x="spei_growing", lags=3,
            controls=BASE_CONTROLS,
        )
    cum1 = results["m1_spei_full"]["cumulative_beta"]
    print(f"  Cumulative SPEI beta: {cum1:.4f}  (SE={results['m1_spei_full']['cumulative_se']:.4f})")
    print(f"  N_obs={results['m1_spei_full']['n_obs']}, N_units={results['m1_spei_full']['n_units']}")

    # --- M2: SPEI DL + no vol_p_10y (cleaner, avoids double-counting P channel) ---
    print("\nFitting M2: SPEI DL — no vol_p_10y (avoid double-counting P)...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m2_spei_no_pvol"] = _fit_dl(
            df_spei, y="log_cbr", x="spei_growing", lags=3,
            controls=BASE_CONTROLS_NO_PVOL,
        )
    cum2 = results["m2_spei_no_pvol"]["cumulative_beta"]
    print(f"  Cumulative SPEI beta: {cum2:.4f}  (SE={results['m2_spei_no_pvol']['cumulative_se']:.4f})")

    # --- M3: Temperature comparator on same SPEI sample ---
    print("\nFitting M3: Temperature DL (comparator, same SPEI sample)...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m3_temperature_comparator"] = _fit_dl(
            df_spei, y="log_cbr", x="t_growing", lags=3,
            controls=BASE_CONTROLS,
        )
    cum3 = results["m3_temperature_comparator"]["cumulative_beta"]
    print(f"  Cumulative T beta: {cum3:.4f}  (SE={results['m3_temperature_comparator']['cumulative_se']:.4f})")

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
            "SPEI = growing-season (Apr-Sep) Thornthwaite SPEI, standardised "
            "per country (Path A, CRU 1901-1950 baseline + ModE-RA anomalies). "
            "All models restricted to SPEI-available rows (ModE-RA coverage "
            "1421-2008) for comparability. "
            "M2 drops vol_p_10y to avoid double-counting the precipitation "
            "channel already captured in SPEI's water-balance construction."
        ),
    }
    OUT.write_text(json.dumps(_clean(results), indent=2))
    print(f"\nWrote {OUT}")

    print("\n=== SUMMARY ===")
    print(f"M1 SPEI (full controls):  cum_beta = {cum1:.4f}")
    print(f"M2 SPEI (no vol_p_10y):   cum_beta = {cum2:.4f}")
    print(f"M3 Temperature comparator: cum_beta = {cum3:.4f}")
    irf1 = [r for r in results["m1_spei_full"]["irf"] if isinstance(r["lag"], int)]
    print("IRF by lag (M1 SPEI):")
    for r in irf1:
        print(f"  lag {r['lag']}: beta={r['beta']:.4f}  SE={r['se']:.4f}")


if __name__ == "__main__":
    run()
