"""Phase 10 Pillar C: Distributed-lag regressions with climate uncertainty channel.

Three models:
  M1: baseline DL + within-season realized SD (headline uncertainty proxy)
  M2: baseline DL + ModE-RA ensstd (comparator)
  M3: realized SD × Hansen threshold interaction (tests regime heterogeneity)

Output:
  /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_uncertainty_dl.json

Critical numbers reported:
  - Cumulative temperature beta per model
  - Uncertainty coefficient per model (realized SD or ensstd)
  - M3 interaction coefficient (t_sd_x_above)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Allow running as a script or as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # -> /Volumes/BIGDATA/HYDE35

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_distributed_lag import (
    fit_pooled_distributed_lag,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REAL_WAGE_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_uncertainty_dl.json"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Hansen Phase 9 threshold
THRESHOLD = 9.97

BASE_CONTROLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought",
    "vol_t_10y", "vol_p_10y",
]


def _add_lags(df: pd.DataFrame, var: str, lags: int, unit_col: str = "iso3"):
    df = df.sort_values([unit_col, "year"]).copy()
    lag_cols = []
    for k in range(lags + 1):
        col = f"{var}_lag{k}"
        df[col] = df.groupby(unit_col)[var].shift(k)
        lag_cols.append(col)
    return df, lag_cols


def _fit_with_controls(df: pd.DataFrame, y: str, x: str, lags: int,
                       controls: list[str], unit_col: str = "iso3") -> dict:
    """Run DL with controls; return DL irf DataFrame + full coef dict."""
    df_lag, lag_cols = _add_lags(df, x, lags, unit_col)
    keep = [y, unit_col, "year"] + lag_cols + controls
    sub = df_lag[keep].dropna()

    unit_dums = pd.get_dummies(sub[unit_col], drop_first=True, dtype=float)
    year_dums = pd.get_dummies(sub["year"].astype(int), drop_first=True, dtype=float)
    year_dums.columns = [f"y_{c}" for c in year_dums.columns]

    X = sm.add_constant(pd.concat([
        sub[lag_cols].astype(float),
        unit_dums,
        year_dums,
        sub[controls].astype(float),
    ], axis=1))
    cluster = sub[unit_col].astype("category").cat.codes.to_numpy()
    res = sm.OLS(sub[y].astype(float).to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )
    col_names = list(X.columns)

    # DL irf rows
    rows = []
    for k, col in enumerate(lag_cols):
        idx = 1 + k  # +1 for const
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
    irf_df = pd.DataFrame(rows)

    # Control coefficients (uncertainty vars + key controls)
    ctrl_coefs = {}
    for ctrl in controls:
        if ctrl in col_names:
            i = col_names.index(ctrl)
            b = float(res.params[i])
            s = float(res.bse[i])
            ctrl_coefs[ctrl] = {"beta": b, "se": s,
                                 "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s}

    return {
        "irf": irf_df.to_dict(orient="records"),
        "cumulative_beta": cum_b,
        "cumulative_se": cum_se,
        "ctrl_coefs": ctrl_coefs,
        "n_obs": int(sub.shape[0]),
        "n_units": int(sub[unit_col].nunique()),
    }


def run():
    print("Loading panel...")
    df = assemble_panel_multi()
    print(f"  Panel shape: {df.shape}")

    # Merge real wages for Hansen threshold
    rw = pd.read_parquet(REAL_WAGE_PATH)[["iso3", "year", "log_real_wage"]]
    df = df.merge(rw, on=["iso3", "year"], how="left")
    df["above_threshold"] = (df["log_real_wage"] > THRESHOLD).astype(float)
    df["above_threshold"] = df["above_threshold"].fillna(0)

    # Interaction term for M3
    df["t_sd_x_above"] = df["t_anom_c_within_season_sd"] * df["above_threshold"]

    # Confirm uncertainty columns present
    unc_cols = ["t_anom_c_within_season_sd", "p_anom_mm_within_season_sd",
                "ensstd_t_growing", "ensstd_p_growing"]
    for col in unc_cols:
        n = df[col].notna().sum()
        print(f"  {col}: {n} non-null")

    results = {}

    # ------------------------------------------------------------------
    # M1: baseline DL + within-season realized SD (headline)
    # ------------------------------------------------------------------
    print("\nFitting M1: baseline DL + realized SD...")
    m1_controls = (["t_anom_c_within_season_sd", "p_anom_mm_within_season_sd"]
                   + BASE_CONTROLS)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m1_baseline_with_realized_sd"] = _fit_with_controls(
            df, y="log_cbr", x="t_growing", lags=3,
            controls=m1_controls,
        )
    cum = results["m1_baseline_with_realized_sd"]["cumulative_beta"]
    ucoef = results["m1_baseline_with_realized_sd"]["ctrl_coefs"].get(
        "t_anom_c_within_season_sd", {})
    print(f"  M1 cumulative T beta: {cum:.4f}")
    print(f"  M1 realized SD coef: {ucoef.get('beta', 'N/A'):.4f} "
          f"(SE={ucoef.get('se', 'N/A'):.4f})")

    # ------------------------------------------------------------------
    # M2: ensstd comparator
    # ------------------------------------------------------------------
    print("\nFitting M2: baseline DL + ensstd comparator...")
    m2_controls = ["ensstd_t_growing", "ensstd_p_growing"] + BASE_CONTROLS
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m2_baseline_with_ensstd"] = _fit_with_controls(
            df, y="log_cbr", x="t_growing", lags=3,
            controls=m2_controls,
        )
    cum2 = results["m2_baseline_with_ensstd"]["cumulative_beta"]
    ucoef2 = results["m2_baseline_with_ensstd"]["ctrl_coefs"].get(
        "ensstd_t_growing", {})
    print(f"  M2 cumulative T beta: {cum2:.4f}")
    print(f"  M2 ensstd_t coef: {ucoef2.get('beta', 'N/A'):.4f} "
          f"(SE={ucoef2.get('se', 'N/A'):.4f})")

    # ------------------------------------------------------------------
    # M3: realized SD × Hansen-regime interaction
    # ------------------------------------------------------------------
    print("\nFitting M3: realized SD × Hansen threshold interaction...")
    m3_controls = (["t_anom_c_within_season_sd", "t_sd_x_above",
                    "above_threshold", "p_anom_mm_within_season_sd"]
                   + BASE_CONTROLS)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m3_realized_sd_x_threshold"] = _fit_with_controls(
            df, y="log_cbr", x="t_growing", lags=3,
            controls=m3_controls,
        )
    cum3 = results["m3_realized_sd_x_threshold"]["cumulative_beta"]
    inter = results["m3_realized_sd_x_threshold"]["ctrl_coefs"].get(
        "t_sd_x_above", {})
    sd_base = results["m3_realized_sd_x_threshold"]["ctrl_coefs"].get(
        "t_anom_c_within_season_sd", {})
    print(f"  M3 cumulative T beta: {cum3:.4f}")
    print(f"  M3 realized SD (base): {sd_base.get('beta', 'N/A'):.4f} "
          f"(SE={sd_base.get('se', 'N/A'):.4f})")
    print(f"  M3 interaction (t_sd_x_above): {inter.get('beta', 'N/A'):.4f} "
          f"(SE={inter.get('se', 'N/A'):.4f})")

    # ------------------------------------------------------------------
    # Serialise
    # ------------------------------------------------------------------
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

    OUT.write_text(json.dumps(_clean(results), indent=2))
    print(f"\nWrote {OUT}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"M1 cumulative T beta = {cum:.4f}, "
          f"realized_SD coef = {results['m1_baseline_with_realized_sd']['ctrl_coefs'].get('t_anom_c_within_season_sd', {}).get('beta', 'N/A')}")
    print(f"M2 cumulative T beta = {cum2:.4f}, "
          f"ensstd_t coef = {results['m2_baseline_with_ensstd']['ctrl_coefs'].get('ensstd_t_growing', {}).get('beta', 'N/A')}")
    print(f"M3 cumulative T beta = {cum3:.4f}, "
          f"t_sd_x_above coef = {results['m3_realized_sd_x_threshold']['ctrl_coefs'].get('t_sd_x_above', {}).get('beta', 'N/A')}")


if __name__ == "__main__":
    run()
