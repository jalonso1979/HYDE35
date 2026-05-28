"""Phase 10 Pillar C (follow-up): Distributed-lag regressions with 10-year rolling SD
as the uncertainty proxy (alternative to the within-season realized SD used in C6).

Motivation: within-season SD has only 5 dof/country-year → large measurement error
→ potential attenuation bias.  10-year centered rolling SD of annual anomalies has 9+
dof and captures medium-frequency climate variability that childbearing may respond to.

Two models (M2/ensstd comparator is unchanged so dropped here):
  M1r: baseline DL + 10-year rolling SD of annual growing-season anomaly
  M3r: baseline DL + rolling SD × Hansen-regime interaction

Output:
  /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_uncertainty_dl_rolling.json

Critical numbers reported:
  - Cumulative temperature beta per model
  - Rolling-SD coefficient per model (with cluster-robust SE)
  - M3r interaction coefficient (rolling_t_x_above)
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REAL_WAGE_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_uncertainty_dl_rolling.json"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Hansen Phase 9 threshold
THRESHOLD = 9.97

# Base controls — drop vol_t_10y/vol_p_10y to avoid collinearity with rolling SD
BASE_CONTROLS_NO_VOL = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought",
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

    # Interaction term for M3r
    df["rolling_t_x_above"] = df["t_growing_rolling10_sd"] * df["above_threshold"]

    # Confirm rolling SD columns present
    roll_cols = ["t_growing_rolling10_sd", "p_growing_rolling10_sd"]
    for col in roll_cols:
        n = df[col].notna().sum()
        print(f"  {col}: {n} non-null")

    results = {}

    # ------------------------------------------------------------------
    # M1r: baseline DL + 10-year rolling SD (headline rolling proxy)
    # ------------------------------------------------------------------
    print("\nFitting M1r: baseline DL + 10-year rolling SD...")
    m1r_controls = (["t_growing_rolling10_sd", "p_growing_rolling10_sd"]
                    + BASE_CONTROLS_NO_VOL)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m1r_baseline_with_rolling10_sd"] = _fit_with_controls(
            df, y="log_cbr", x="t_growing", lags=3,
            controls=m1r_controls,
        )
    cum = results["m1r_baseline_with_rolling10_sd"]["cumulative_beta"]
    ucoef = results["m1r_baseline_with_rolling10_sd"]["ctrl_coefs"].get(
        "t_growing_rolling10_sd", {})
    print(f"  M1r cumulative T beta: {cum:.4f}")
    print(f"  M1r rolling SD coef: {ucoef.get('beta', 'N/A'):.4f} "
          f"(SE={ucoef.get('se', 'N/A'):.4f})")

    # ------------------------------------------------------------------
    # M3r: rolling SD × Hansen-regime interaction
    # ------------------------------------------------------------------
    print("\nFitting M3r: rolling SD × Hansen threshold interaction...")
    m3r_controls = (["t_growing_rolling10_sd", "rolling_t_x_above", "above_threshold",
                     "p_growing_rolling10_sd"]
                    + BASE_CONTROLS_NO_VOL)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results["m3r_rolling_sd_x_threshold"] = _fit_with_controls(
            df, y="log_cbr", x="t_growing", lags=3,
            controls=m3r_controls,
        )
    cum3 = results["m3r_rolling_sd_x_threshold"]["cumulative_beta"]
    inter = results["m3r_rolling_sd_x_threshold"]["ctrl_coefs"].get(
        "rolling_t_x_above", {})
    sd_base = results["m3r_rolling_sd_x_threshold"]["ctrl_coefs"].get(
        "t_growing_rolling10_sd", {})
    print(f"  M3r cumulative T beta: {cum3:.4f}")
    print(f"  M3r rolling SD (base): {sd_base.get('beta', 'N/A'):.4f} "
          f"(SE={sd_base.get('se', 'N/A'):.4f})")
    print(f"  M3r interaction (rolling_t_x_above): {inter.get('beta', 'N/A'):.4f} "
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
    m1r_sd = results["m1r_baseline_with_rolling10_sd"]["ctrl_coefs"].get(
        "t_growing_rolling10_sd", {})
    m3r_sd = results["m3r_rolling_sd_x_threshold"]["ctrl_coefs"].get(
        "t_growing_rolling10_sd", {})
    m3r_int = results["m3r_rolling_sd_x_threshold"]["ctrl_coefs"].get(
        "rolling_t_x_above", {})
    print(f"M1r cumul T beta = {cum:.4f}, rolling_SD coef = {m1r_sd.get('beta','N/A'):.4f}"
          f"  SE = {m1r_sd.get('se','N/A'):.4f}")
    print(f"M3r cumul T beta = {cum3:.4f}, rolling_SD base = {m3r_sd.get('beta','N/A'):.4f}"
          f"  SE = {m3r_sd.get('se','N/A'):.4f}"
          f",  interaction = {m3r_int.get('beta','N/A'):.4f}"
          f"  SE = {m3r_int.get('se','N/A'):.4f}")


if __name__ == "__main__":
    run()
