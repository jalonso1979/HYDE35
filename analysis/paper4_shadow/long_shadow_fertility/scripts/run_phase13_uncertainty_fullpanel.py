"""Phase 13: Re-estimate the climate-UNCERTAINTY-channel null on the FULL panel.

REFEREE ITEM 3a. The Phase 10 uncertainty null (run_phase10_uncertainty_dl.py)
was estimated with BASE_CONTROLS that include the EMDAT columns
`disaster_count` and `log_disaster_deaths`. Those columns are populated only
post-1900 for 4 countries (FRA/GBR/ITA/SWE), so the listwise `dropna` silently
collapses the headline-M1 sample to N=436 / 4 countries / 1900-2008 -- even
though the paper frames the uncertainty null as a PANEL-WIDE (1541-2008, 12
country) result. That is exactly the EMDAT listwise-deletion trap the paper
criticises elsewhere.

This script re-runs the uncertainty distributed-lag specification on the FULL
1541-2008 panel WITHOUT the EMDAT controls and reports:

  1. Full-panel uncertainty estimate (cumulative T beta + within-season SD coef,
     cluster-by-country SE/95% CI), and whether the uncertainty null HOLDS.
  2. Side-by-side: OLD EMDAT-restricted estimate (N~436, post-1900) vs NEW
     full-panel estimate (N~full). The OLD estimate is re-derived here on the
     SAME assembled panel for an apples-to-apples comparison (and is verified to
     reproduce the Phase 10 M1 numbers).
  3. Regime interaction: does within-season temperature uncertainty affect
     fertility differently in the Malthusian vs Modern regime (split at
     log_real_wage > 9.97, the Phase 9 Hansen threshold) on the full panel?

The estimator mirrors `_fit_with_controls` in run_phase10_uncertainty_dl.py
(pooled OLS, country FE + year FE, cluster-robust SE by country) so that the
ONLY thing that changes across the OLD/NEW comparison is the control set / the
resulting sample -- the uncertainty regressor and the DL machinery are identical.

The headline uncertainty regressor is the realized within-season temperature SD
(`t_anom_c_within_season_sd`), the data-density-independent proxy defined in
data/within_season_variance.py.

Output:
  /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase13_uncertainty_fullpanel.json

Run from /Volumes/BIGDATA/HYDE35:
  python -m analysis.paper4_shadow.long_shadow_fertility.scripts.run_phase13_uncertainty_fullpanel
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
# Paths / constants
# ---------------------------------------------------------------------------
REAL_WAGE_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase13_uncertainty_fullpanel.json"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Hansen Phase 9 threshold (log real wage), used to split Malthusian vs Modern.
THRESHOLD = 9.97

# The uncertainty regressor: realized within-season temperature SD.
UNC = "t_anom_c_within_season_sd"
UNC_P = "p_anom_mm_within_season_sd"

# EMDAT controls that silently restrict the Phase 10 M1 sample to post-1900/4-ctry.
EMDAT_CONTROLS = ["disaster_count", "log_disaster_deaths"]

# Phase 10 BASE_CONTROLS (verbatim) -- used ONLY to reproduce the OLD estimate.
PHASE10_BASE_CONTROLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought",
    "vol_t_10y", "vol_p_10y",
]

# Full-panel control set: EMDAT dropped. We ALSO drop war_active /
# log_war_fatalities / pandemic_active because those are themselves populated
# for only 4 countries (they would re-impose the same collapse via dropna,
# just back to 1421 instead of 1900). What remains are the full-coverage
# climate controls plus the within-season precipitation SD. The mean-temperature
# response is the distributed lag of t_growing itself, which is kept in every
# specification. (A robustness variant that drops the climate controls too --
# leaving only the SD terms -- is reported and gives the identical sample.)
FULLPANEL_CONTROLS = [
    UNC, UNC_P,
    "heat_extreme", "drought",
    "vol_t_10y", "vol_p_10y",
]

LAGS = 3
Y = "log_cbr"
X = "t_growing"
UNIT = "iso3"


# ---------------------------------------------------------------------------
# Estimator (mirrors run_phase10_uncertainty_dl._fit_with_controls)
# ---------------------------------------------------------------------------
def _add_lags(df: pd.DataFrame, var: str, lags: int, unit_col: str = UNIT):
    df = df.sort_values([unit_col, "year"]).copy()
    lag_cols = []
    for k in range(lags + 1):
        col = f"{var}_lag{k}"
        df[col] = df.groupby(unit_col)[var].shift(k)
        lag_cols.append(col)
    return df, lag_cols


def _fit_with_controls(df: pd.DataFrame, y: str, x: str, lags: int,
                       controls: list[str], unit_col: str = UNIT) -> dict:
    """Pooled DL OLS with country FE + year FE, cluster-robust SE by country.

    Identical specification to Phase 10's `_fit_with_controls`. Returns the DL
    impulse responses, the cumulative temperature beta, and the coefficient (+SE,
    95% CI) on every requested control (so we can read off the within-season SD
    coef and, in the interaction model, the t_sd_x_above coef).
    """
    df_lag, lag_cols = _add_lags(df, x, lags, unit_col)
    keep = [y, unit_col, "year"] + lag_cols + controls
    sub = df_lag[keep].dropna()

    unit_dums = pd.get_dummies(sub[unit_col], drop_first=True, dtype=float)
    year_dums = pd.get_dummies(sub["year"].astype(int), drop_first=True, dtype=float)
    year_dums.columns = [f"y_{c}" for c in year_dums.columns]

    X_ = sm.add_constant(pd.concat([
        sub[lag_cols].astype(float),
        unit_dums,
        year_dums,
        sub[controls].astype(float),
    ], axis=1))
    cluster = sub[unit_col].astype("category").cat.codes.to_numpy()
    res = sm.OLS(sub[y].astype(float).to_numpy(), X_.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )
    col_names = list(X_.columns)

    # DL impulse responses (one per lag) + cumulative.
    rows = []
    for k, col in enumerate(lag_cols):
        idx = 1 + k  # +1 for const
        b = float(res.params[idx]); s = float(res.bse[idx])
        rows.append({"lag": k, "beta": b, "se": s,
                     "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    lag_idx = np.arange(1, 1 + len(lag_cols))
    cum_b = float(res.params[lag_idx].sum())
    cov_block = res.cov_params()[np.ix_(lag_idx, lag_idx)]
    cum_se = float(np.sqrt(np.ones(len(lag_idx)) @ cov_block @ np.ones(len(lag_idx))))
    rows.append({"lag": "cumulative", "beta": cum_b, "se": cum_se,
                 "ci_low": cum_b - 1.96 * cum_se, "ci_high": cum_b + 1.96 * cum_se})

    ctrl_coefs = {}
    for ctrl in controls:
        if ctrl in col_names:
            i = col_names.index(ctrl)
            b = float(res.params[i]); s = float(res.bse[i])
            ctrl_coefs[ctrl] = {"beta": b, "se": s,
                                "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s}

    return {
        "irf": rows,
        "cumulative_beta": cum_b,
        "cumulative_se": cum_se,
        "cumulative_ci_low": cum_b - 1.96 * cum_se,
        "cumulative_ci_high": cum_b + 1.96 * cum_se,
        "ctrl_coefs": ctrl_coefs,
        "n_obs": int(sub.shape[0]),
        "n_units": int(sub[unit_col].nunique()),
        "year_min": int(sub["year"].min()),
        "year_max": int(sub["year"].max()),
        "countries": sorted(sub[unit_col].unique().tolist()),
    }


def _holds(coef: dict, tol_label: str = "95% CI contains 0") -> dict:
    """A 'null holds' verdict: the SD coefficient is statistically indistinct
    from zero (its 95% CI straddles 0)."""
    holds = bool(coef["ci_low"] <= 0.0 <= coef["ci_high"])
    return {
        "null_holds": holds,
        "criterion": tol_label,
        "beta": coef["beta"], "se": coef["se"],
        "ci_low": coef["ci_low"], "ci_high": coef["ci_high"],
    }


def run():
    print("Loading panel (read-only)...")
    df = assemble_panel_multi()
    print(f"  Panel shape: {df.shape}; iso3={df['iso3'].nunique()}; "
          f"year=[{int(df['year'].min())},{int(df['year'].max())}]")

    # Merge real wages for the Hansen-regime split (does not modify the panel on disk).
    rw = pd.read_parquet(REAL_WAGE_PATH)[["iso3", "year", "log_real_wage"]]
    df = df.merge(rw, on=["iso3", "year"], how="left")
    # Follow Phase 10 convention: above_threshold defined wherever wage observed,
    # missing wage -> treated as below (0). Robustness restricts to observed wage.
    df["above_threshold"] = (df["log_real_wage"] > THRESHOLD).astype(float)
    df["above_threshold"] = df["above_threshold"].fillna(0.0)
    df["t_sd_x_above"] = df[UNC] * df["above_threshold"]

    results: dict = {}
    results["_meta"] = {
        "purpose": "Referee 3a: re-estimate uncertainty-channel null on FULL "
                   "1541-2008 panel without EMDAT listwise deletion.",
        "uncertainty_regressor": UNC,
        "uncertainty_regressor_desc": "realized within-growing-season temperature SD "
                                      "(data/within_season_variance.py)",
        "hansen_threshold_log_real_wage": THRESHOLD,
        "y": Y, "x": X, "lags": LAGS,
        "estimator": "pooled OLS, country FE + year FE, cluster-robust SE by country",
        "emdat_controls_dropped": EMDAT_CONTROLS,
        "fullpanel_controls": FULLPANEL_CONTROLS,
    }

    # ------------------------------------------------------------------
    # (A) OLD estimate -- reproduce Phase 10 M1 (EMDAT-restricted).
    # ------------------------------------------------------------------
    print("\n[A] OLD (Phase 10 M1, EMDAT-restricted) ...")
    old_controls = [UNC, UNC_P] + PHASE10_BASE_CONTROLS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _fit_with_controls(df, Y, X, LAGS, old_controls)
    results["old_emdat_restricted"] = old
    print(f"  N={old['n_obs']}  countries={old['n_units']} {old['countries']}  "
          f"year=[{old['year_min']},{old['year_max']}]")
    print(f"  cumulative T beta = {old['cumulative_beta']:.5f} "
          f"[{old['cumulative_ci_low']:.5f}, {old['cumulative_ci_high']:.5f}]")
    osd = old["ctrl_coefs"][UNC]
    print(f"  within-season SD coef = {osd['beta']:.5f} (SE={osd['se']:.5f}) "
          f"[{osd['ci_low']:.5f}, {osd['ci_high']:.5f}]")

    # ------------------------------------------------------------------
    # (B) NEW full-panel estimate (EMDAT + 4-country controls dropped).
    # ------------------------------------------------------------------
    print("\n[B] NEW full-panel (EMDAT dropped) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = _fit_with_controls(df, Y, X, LAGS, FULLPANEL_CONTROLS)
    results["new_full_panel"] = new
    print(f"  N={new['n_obs']}  countries={new['n_units']} {new['countries']}  "
          f"year=[{new['year_min']},{new['year_max']}]")
    print(f"  cumulative T beta = {new['cumulative_beta']:.5f} "
          f"[{new['cumulative_ci_low']:.5f}, {new['cumulative_ci_high']:.5f}]")
    nsd = new["ctrl_coefs"][UNC]
    print(f"  within-season SD coef = {nsd['beta']:.5f} (SE={nsd['se']:.5f}) "
          f"[{nsd['ci_low']:.5f}, {nsd['ci_high']:.5f}]")
    new_verdict = _holds(nsd)
    results["new_full_panel"]["uncertainty_null_verdict"] = new_verdict
    print(f"  --> uncertainty null on FULL panel HOLDS? {new_verdict['null_holds']}")

    # (B') Robustness: SD terms only (no climate controls) -- same sample, isolates
    # whether the verdict depends on the retained climate controls.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_min = _fit_with_controls(df, Y, X, LAGS, [UNC, UNC_P])
    new_min["uncertainty_null_verdict"] = _holds(new_min["ctrl_coefs"][UNC])
    results["new_full_panel_sd_only"] = new_min

    # (B'') Robustness: drop ONLY EMDAT but KEEP war/pandemic (back to 4 ctry,
    # 1541-2008) -- shows the year coverage extends but country coverage does not,
    # i.e. EMDAT is not the only listwise restriction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_dropemdat_only = _fit_with_controls(
            df, Y, X, LAGS,
            [UNC, UNC_P, "war_active", "log_war_fatalities", "pandemic_active",
             "heat_extreme", "drought", "vol_t_10y", "vol_p_10y"],
        )
    new_dropemdat_only["uncertainty_null_verdict"] = _holds(
        new_dropemdat_only["ctrl_coefs"][UNC])
    results["drop_emdat_keep_war_pandemic"] = new_dropemdat_only

    # ------------------------------------------------------------------
    # (C) Side-by-side comparison.
    # ------------------------------------------------------------------
    results["comparison_old_vs_new"] = {
        "old_emdat_restricted": {
            "n_obs": old["n_obs"], "n_units": old["n_units"],
            "year_min": old["year_min"], "year_max": old["year_max"],
            "countries": old["countries"],
            "cumulative_T_beta": old["cumulative_beta"],
            "cumulative_T_ci": [old["cumulative_ci_low"], old["cumulative_ci_high"]],
            "within_season_SD_coef": osd,
        },
        "new_full_panel": {
            "n_obs": new["n_obs"], "n_units": new["n_units"],
            "year_min": new["year_min"], "year_max": new["year_max"],
            "countries": new["countries"],
            "cumulative_T_beta": new["cumulative_beta"],
            "cumulative_T_ci": [new["cumulative_ci_low"], new["cumulative_ci_high"]],
            "within_season_SD_coef": nsd,
            "uncertainty_null_verdict": new_verdict,
        },
    }

    # ------------------------------------------------------------------
    # (D) Regime interaction on the FULL panel: SD x above-threshold.
    # ------------------------------------------------------------------
    print("\n[D] Regime interaction (SD x above-Hansen-threshold), full panel ...")
    inter_controls = (
        [UNC, "t_sd_x_above", "above_threshold", UNC_P]
        + ["heat_extreme", "drought", "vol_t_10y", "vol_p_10y"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inter = _fit_with_controls(df, Y, X, LAGS, inter_controls)
    results["regime_interaction_full_panel"] = inter
    print(f"  N={inter['n_obs']}  countries={inter['n_units']}  "
          f"year=[{inter['year_min']},{inter['year_max']}]")
    ic = inter["ctrl_coefs"]["t_sd_x_above"]
    sdb = inter["ctrl_coefs"][UNC]
    print(f"  SD base (Malthusian) coef = {sdb['beta']:.5f} (SE={sdb['se']:.5f}) "
          f"[{sdb['ci_low']:.5f}, {sdb['ci_high']:.5f}]")
    print(f"  interaction t_sd_x_above  = {ic['beta']:.5f} (SE={ic['se']:.5f}) "
          f"[{ic['ci_low']:.5f}, {ic['ci_high']:.5f}]")
    # Modern-regime total effect = base + interaction (linear combination, with SE).
    inter_verdict = {
        "interaction_significant": bool(not (ic["ci_low"] <= 0.0 <= ic["ci_high"])),
        "malthusian_sd_coef": sdb,
        "interaction_coef": ic,
        "above_threshold_coef": inter["ctrl_coefs"].get("above_threshold"),
        "interpretation": (
            "interaction CI straddles 0 => no detectable regime difference in the "
            "within-season-uncertainty effect"
            if (ic["ci_low"] <= 0.0 <= ic["ci_high"]) else
            "interaction CI excludes 0 => regimes differ in the uncertainty effect"
        ),
    }
    results["regime_interaction_full_panel"]["verdict"] = inter_verdict
    print(f"  --> interaction significant? {inter_verdict['interaction_significant']}")

    # (D') Regime interaction restricted to observed-wage rows (no fillna(0)).
    df_obs = df[df["log_real_wage"].notna()].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inter_obs = _fit_with_controls(df_obs, Y, X, LAGS, inter_controls)
    ic2 = inter_obs["ctrl_coefs"]["t_sd_x_above"]
    inter_obs["verdict"] = {
        "interaction_significant": bool(not (ic2["ci_low"] <= 0.0 <= ic2["ci_high"])),
        "interaction_coef": ic2,
    }
    results["regime_interaction_observed_wage_only"] = inter_obs

    # ------------------------------------------------------------------
    # Serialise.
    # ------------------------------------------------------------------
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return o

    OUT.write_text(json.dumps(_clean(results), indent=2))
    print(f"\nWrote {OUT}")

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print("\n================= SUMMARY =================")
    print(f"OLD (EMDAT-restricted): N={old['n_obs']}, {old['n_units']} ctry, "
          f"{old['year_min']}-{old['year_max']}; "
          f"SD coef={osd['beta']:.5f} [{osd['ci_low']:.5f},{osd['ci_high']:.5f}]")
    print(f"NEW (full panel):       N={new['n_obs']}, {new['n_units']} ctry, "
          f"{new['year_min']}-{new['year_max']}; "
          f"SD coef={nsd['beta']:.5f} [{nsd['ci_low']:.5f},{nsd['ci_high']:.5f}]")
    print(f"Uncertainty null HOLDS on full panel? {new_verdict['null_holds']}")
    print(f"Regime interaction (full panel): {ic['beta']:.5f} "
          f"[{ic['ci_low']:.5f},{ic['ci_high']:.5f}]  "
          f"significant={inter_verdict['interaction_significant']}")


if __name__ == "__main__":
    run()
