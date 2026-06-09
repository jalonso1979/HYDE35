"""Sub-national long-shadow regression: pre-industrial sub-national climate
(1421-1750) -> modern sub-national outcome changes (1955-1960 vs 2015-2020),
with country fixed effects.

If the pre-industrial climate measures predict modern within-country
sub-unit outcomes after country FE, that is the cleanest possible
persistence-of-climate-endowments result: every country-level confounder
(institutions, language, colonization, latitude band) is absorbed.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def _build_subnational_features() -> pd.DataFrame:
    """Pre-industrial sub-national climate features 1421-1750."""
    mod = pd.read_parquet(DATA / "modera_subnational_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    mod = mod.merge(clim, on=["iso3", "month"], how="inner")
    mod["t_abs"] = mod["t_anom_c"] + mod["tmp_c_clim"]
    mod["p_abs"] = (mod["p_anom_mm"] + mod["pre_mm_clim"]).clip(lower=0)
    pre = mod[mod["year"].between(1421, 1750)].copy()
    pre["prod"] = ((pre["t_abs"] >= 5) & (pre["t_abs"] <= 30)
                   & (pre["p_abs"] >= 30)).astype(np.int8)
    yr = pre.groupby(["sub_id", "iso3", "year"], as_index=False).agg(
        prod_months=("prod", "sum"),
        t_mean=("t_abs", "mean"),
        t_max=("t_abs", "max"),
        t_min=("t_abs", "min"),
        p_total=("p_abs", "sum"),
    )
    yr["sigma_s"] = yr["t_max"] - yr["t_min"]
    feats = yr.groupby(["sub_id", "iso3"], as_index=False).agg(
        productive_months=("prod_months", "mean"),
        sigma_s_real=("sigma_s", "mean"),
        sigma_v_real=("t_mean", "std"),       # year-to-year SD of annual mean
        t_mean_preind=("t_mean", "mean"),
        p_annual_preind=("p_total", "mean"),
    )
    return feats


def _modern_outcome_changes() -> pd.DataFrame:
    """Modern sub-national outcome changes (1955-1960 baseline vs 2015-2020 late)."""
    hyde = pd.read_parquet(DATA / "subnational_hyde.parquet")
    e = hyde[hyde["year"].between(1955, 1960)].groupby(
        ["sub_id", "iso3"], as_index=False).agg(
        pop_e=("subpop", "mean"),
        crop_e=("cropland_ha", "mean"),
        graz_e=("grazing_ha", "mean"),
    )
    l = hyde[hyde["year"].between(2015, 2020)].groupby(
        ["sub_id", "iso3"], as_index=False).agg(
        pop_l=("subpop", "mean"),
        crop_l=("cropland_ha", "mean"),
        graz_l=("grazing_ha", "mean"),
    )
    m = e.merge(l, on=["sub_id", "iso3"])
    m["crop_share_e"] = np.where(m["crop_e"] + m["graz_e"] > 0,
                                  m["crop_e"] / (m["crop_e"] + m["graz_e"]), np.nan)
    m["crop_share_l"] = np.where(m["crop_l"] + m["graz_l"] > 0,
                                  m["crop_l"] / (m["crop_l"] + m["graz_l"]), np.nan)
    m["d_crop_share"] = m["crop_share_l"] - m["crop_share_e"]
    m["log_pop_growth"] = np.where(
        (m["pop_e"] > 0) & (m["pop_l"] > 0),
        np.log(m["pop_l"] / m["pop_e"]), np.nan,
    )
    return m[["sub_id", "iso3", "d_crop_share", "log_pop_growth"]]


def fe_reg(df: pd.DataFrame, outcome: str, predictor: str,
           use_fe: bool, min_per_group: int = 2) -> dict:
    d = df.dropna(subset=[outcome, predictor, "iso3"]).copy()
    if use_fe:
        counts = d["iso3"].value_counts()
        d = d[d["iso3"].isin(counts[counts >= min_per_group].index)]
    if len(d) < 30:
        return {}
    if use_fe:
        g = d.groupby("iso3")
        d[outcome] = d[outcome] - g[outcome].transform("mean")
        d[predictor] = d[predictor] - g[predictor].transform("mean")
    X = sm.add_constant(d[[predictor]])
    y = d[outcome].astype(float)
    n_g = d["iso3"].nunique()
    if use_fe and n_g > 1:
        r = sm.OLS(y, X).fit(cov_type="cluster",
                              cov_kwds={"groups": d["iso3"].values})
    else:
        r = sm.OLS(y, X).fit(cov_type="HC1")
    return {
        "beta": r.params[predictor], "se": r.bse[predictor],
        "p": r.pvalues[predictor], "rsq": r.rsquared,
        "n": int(r.nobs), "n_groups": n_g,
    }


def main() -> None:
    print("Building features...")
    feats = _build_subnational_features()
    outcomes = _modern_outcome_changes()
    df = feats.merge(outcomes, on=["sub_id", "iso3"], how="inner")
    print(f"Sample: {len(df):,} sub-units, {df['iso3'].nunique()} countries")
    df.to_parquet(DATA / "subnational_long_shadow_panel.parquet", index=False)

    print("\n=== Sub-national long shadow: pre-industrial -> modern, country FE ===")
    rows = []
    for predictor in ["sigma_v_real", "sigma_s_real", "productive_months"]:
        for outcome in ["log_pop_growth", "d_crop_share"]:
            for use_fe in [False, True]:
                r = fe_reg(df, outcome, predictor, use_fe)
                if not r: continue
                rows.append({
                    "predictor": predictor, "outcome": outcome,
                    "country_FE": use_fe,
                    "beta": r["beta"], "se": r["se"], "p": r["p"],
                    "rsq": r["rsq"], "n": r["n"], "n_groups": r["n_groups"],
                })
    res = pd.DataFrame(rows)
    res["sig"] = np.where(res["p"] < 0.001, "***",
                  np.where(res["p"] < 0.01, "**",
                   np.where(res["p"] < 0.05, "*",
                    np.where(res["p"] < 0.10, ".", ""))))
    print(res.to_string(index=False))
    res.to_parquet(DATA / "subnational_long_shadow_results.parquet", index=False)


if __name__ == "__main__":
    main()
