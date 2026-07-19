"""Re-run the headline Long Shadow regression using REAL pre-industrial
seasonality measures from ModE-RA + CRU.

Original (in paper): pre-industrial seasonality (proxy) -> modern crop share
change and pop growth, beta = -3.10 / -0.42, p < 0.001, with pathway FE.

We replace the proxy seasonality with:
  - sigma_s_real: country-mean intra-annual temperature range, 1421-1750
  - productive_months: storage-demand index, 1421-1750
  - sigma_v_real: inter-annual T volatility, 1421-1750

Outcome variables (same as paper):
  - Delta crop share: 2015-2025 vs 1950-1960 country means
  - Pop growth (log): 2015-2025 vs 1950-1960
  - Delta urban share, Delta irrigation

Save results table to data/long_shadow_rerun.parquet.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}


def _build_modern_outcomes() -> pd.DataFrame:
    """Modern outcome changes per country: 2015-2025 vs 1950-1960."""
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        crop_share_early=("crop_share", "mean"),
        urban_share_early=("urban_share", "mean"),
        irrigation_share_early=("irrigation_share", "mean"),
        pop_early=("pop", "mean"),
    )
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        crop_share_late=("crop_share", "mean"),
        urban_share_late=("urban_share", "mean"),
        irrigation_share_late=("irrigation_share", "mean"),
        pop_late=("pop", "mean"),
    )
    m = early.merge(late, on="iso3", how="inner")
    m = m[(m["pop_early"] > 0) & (m["pop_late"] > 0)]
    m["d_crop"] = m["crop_share_late"] - m["crop_share_early"]
    m["d_urban"] = m["urban_share_late"] - m["urban_share_early"]
    m["d_irrigation"] = m["irrigation_share_late"] - m["irrigation_share_early"]
    m["log_pop_growth"] = np.log(m["pop_late"] / m["pop_early"])
    return m


def _preind_climate_features() -> pd.DataFrame:
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)]
    feats = pre.groupby("iso3", as_index=False).agg(
        sigma_s_real=("sigma_s", "mean"),
        t_mean=("t_mean", "mean"),
        p_annual=("p_annual", "mean"),
        growing_dd=("growing_dd", "mean"),
        sigma_v_real=("t_mean", "std"),
    )

    # productive-months and storage index from the storage-index analysis
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    df["prod"] = ((df["t_abs"] >= 5) & (df["t_abs"] <= 30) & (df["p_abs"] >= 30)).astype(float)
    pre_m = df[df["year"].between(1421, 1750)]
    yr = pre_m.groupby(["iso3", "year"], as_index=False).agg(prod_months=("prod", "sum"))
    pm = yr.groupby("iso3", as_index=False).agg(productive_months=("prod_months", "mean"))
    pm["storage_index"] = (12 - pm["productive_months"])
    return feats.merge(pm, on="iso3", how="inner")


def _attach_pathway(df: pd.DataFrame) -> pd.DataFrame:
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)
    clust["cluster"] = clust["cluster"].astype(int)
    return df.merge(clust[["iso3", "cluster"]], on="iso3", how="left")


def run_one(d: pd.DataFrame, outcome: str, predictor: str,
            pathway_fe: bool) -> dict:
    d = d.dropna(subset=[outcome, predictor]).copy()
    if pathway_fe:
        if "cluster" not in d or d["cluster"].isna().all():
            return {}
        d = d.dropna(subset=["cluster"])
        dums = pd.get_dummies(d["cluster"], drop_first=True).astype(float)
        X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                       d[[predictor]], dums], axis=1).astype(float)
    else:
        X = sm.add_constant(d[[predictor]])
    y = d[outcome].astype(float)
    if len(d) < 10:
        return {}
    res = sm.OLS(y, X).fit()
    return {
        "beta": res.params[predictor],
        "se": res.bse[predictor],
        "p": res.pvalues[predictor],
        "rsq": res.rsquared,
        "n": int(res.nobs),
    }


def main() -> None:
    outcomes = _build_modern_outcomes()
    feats = _preind_climate_features()
    df = outcomes.merge(feats, on="iso3", how="inner")
    df = _attach_pathway(df)
    print(f"Sample: {len(df)} countries, {df['cluster'].notna().sum()} with pathway")
    print()

    rows = []
    for predictor in ["sigma_s_real", "productive_months", "storage_index",
                      "sigma_v_real", "p_annual"]:
        for outcome in ["d_crop", "log_pop_growth", "d_urban", "d_irrigation"]:
            for fe in [False, True]:
                r = run_one(df, outcome, predictor, fe)
                if not r:
                    continue
                rows.append({
                    "predictor": predictor,
                    "outcome": outcome,
                    "pathway_FE": fe,
                    **r,
                })
    out = pd.DataFrame(rows)
    out["sig"] = np.where(out["p"] < 0.001, "***",
                  np.where(out["p"] < 0.01, "**",
                   np.where(out["p"] < 0.05, "*",
                    np.where(out["p"] < 0.10, ".", ""))))

    print("=== Long-Shadow re-run with REAL climate measures ===")
    for predictor in out["predictor"].unique():
        sub = out[out["predictor"] == predictor]
        print(f"\n-- predictor: {predictor} --")
        print(sub[["outcome", "pathway_FE", "beta", "se", "p", "rsq", "n", "sig"]]
              .to_string(index=False))

    out.to_parquet(DATA / "long_shadow_rerun.parquet", index=False)
    print(f"\nSaved {DATA / 'long_shadow_rerun.parquet'}")


if __name__ == "__main__":
    main()
