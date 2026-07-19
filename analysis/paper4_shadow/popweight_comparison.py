"""Compare area-weighted vs population-weighted ModE-RA aggregations.

For the headline regressions of the paper, we re-derive the climate features
from the pop-weighted panels and re-estimate. We report old vs new coefficients
side by side. The pop-weighted version captures the climate where people
actually lived in the pre-industrial era rather than the climate over the
empty quarters of large countries.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def _country_features(mod: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    pre = df[df["year"].between(1421, 1750)].copy()
    pre["prod"] = ((pre["t_abs"] >= 5) & (pre["t_abs"] <= 30)
                   & (pre["p_abs"] >= 30)).astype(float)
    yr = pre.groupby(["iso3", "year"], as_index=False).agg(
        prod_months=("prod", "sum"),
        t_max=("t_abs", "max"),
        t_min=("t_abs", "min"),
        t_mean=("t_abs", "mean"),
        p_annual=("p_abs", "sum"),
    )
    yr["sigma_s"] = yr["t_max"] - yr["t_min"]
    feats = yr.groupby("iso3", as_index=False).agg(
        productive_months=("prod_months", "mean"),
        sigma_s=("sigma_s", "mean"),
        sigma_v=("t_mean", "std"),
        t_mean=("t_mean", "mean"),
        p_annual=("p_annual", "mean"),
    )
    return feats


def _stage1_anova(feats: pd.DataFrame, label: str) -> dict:
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    d = feats.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    big = d[d["cluster"] != 2]
    groups = [big.loc[big["cluster"] == k, "productive_months"].values
              for k in sorted(big["cluster"].unique())]
    F, p = stats.f_oneway(*[g[~np.isnan(g)] for g in groups])
    return {"label": label, "N": len(big), "F_anova": F, "p_anova": p,
            "mean_prod_months": big["productive_months"].mean(),
            "sd_prod_months": big["productive_months"].std()}


def _long_shadow(feats: pd.DataFrame, label: str) -> dict:
    """sigma_v -> log pop growth, with and without latitude control."""
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    centroids = ext[["iso3", "centroid_lat"]].drop_duplicates()
    centroids["abs_lat"] = centroids["centroid_lat"].abs()
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        pop_e=("pop", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        pop_l=("pop", "mean"))
    out = early.merge(late, on="iso3")
    out = out[(out["pop_e"] > 0) & (out["pop_l"] > 0)]
    out["log_pop_growth"] = np.log(out["pop_l"] / out["pop_e"])
    df = out.merge(feats, on="iso3").merge(centroids[["iso3", "abs_lat"]], on="iso3")
    df = df.dropna(subset=["sigma_v", "log_pop_growth", "abs_lat"])
    X1 = sm.add_constant(df[["sigma_v"]])
    r1 = sm.OLS(df["log_pop_growth"], X1).fit(cov_type="HC1")
    X2 = sm.add_constant(df[["sigma_v", "abs_lat"]])
    r2 = sm.OLS(df["log_pop_growth"], X2).fit(cov_type="HC1")
    return {"label": label, "N": int(r1.nobs),
            "beta_noFE": r1.params["sigma_v"], "p_noFE": r1.pvalues["sigma_v"],
            "beta_lat": r2.params["sigma_v"], "p_lat": r2.pvalues["sigma_v"]}


def main() -> None:
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")

    # OLD (area-weighted)
    print("Loading area-weighted (current) ModE-RA panel...", flush=True)
    mod_old = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    feats_old = _country_features(mod_old, clim)
    print(f"  {len(feats_old)} countries")

    # NEW (pop-weighted)
    print("Loading pop-weighted ModE-RA panel...", flush=True)
    mod_new = pd.read_parquet(DATA / "modera_country_monthly_popw.parquet")
    feats_new = _country_features(mod_new, clim)
    print(f"  {len(feats_new)} countries")

    # ── Stage 1 ANOVA ─────────────────────────────────────────────────
    print("\n=== Stage 1: productive-months by pathway (ANOVA) ===")
    a_old = _stage1_anova(feats_old, "area-weighted")
    a_new = _stage1_anova(feats_new, "pop-weighted")
    for r in [a_old, a_new]:
        print(f"  {r['label']:<20s} N={r['N']:>4}  F={r['F_anova']:.3f}  "
              f"p={r['p_anova']:.4g}  prod_months mean={r['mean_prod_months']:.2f} "
              f"sd={r['sd_prod_months']:.2f}")

    # ── Long shadow ───────────────────────────────────────────────────
    print("\n=== Long shadow: σ_v → log pop growth, with vs without |lat| ===")
    s_old = _long_shadow(feats_old, "area-weighted")
    s_new = _long_shadow(feats_new, "pop-weighted")
    for r in [s_old, s_new]:
        print(f"  {r['label']:<20s} N={r['N']:>4}  "
              f"β (no FE) = {r['beta_noFE']:+.4f} (p={r['p_noFE']:.4g})  "
              f"β (+|lat|) = {r['beta_lat']:+.4f} (p={r['p_lat']:.4g})")

    # Country-by-country temperature comparison
    cmp = feats_old.rename(columns={"t_mean": "t_old", "sigma_s": "sigma_s_old",
                                       "sigma_v": "sigma_v_old"})\
                    .merge(feats_new.rename(columns={"t_mean": "t_new",
                                                        "sigma_s": "sigma_s_new",
                                                        "sigma_v": "sigma_v_new"}),
                            on="iso3")
    cmp["dt"] = cmp["t_new"] - cmp["t_old"]
    cmp["ds_s"] = cmp["sigma_s_new"] - cmp["sigma_s_old"]

    print("\n=== Country-level shift (pop-weighted minus area-weighted) ===")
    print(f"  ΔT median: {cmp['dt'].median():+.2f} °C, "
          f"  IQR: [{cmp['dt'].quantile(0.25):+.2f}, {cmp['dt'].quantile(0.75):+.2f}]")
    print(f"  Δσ_s median: {cmp['ds_s'].median():+.2f} °C, "
          f"  IQR: [{cmp['ds_s'].quantile(0.25):+.2f}, {cmp['ds_s'].quantile(0.75):+.2f}]")
    print(f"\n  Top 5 countries with biggest warming under pop-weight:")
    print(cmp.nlargest(5, 'dt')[["iso3", "t_old", "t_new", "dt"]].to_string(index=False))
    print(f"\n  Top 5 countries with biggest σ_s reduction under pop-weight:")
    print(cmp.nsmallest(5, 'ds_s')[["iso3", "sigma_s_old", "sigma_s_new", "ds_s"]].to_string(index=False))

    cmp.to_parquet(DATA / "popweight_country_comparison.parquet", index=False)


if __name__ == "__main__":
    main()
