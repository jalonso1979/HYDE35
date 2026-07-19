"""Compare area- vs pop- vs cropland-weighted ModE-RA aggregations.

The user-facing question: in a country like Egypt where most territory is
uninhabited desert, the aggregation should reflect where the actual
agricultural activity happened (Nile valley), not the empty quarter.
Pop-weighting partially addresses this; cropland+grazing-weighting addresses
it more directly. We compare all three on the country-level seasonality
features used in the headline regressions.
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

WEIGHTING_FILES = {
    "area":     "modera_country_monthly.parquet",
    "pop":      "modera_country_monthly_popw.parquet",
    "cropland": "modera_country_monthly_cropw.parquet",
}


def features_for(weighting: str) -> pd.DataFrame:
    mod = pd.read_parquet(DATA / WEIGHTING_FILES[weighting])
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    pre = df[df["year"].between(1421, 1750)].copy()
    pre["prod"] = ((pre["t_abs"]>=5)&(pre["t_abs"]<=30)&(pre["p_abs"]>=30)).astype(float)
    yr = pre.groupby(["iso3", "year"], as_index=False).agg(
        prod_months=("prod", "sum"),
        t_max=("t_abs", "max"),
        t_min=("t_abs", "min"),
        t_mean=("t_abs", "mean"),
        p_total=("p_abs", "sum"),
    )
    yr["sigma_s"] = yr["t_max"] - yr["t_min"]
    return yr.groupby("iso3", as_index=False).agg(
        productive_months=("prod_months", "mean"),
        sigma_s=("sigma_s", "mean"),
        sigma_v=("t_mean", "std"),
        t_mean=("t_mean", "mean"),
        p_annual=("p_total", "mean"),
    )


def main() -> None:
    print("Building features for three weightings...")
    feats = {w: features_for(w) for w in WEIGHTING_FILES}
    print(f"  Sample sizes: " +
          ", ".join(f"{w}={len(d)}" for w, d in feats.items()))

    # ── Comparison for key countries ────────────────────────────────────
    keys = ["EGY", "SAU", "DZA", "LBY", "RUS", "MNG", "CHN", "AUS",
            "USA", "BRA", "IND", "FRA", "GBR", "DEU"]
    print("\n=== Country-level climate under each weighting ===")
    print(f"{'iso3':<5} {'t_area':>7} {'t_pop':>7} {'t_crop':>7} "
          f"  {'σs_area':>7} {'σs_pop':>7} {'σs_crop':>7}  "
          f"{'prod_area':>9} {'prod_pop':>9} {'prod_crop':>9}")
    for iso in keys:
        rows = {w: f[f["iso3"]==iso] for w, f in feats.items()}
        if any(len(r)==0 for r in rows.values()): continue
        ta = rows["area"]["t_mean"].iloc[0]
        tp = rows["pop"]["t_mean"].iloc[0]
        tc = rows["cropland"]["t_mean"].iloc[0]
        sa = rows["area"]["sigma_s"].iloc[0]
        sp = rows["pop"]["sigma_s"].iloc[0]
        sc = rows["cropland"]["sigma_s"].iloc[0]
        pa = rows["area"]["productive_months"].iloc[0]
        pp = rows["pop"]["productive_months"].iloc[0]
        pc = rows["cropland"]["productive_months"].iloc[0]
        print(f"{iso:<5} {ta:>7.2f} {tp:>7.2f} {tc:>7.2f}   "
              f"{sa:>7.2f} {sp:>7.2f} {sc:>7.2f}  "
              f"{pa:>9.2f} {pp:>9.2f} {pc:>9.2f}")

    # ── Stage 1 ANOVA under each weighting ─────────────────────────────
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    print("\n=== Stage 1: productive_months ANOVA across pathways ===")
    print(f"{'weighting':<12} {'N':>4} {'F':>8} {'p':>10}  mean  sd")
    for w, f in feats.items():
        d = f.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
        big = d[d["cluster"] != 2]
        groups = [big.loc[big["cluster"]==k, "productive_months"].values
                  for k in sorted(big["cluster"].unique())]
        groups = [g[~np.isnan(g)] for g in groups]
        F, p = stats.f_oneway(*groups)
        print(f"{w:<12} {len(big):>4} {F:>8.3f} {p:>10.4g}  "
              f"{big['productive_months'].mean():>4.2f}  {big['productive_months'].std():>3.2f}")

    # ── Long-shadow σ_v → log pop growth under each weighting ──────────
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    cents = ext[["iso3", "centroid_lat"]].drop_duplicates()
    cents["abs_lat"] = cents["centroid_lat"].abs()
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        pop_e=("pop", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        pop_l=("pop", "mean"))
    outc = early.merge(late, on="iso3")
    outc = outc[(outc["pop_e"] > 0) & (outc["pop_l"] > 0)]
    outc["log_pop_growth"] = np.log(outc["pop_l"] / outc["pop_e"])

    print("\n=== Long-shadow: σ_v → log pop growth, +|lat| ===")
    print(f"{'weighting':<12} {'N':>4} {'beta':>8} {'SE':>8} {'p':>10}  R²")
    for w, f in feats.items():
        df = outc.merge(f[["iso3", "sigma_v"]], on="iso3").merge(
            cents[["iso3", "abs_lat"]], on="iso3").dropna()
        X = sm.add_constant(df[["sigma_v", "abs_lat"]])
        r = sm.OLS(df["log_pop_growth"], X).fit(cov_type="HC1")
        print(f"{w:<12} {int(r.nobs):>4} {r.params['sigma_v']:>+8.3f} "
              f"{r.bse['sigma_v']:>8.3f} {r.pvalues['sigma_v']:>10.4g}  "
              f"{r.rsquared:>4.3f}")


if __name__ == "__main__":
    main()
