"""Deep-determinants robustness for the long-shadow regression.

We add the standard controls used in cross-country deep-determinants
literature and test whether sigma_v -> modern pop growth survives.

Variables constructed (per country, from centroid + raster lookups):
  - dist_neolithic: km to nearest of seven Neolithic origin centers
  - tya_proxy: log(1 + dist_neolithic) as proxy for "years since exposure"
  - landlocked: 1 if all country centroid sub-units are >0 km from coast
  - abs_lat: absolute centroid latitude
  - log_area: ln(land area)

We also use distance-to-Neolithic as an alternative Stage 1 predictor and
test whether the Sigl volcanic response differs by Neolithic distance.

The seven primary Neolithic centers (lat, lon, name) follow Diamond (1997),
Putterman (2008), and the consensus archaeological dating.
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

# (lat, lon, name, approximate transition years BP)
NEOLITHIC_CENTERS = [
    (33.0,  44.0, "Fertile Crescent",       10500),
    (30.0, 113.0, "Yangtze China",            9000),
    (36.0, 110.0, "Yellow River China",       9000),
    (12.0,   5.0, "Sahel/West Africa",        5000),
    (18.0, -97.0, "Mesoamerica",              6000),
    (-15.0,-75.0, "Andes",                    4000),
    (37.0, -82.0, "Eastern North America",    4500),
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def _country_centroids() -> pd.DataFrame:
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    return ext[["iso3", "centroid_lat", "centroid_lon", "area_km2"]].drop_duplicates()


def build_deep_determinants() -> pd.DataFrame:
    cs = _country_centroids().dropna(subset=["centroid_lat", "centroid_lon"]).copy()
    dists = np.full((len(cs), len(NEOLITHIC_CENTERS)), np.nan)
    for j, (lat0, lon0, _, _) in enumerate(NEOLITHIC_CENTERS):
        dists[:, j] = haversine_km(cs["centroid_lat"].values,
                                     cs["centroid_lon"].values, lat0, lon0)
    cs["dist_neolithic"] = dists.min(axis=1)
    cs["nearest_center_idx"] = dists.argmin(axis=1)
    cs["log_dist_neolithic"] = np.log1p(cs["dist_neolithic"])
    cs["abs_lat"] = cs["centroid_lat"].abs()
    cs["log_area"] = np.log(cs["area_km2"].clip(lower=1))
    # Landlocked dummy from a hand-coded list of fully-landlocked countries
    landlocked = {
        "AFG","AND","ARM","AUT","AZE","BLR","BTN","BOL","BFA","BDI","CAF",
        "TCD","CZE","ETH","HUN","KAZ","KGZ","LAO","LSO","LIE","LUX","MWI",
        "MLI","MDA","MNG","NPL","NER","MKD","PRY","RWA","SMR","SRB","SVK",
        "SSS","SWZ","CHE","TJK","TKM","UGA","UZB","VAT","ZMB","ZWE","XKX",
        "ESH","SDS"
    }
    cs["landlocked"] = cs["iso3"].isin(landlocked).astype(int)
    return cs[["iso3", "abs_lat", "log_area", "landlocked",
                "dist_neolithic", "log_dist_neolithic"]]


def _outcomes_and_climate() -> pd.DataFrame:
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        pop_e=("pop", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        pop_l=("pop", "mean"))
    out = early.merge(late, on="iso3")
    out = out[(out["pop_e"] > 0) & (out["pop_l"] > 0)]
    out["log_pop_growth"] = np.log(out["pop_l"] / out["pop_e"])

    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)]
    feats = pre.groupby("iso3", as_index=False).agg(
        sigma_s_real=("sigma_s", "mean"),
        sigma_v_real=("t_mean", "std"),
        t_mean_preind=("t_mean", "mean"),
    )
    return out.merge(feats, on="iso3")


def long_shadow_battery(df: pd.DataFrame) -> None:
    print("\n=== Long-shadow with deep-determinants battery ===")
    df = df.dropna(subset=["sigma_v_real", "log_pop_growth", "abs_lat",
                            "log_dist_neolithic", "landlocked", "log_area"])
    print(f"N = {len(df)}")
    specs = [
        ("baseline",                   []),
        ("+ |lat|",                    ["abs_lat"]),
        ("+ |lat| + log_area",         ["abs_lat", "log_area"]),
        ("+ |lat| + landlocked",       ["abs_lat", "landlocked"]),
        ("+ |lat| + Neolithic dist",   ["abs_lat", "log_dist_neolithic"]),
        ("+ all deep determinants",    ["abs_lat", "log_area", "landlocked",
                                          "log_dist_neolithic", "t_mean_preind"]),
    ]
    print(f"{'specification':<32s} {'beta_σv':>10} {'SE':>8} {'p':>10}  R²    N")
    for label, ctrls in specs:
        X = sm.add_constant(df[["sigma_v_real"] + ctrls])
        y = df["log_pop_growth"]
        r = sm.OLS(y, X).fit(cov_type="HC1")
        print(f"{label:<32s} {r.params['sigma_v_real']:>+10.3f} "
              f"{r.bse['sigma_v_real']:>8.3f} {r.pvalues['sigma_v_real']:>10.4g}  "
              f"{r.rsquared:>5.3f} {int(r.nobs):>4}")


def neolithic_alt_stage1(df: pd.DataFrame) -> None:
    print("\n=== Stage 1: distance-to-Neolithic vs productive months ===")
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    # Build features needed for Stage 1
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    cd = mod.merge(clim, on=["iso3", "month"], how="inner")
    cd["t_abs"] = cd["t_anom_c"] + cd["tmp_c_clim"]
    cd["p_abs"] = (cd["p_anom_mm"] + cd["pre_mm_clim"]).clip(lower=0)
    cd["prod"] = ((cd["t_abs"]>=5)&(cd["t_abs"]<=30)&(cd["p_abs"]>=30)).astype(float)
    pre = cd[cd["year"].between(1421, 1750)]
    yr = pre.groupby(["iso3","year"], as_index=False)["prod"].sum().rename(
        columns={"prod": "prod_months"})
    feats = yr.groupby("iso3", as_index=False)["prod_months"].mean().rename(
        columns={"prod_months": "productive_months"})

    d = df.merge(feats, on="iso3").merge(clust[["iso3","cluster"]], on="iso3", how="inner")
    big = d[d["cluster"] != 2]
    big = big.dropna(subset=["productive_months", "log_dist_neolithic"])

    # ANOVA on log_dist_neolithic
    groups = [big.loc[big["cluster"] == k, "log_dist_neolithic"].values
              for k in sorted(big["cluster"].unique())]
    F, p = stats.f_oneway(*[g[~np.isnan(g)] for g in groups])
    print(f"  log_dist_neolithic:    ANOVA F = {F:.3f}, p = {p:.4g}")
    groups2 = [big.loc[big["cluster"] == k, "productive_months"].values
                for k in sorted(big["cluster"].unique())]
    F2, p2 = stats.f_oneway(*[g[~np.isnan(g)] for g in groups2])
    print(f"  productive_months:     ANOVA F = {F2:.3f}, p = {p2:.4g}  (paper headline)")
    print()
    # Joint MNLogit
    print("MNLogit pathway ~ log_dist_neolithic + productive_months")
    X = sm.add_constant(big[["log_dist_neolithic", "productive_months"]])
    res = sm.MNLogit(big["cluster"], X).fit(method="bfgs", maxiter=200, disp=False)
    print(f"  pseudo-R² = {res.prsquared:.4f}, LLR p = {res.llr_pvalue:.4g}")
    # Coefficients of productive_months in each row (against baseline)
    pw_names = {0:"Crop-dom",1:"Pastoral",3:"HighDens",4:"EarlyExt"}
    for col in res.params.columns:
        cluster_label = pw_names.get(int(col), str(col))
        print(f"  vs baseline → {cluster_label}: "
              f"log_dist_neolithic β = {res.params.loc['log_dist_neolithic', col]:+.3f} "
              f"(p={res.pvalues.loc['log_dist_neolithic', col]:.3g}), "
              f"productive_months β = {res.params.loc['productive_months', col]:+.3f} "
              f"(p={res.pvalues.loc['productive_months', col]:.3g})")


def sigl_neolithic_interaction(deep: pd.DataFrame) -> None:
    """Does volcanic resilience differ by Neolithic distance?"""
    print("\n=== Sigl volcanic forcing × Neolithic distance interaction ===")
    panel = pd.read_parquet(DATA / "sigl_volcanic_panel.parquet")
    df = panel.merge(deep[["iso3", "log_dist_neolithic"]], on="iso3", how="inner")
    df = df.dropna(subset=["pop_growth_ann", "vssi_int", "t_anom_int", "log_dist_neolithic"])
    # Within country demeaning
    g = df.groupby("iso3")
    for c in ["pop_growth_ann", "vssi_int", "t_anom_int"]:
        df[c] = df[c] - g[c].transform("mean")
    # Standardise Neolithic distance to make the interaction interpretable
    nd = (df["log_dist_neolithic"] - df["log_dist_neolithic"].mean()) / df["log_dist_neolithic"].std()
    df["vssi_x_neoldist"] = df["vssi_int"] * nd
    X = sm.add_constant(df[["vssi_int", "t_anom_int", "vssi_x_neoldist"]])
    r = sm.OLS(df["pop_growth_ann"], X).fit(cov_type="cluster", cov_kwds={"groups": df["iso3"]})
    print(f"N = {int(r.nobs)}, R² = {r.rsquared:.4f}")
    for v in ["vssi_int", "t_anom_int", "vssi_x_neoldist"]:
        print(f"  {v:<20s}: β = {r.params[v]:+.5e}  "
              f"SE = {r.bse[v]:.5e}  p = {r.pvalues[v]:.4g}")
    if r.pvalues["vssi_x_neoldist"] < 0.10:
        print("  → Significant interaction: volcanic response differs by Neolithic distance.")
    else:
        print("  → Insignificant interaction: volcanic response is NOT driven by deep-historical exposure.")


def main() -> None:
    print("Building deep-determinants table...")
    deep = build_deep_determinants()
    print(f"  Countries with deep-determinants data: {len(deep)}")
    print(f"  Distance to Neolithic centers (km): "
          f"median {deep['dist_neolithic'].median():.0f}, "
          f"max {deep['dist_neolithic'].max():.0f}")
    deep.to_parquet(DATA / "deep_determinants.parquet", index=False)

    outcomes = _outcomes_and_climate()
    df = outcomes.merge(deep, on="iso3")
    long_shadow_battery(df)
    neolithic_alt_stage1(df)
    sigl_neolithic_interaction(deep)


if __name__ == "__main__":
    main()
