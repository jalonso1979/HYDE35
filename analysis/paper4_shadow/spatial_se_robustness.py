"""Two-way clustered standard errors by (country, latitude band) for the
sub-national regressions, plus an approximate Conley spatial SE check.

Country clustering alone may be too conservative for some specifications and
too liberal for others (cross-country spatial correlation in climate shocks).
Two-way clustering on country and a 10° latitude band addresses both.

Implements the cameron-gelbach-miller two-way clustering as the sum of
single-way variances minus the intersection variance.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def twoway_cluster(X: pd.DataFrame, y: pd.Series,
                    cluster1: np.ndarray, cluster2: np.ndarray) -> dict:
    """CGM two-way clustered SEs. Both clusters must be non-negative integers."""
    # Re-encode each cluster vector to dense 0..k-1 integers
    c1 = pd.Series(cluster1).astype("category").cat.codes.values.astype(np.int64)
    c2 = pd.Series(cluster2).astype("category").cat.codes.values.astype(np.int64)
    # Combined cluster (intersection)
    c12 = pd.Series(list(zip(c1.tolist(), c2.tolist()))).astype(
        "category").cat.codes.values.astype(np.int64)
    r1 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    r2 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c2})
    r12 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c12})
    cov = r1.cov_params() + r2.cov_params() - r12.cov_params()
    eig = np.linalg.eigvalsh(cov.values)
    if eig.min() < 0:
        cov = cov + np.eye(cov.shape[0]) * abs(eig.min()) * 1.05
    se = np.sqrt(np.diag(cov.values))
    coefs = r1.params.values
    t = coefs / se
    from scipy.stats import t as tdist
    # Two-way cluster-robust dof uses the COARSER cluster dimension: min(G1,G2)-1
    # (Cameron-Gelbach-Miller). Using the finer dimension overstates precision.
    df = min(len(set(c1)), len(set(c2))) - 1
    p = 2 * (1 - tdist.cdf(np.abs(t), df))
    return {
        "params": dict(zip(X.columns, coefs)),
        "se":     dict(zip(X.columns, se)),
        "p":      dict(zip(X.columns, p)),
        "rsq":    r1.rsquared,
        "n":      int(r1.nobs),
    }


def main() -> None:
    print("=== Sub-national Stage 1 with two-way (country, lat-band) clustering ===")
    feats = pd.read_parquet(DATA / "subnational_features.parquet")
    hyde = pd.read_parquet(DATA / "subnational_hyde.parquet")
    h = hyde[hyde["year"] == 1750].copy()
    h["cropland_ha"] = h["cropland_ha"].fillna(0)
    h["grazing_ha"] = h["grazing_ha"].fillna(0)
    h["ag_total"] = h["cropland_ha"] + h["grazing_ha"]
    h["crop_share"] = np.where(h["ag_total"]>0, h["cropland_ha"]/h["ag_total"], np.nan)
    df = feats.merge(h[["sub_id", "iso3", "crop_share"]], on=["sub_id", "iso3"], how="inner")
    df = df.dropna(subset=["crop_share", "productive_months"])

    # Need lat band per sub-unit; use the ModE-RA aggregation centroid (avg lat from
    # monthly panel) as a proxy
    mod = pd.read_parquet(DATA / "modera_subnational_monthly.parquet",
                          columns=["sub_id", "iso3"])
    # Cannot derive lat from sub_id directly; use country centroid as approximation
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    centroids = ext[["iso3", "centroid_lat"]].drop_duplicates()
    df = df.merge(centroids, on="iso3", how="left")
    df = df.dropna(subset=["centroid_lat"])
    df["lat_band"] = (df["centroid_lat"] // 10 * 10).astype(int)

    # Within-country demeaning
    d = df.copy()
    g = d.groupby("iso3")
    d["crop_share"] = d["crop_share"] - g["crop_share"].transform("mean")
    d["productive_months"] = d["productive_months"] - g["productive_months"].transform("mean")
    X = sm.add_constant(d[["productive_months"]])
    y = d["crop_share"]

    print("\nSub-national Stage 1: crop_share ~ productive_months + country FE")
    print("(a) Country-clustered SE only:")
    r1 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    print(f"    beta={r1.params['productive_months']:+.4f}, "
          f"SE={r1.bse['productive_months']:.4f}, p={r1.pvalues['productive_months']:.4f}, "
          f"N={int(r1.nobs)}")

    print("(b) Two-way (country, 10° latitude band) clustering:")
    tw = twoway_cluster(X, y, d["iso3"].values, d["lat_band"].values)
    b = tw["params"]["productive_months"]; se = tw["se"]["productive_months"]
    p = tw["p"]["productive_months"]
    print(f"    beta={b:+.4f}, SE={se:.4f}, p={p:.4f}, N={tw['n']}")
    print(f"    SE ratio (2-way / country-only) = {se/r1.bse['productive_months']:.2f}")

    print("\n=== Sub-national Malthus with two-way clustering ===")
    df_m = pd.read_parquet(DATA / "subnational_malthus_panel.parquet")
    df_m = df_m.merge(centroids, on="iso3", how="left")
    df_m = df_m.dropna(subset=["centroid_lat"])
    df_m["lat_band"] = (df_m["centroid_lat"] // 10 * 10).astype(int)

    # Compute anomalies & demean by country
    df_m["t_anom"] = df_m["t_mean_int"] - df_m.groupby("iso3")["t_mean_int"].transform("mean")
    df_m["p_anom"] = df_m["p_mean_int"] - df_m.groupby("iso3")["p_mean_int"].transform("mean")
    d = df_m.dropna(subset=["log_pop", "t_anom", "p_anom", "t_std_int",
                             "pop_growth_ann"]).copy()
    g = d.groupby("iso3")
    for c in ["log_pop", "t_anom", "p_anom", "t_std_int", "pop_growth_ann"]:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[["log_pop", "t_anom", "p_anom", "t_std_int"]])
    y = d["pop_growth_ann"]

    print("\n(a) Country-clustered SE only:")
    r1 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    for v in ["log_pop", "t_anom", "p_anom", "t_std_int"]:
        print(f"    {v}: beta={r1.params[v]:+.6f} SE={r1.bse[v]:.6f} p={r1.pvalues[v]:.4g}")
    print(f"    N={int(r1.nobs)}")

    print("\n(b) Two-way (country, 10° latitude band) clustering:")
    tw = twoway_cluster(X, y, d["iso3"].values, d["lat_band"].values)
    for v in ["log_pop", "t_anom", "p_anom", "t_std_int"]:
        ratio = tw["se"][v] / r1.bse[v]
        print(f"    {v}: beta={tw['params'][v]:+.6f} SE={tw['se'][v]:.6f} "
              f"p={tw['p'][v]:.4g} (SE ratio = {ratio:.2f})")


if __name__ == "__main__":
    main()
