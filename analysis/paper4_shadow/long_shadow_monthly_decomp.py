"""Monthly decomposition of σ_v^T in the long-shadow cross-section.

For each calendar month m, compute σ_v^{T,m}_i = std over 1421-1750 of
country-mean T in month m, then regress modern log pop growth on each
σ_v^{T,m} separately (with |lat| + pathway FE, HC1 SEs). The pattern
across months identifies which calendar months carry the long-shadow
signal:
  - winter loading → cold-mortality channel
  - shoulder-season loading → planting/harvest failure
  - broadly distributed → "genuinely climate-deep"

Spec: extension of docs/superpowers/specs/2026-05-19-growing-season-volatility-design.md
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures"

PRE_WINDOW = (1421, 1750)
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _absolute_levels(mod: pd.DataFrame) -> pd.DataFrame:
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    return df


def build_monthly_sigma_v() -> pd.DataFrame:
    """One row per (iso3, month) with std of yearly mean T in that month,
    1421-1750, on the cropland-weighted ModE-RA panel.

    Returns DataFrame wide: iso3, sigma_v_T_m01, ..., sigma_v_T_m12.
    """
    mod = pd.read_parquet(DATA / "modera_country_monthly_cropw.parquet")
    abs_df = _absolute_levels(mod)
    pre = abs_df[abs_df["year"].between(*PRE_WINDOW)].copy()
    # std across years of (iso3, month) — each year contributes one observation
    g = pre.groupby(["iso3", "month"], as_index=False).agg(
        sigma_v_T_m=("t_abs", "std"))
    # Pivot to wide
    wide = g.pivot(index="iso3", columns="month", values="sigma_v_T_m")
    wide.columns = [f"sigma_v_T_m{int(m):02d}" for m in wide.columns]
    return wide.reset_index()


def _outcome_and_covariates() -> pd.DataFrame:
    modern = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    p0 = (modern[modern["year"].between(1950, 1960)]
            .groupby("iso3", as_index=False).agg(p0=("pop", "mean")))
    p1 = (modern[modern["year"].between(2015, 2025)]
            .groupby("iso3", as_index=False).agg(p1=("pop", "mean")))
    out = p0.merge(p1, on="iso3").query("p0 > 0 and p1 > 0").copy()
    out["log_pop_growth"] = np.log(out["p1"] / out["p0"])
    cov = (modern.dropna(subset=["centroid_lat"])
                 .groupby("iso3", as_index=False)
                 .agg(centroid_lat=("centroid_lat", "first")))
    cov["abs_lat"] = cov["centroid_lat"].abs()
    pathways = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    return (out[["iso3", "log_pop_growth"]]
              .merge(cov[["iso3", "abs_lat", "centroid_lat"]], on="iso3", how="inner")
              .merge(pathways[["iso3", "cluster"]], on="iso3", how="left"))


def _fit(d: pd.DataFrame, key: str) -> dict:
    sub = d[[key, "log_pop_growth", "abs_lat", "cluster"]].dropna()
    if len(sub) < 5:
        return {"beta": np.nan, "se": np.nan, "p": np.nan, "r2": np.nan, "n": 0}
    dums = pd.get_dummies(sub["cluster"], prefix="pw", drop_first=True, dtype=float)
    X_cols = [key, "abs_lat"] + list(dums.columns)
    sub = pd.concat([sub, dums], axis=1)
    X = sm.add_constant(sub[X_cols].astype(float))
    r = sm.OLS(sub["log_pop_growth"], X).fit(cov_type="HC1")
    return {"beta": float(r.params[key]), "se": float(r.bse[key]),
            "p": float(r.pvalues[key]), "r2": float(r.rsquared),
            "n": int(r.nobs)}


def run() -> pd.DataFrame:
    print("[monthly-decomp] building per-month σ_v^T (1421-1750)...", flush=True)
    sv = build_monthly_sigma_v()
    base = _outcome_and_covariates()
    df = base.merge(sv, on="iso3", how="left")

    rows = []
    for m in range(1, 13):
        key = f"sigma_v_T_m{m:02d}"
        r = _fit(df, key)
        r["month"] = m
        r["month_name"] = MONTH_NAMES[m - 1]
        rows.append(r)
    res = pd.DataFrame(rows)[["month", "month_name", "beta", "se", "p", "r2", "n"]]

    print("\n=== Monthly σ_v^T → log pop growth, 1950→2025, +|lat| +pathway FE ===")
    print(res.to_string(index=False))

    # Hemisphere split: do NH and SH show different patterns?
    print("\n--- Northern Hemisphere only (centroid_lat > 0) ---")
    nh = df[df["centroid_lat"] > 0]
    rows_nh = []
    for m in range(1, 13):
        key = f"sigma_v_T_m{m:02d}"
        r = _fit(nh, key); r["month"] = m
        rows_nh.append(r)
    res_nh = pd.DataFrame(rows_nh)[["month", "beta", "se", "p", "n"]]
    print(res_nh.to_string(index=False))

    print("\n--- Southern Hemisphere only (centroid_lat < 0) ---")
    sh = df[df["centroid_lat"] < 0]
    rows_sh = []
    for m in range(1, 13):
        key = f"sigma_v_T_m{m:02d}"
        r = _fit(sh, key); r["month"] = m
        rows_sh.append(r)
    res_sh = pd.DataFrame(rows_sh)[["month", "beta", "se", "p", "n"]]
    print(res_sh.to_string(index=False))

    out_pq = DATA / "long_shadow_monthly_sigma_v_results.parquet"
    res.to_parquet(out_pq, index=False)
    print(f"\n[monthly-decomp] wrote {out_pq}")

    # Make a small figure showing β by month
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(res["month"], res["beta"],
            yerr=1.96 * res["se"], capsize=3, color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel(r"$\hat\beta(\sigma_v^{T,m})$ on log pop growth")
    ax.set_title("Long-shadow coefficient on monthly $\\sigma_v^T$ (1421–1750), "
                  "+|lat| +pathway FE, HC1 95% CI")
    plt.tight_layout()
    fig_out = FIG / "long_shadow_monthly_sigma_v.png"
    plt.savefig(fig_out, dpi=150)
    plt.close()
    print(f"[monthly-decomp] wrote {fig_out}")
    return res


if __name__ == "__main__":
    run()
