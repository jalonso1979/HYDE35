"""Harvest-shifted mask test for the long-shadow channel.

The monthly decomposition (long_shadow_monthly_decomp.py) shows the long-shadow
coefficient peaks in NH Sep-Dec (post-peak-warmth months). The hypothesis is
that volatility in the post-harvest storage window — not in the growing season
itself — drives the long shadow, a refinement of the Matranga (2024) storage-
and-intensification mechanism.

Test: for each country i, find the warmest month m_max from the 1421-1750
climatology. Define four candidate harvest-shifted masks:

  HS1 = {m_max+1, m_max+2, m_max+3}  immediate post-peak (Aug-Oct in NH temperate)
  HS2 = {m_max+2, m_max+3, m_max+4}  storage window     (Sep-Nov in NH temperate)
  HS3 = {m_max+1, ..., m_max+5}      broad post-harvest (Aug-Dec in NH temperate)
  HS4 = {m_max+3, m_max+4, m_max+5}  deep-storage      (Oct-Dec in NH temperate)

Compute σ_v^T_HS for each mask and run the headline regression alongside
annual σ_v^T and σ_v^T_GS for comparison.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PRE_WINDOW = (1421, 1750)


def _absolute_levels(mod: pd.DataFrame) -> pd.DataFrame:
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    return df


def build_warmest_month() -> pd.DataFrame:
    """Per-country warmest month (m_max) from 1421-1750 cropw climatology."""
    mod = pd.read_parquet(DATA / "modera_country_monthly_cropw.parquet")
    abs_df = _absolute_levels(mod)
    pre = abs_df[abs_df["year"].between(*PRE_WINDOW)]
    clim = pre.groupby(["iso3", "month"], as_index=False).agg(t_clim=("t_abs", "mean"))
    m_max = (clim.sort_values("t_clim", ascending=False)
                 .drop_duplicates("iso3")[["iso3", "month"]]
                 .rename(columns={"month": "m_max"}))
    return m_max


def _shift_mask(m_max: int, offsets: list[int]) -> list[int]:
    """Return months m_max+offset (1-indexed, wrapping mod 12)."""
    return [((m_max - 1 + o) % 12) + 1 for o in offsets]


def build_hs_sigma_v(mask_name: str, offsets: list[int]) -> pd.DataFrame:
    """One row per iso3 with σ_v^T over the harvest-shifted months."""
    mod = pd.read_parquet(DATA / "modera_country_monthly_cropw.parquet")
    abs_df = _absolute_levels(mod)
    pre = abs_df[abs_df["year"].between(*PRE_WINDOW)].copy()
    m_max = build_warmest_month()
    pre = pre.merge(m_max, on="iso3", how="inner")
    pre["hs_months"] = pre.apply(
        lambda r: _shift_mask(int(r["m_max"]), offsets), axis=1)
    # Filter to HS months
    pre["in_hs"] = pre.apply(lambda r: r["month"] in r["hs_months"], axis=1)
    hs = pre[pre["in_hs"]]
    # Per (iso3, year) mean over HS months
    hs_yr = hs.groupby(["iso3", "year"], as_index=False).agg(t_hs=("t_abs", "mean"))
    # std across years
    out = hs_yr.groupby("iso3", as_index=False).agg(
        **{f"sigma_v_T_{mask_name}": ("t_hs", "std")})
    return out


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
              .merge(cov[["iso3", "abs_lat"]], on="iso3", how="inner")
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
    return {"key": key, "beta": float(r.params[key]), "se": float(r.bse[key]),
            "p": float(r.pvalues[key]), "r2": float(r.rsquared),
            "n": int(r.nobs)}


def run() -> pd.DataFrame:
    print("[harvest-mask] building 4 harvest-shifted masks...", flush=True)
    masks = {
        "HS1_immediate":   [1, 2, 3],   # m_max+1..+3
        "HS2_storage":     [2, 3, 4],   # m_max+2..+4
        "HS3_broad":       [1, 2, 3, 4, 5],  # m_max+1..+5
        "HS4_deepstorage": [3, 4, 5],   # m_max+3..+5
    }
    pieces = []
    for name, offsets in masks.items():
        print(f"  {name}: offsets {offsets}", flush=True)
        pieces.append(build_hs_sigma_v(name, offsets))
    sv = pieces[0]
    for p in pieces[1:]:
        sv = sv.merge(p, on="iso3", how="outer")

    # Also load comparators: annual σ_v^T and GS σ_v^T
    annual = pd.read_parquet(DATA / "country_seasonality_preindustrial.parquet")
    gs = pd.read_parquet(DATA / "country_seasonality_gs_preindustrial.parquet")
    sv = sv.merge(annual[["iso3", "sigma_v_preind"]], on="iso3", how="outer")
    sv = sv.merge(gs[["iso3", "sigma_v_T_gs_pre1750_cropw",
                      "sigma_v_T_nongs_pre1750_cropw", "n_gs_months_cropw"]],
                  on="iso3", how="outer")

    base = _outcome_and_covariates()
    df = base.merge(sv, on="iso3", how="left")

    keys = ["sigma_v_preind",
            "sigma_v_T_gs_pre1750_cropw",
            "sigma_v_T_nongs_pre1750_cropw"] + [f"sigma_v_T_{k}" for k in masks]
    print("\n=== Long-shadow coefficient by mask (full sample, +|lat| +pathway FE, HC1) ===")
    rows = []
    for k in keys:
        r = _fit(df, k)
        rows.append(r)
        print(f"  {k:<40s}  β={r['beta']:+.3f}  SE={r['se']:.3f}  "
              f"p={r['p']:.4f}  R²={r['r2']:.3f}  N={r['n']}")
    res = pd.DataFrame(rows)

    # Restrict to empty-GS-dropped sample (matches headline)
    headline = df[df["n_gs_months_cropw"] > 0].copy()
    print(f"\n=== Same battery on empty-GS-dropped headline sample (N={len(headline)}) ===")
    rows2 = []
    for k in keys:
        r = _fit(headline, k)
        rows2.append(r)
        print(f"  {k:<40s}  β={r['beta']:+.3f}  SE={r['se']:.3f}  "
              f"p={r['p']:.4f}  R²={r['r2']:.3f}  N={r['n']}")
    res2 = pd.DataFrame(rows2)
    res2["sample"] = "headline"
    res["sample"] = "full"
    final = pd.concat([res, res2], ignore_index=True)

    # Joint test: HS2_storage + GS together
    print("\n=== Joint spec: σ_v^T_HS2_storage + σ_v^T_GS (headline sample) ===")
    sub = headline[["sigma_v_T_HS2_storage", "sigma_v_T_gs_pre1750_cropw",
                    "log_pop_growth", "abs_lat", "cluster"]].dropna()
    dums = pd.get_dummies(sub["cluster"], prefix="pw", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([sub[["sigma_v_T_HS2_storage",
                                         "sigma_v_T_gs_pre1750_cropw",
                                         "abs_lat"]], dums], axis=1).astype(float))
    r = sm.OLS(sub["log_pop_growth"], X).fit(cov_type="HC1")
    print(f"  sigma_v_T_HS2_storage:    β={r.params['sigma_v_T_HS2_storage']:+.3f}  "
          f"p={r.pvalues['sigma_v_T_HS2_storage']:.4f}")
    print(f"  sigma_v_T_gs_pre1750:     β={r.params['sigma_v_T_gs_pre1750_cropw']:+.3f}  "
          f"p={r.pvalues['sigma_v_T_gs_pre1750_cropw']:.4f}")
    print(f"  R²={r.rsquared:.3f}  N={int(r.nobs)}")

    # Joint test: HS2_storage + non-GS together (does the autumn signal subsume the non-GS signal?)
    print("\n=== Joint spec: σ_v^T_HS2_storage + σ_v^T_nonGS (headline sample) ===")
    sub2 = headline[["sigma_v_T_HS2_storage", "sigma_v_T_nongs_pre1750_cropw",
                     "log_pop_growth", "abs_lat", "cluster"]].dropna()
    dums = pd.get_dummies(sub2["cluster"], prefix="pw", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([sub2[["sigma_v_T_HS2_storage",
                                          "sigma_v_T_nongs_pre1750_cropw",
                                          "abs_lat"]], dums], axis=1).astype(float))
    r = sm.OLS(sub2["log_pop_growth"], X).fit(cov_type="HC1")
    print(f"  sigma_v_T_HS2_storage:    β={r.params['sigma_v_T_HS2_storage']:+.3f}  "
          f"p={r.pvalues['sigma_v_T_HS2_storage']:.4f}")
    print(f"  sigma_v_T_nongs:          β={r.params['sigma_v_T_nongs_pre1750_cropw']:+.3f}  "
          f"p={r.pvalues['sigma_v_T_nongs_pre1750_cropw']:.4f}")
    print(f"  R²={r.rsquared:.3f}  N={int(r.nobs)}")

    out = DATA / "long_shadow_harvest_mask_results.parquet"
    final.to_parquet(out, index=False)
    print(f"\n[harvest-mask] wrote {out}")
    return final


if __name__ == "__main__":
    run()
