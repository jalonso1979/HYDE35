"""Sub-national Malthusian regression 1500-1950.

Repeats the country-level pre-industrial Malthus regression with annual climate
forcing at the sub-national level. With 3,000+ sub-units and 17+ HYDE decadal
timesteps, we expect ~50x the cells of the country-level analysis.

Identification: country fixed effects absorb country-level confounders;
within-country variation across sub-units identifies pathway / climate effects.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
START, END = 1500, 1950


def _annual_subnat_climate() -> pd.DataFrame:
    """Sub-national absolute annual climate from ModE-RA + CRU climatology."""
    mod = pd.read_parquet(DATA / "modera_subnational_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    df = df[df["year"].between(START, END)]
    annual = df.groupby(["sub_id", "iso3", "year"], as_index=False).agg(
        t_c=("t_abs", "mean"), p_mm=("p_abs", "sum"),
    )
    return annual


def _hyde_pop_intervals() -> pd.DataFrame:
    """Sub-national HYDE timesteps for the period, yielding interval pairs."""
    hyde = pd.read_parquet(DATA / "subnational_hyde.parquet")
    h = hyde[hyde["year"].between(START, END)][["sub_id", "iso3", "year", "subpop"]].copy()
    h = h[h["subpop"] > 0]
    h["log_pop"] = np.log(h["subpop"])
    h = h.sort_values(["sub_id", "year"]).reset_index(drop=True)
    h["year_next"] = h.groupby("sub_id")["year"].shift(-1)
    h["log_pop_next"] = h.groupby("sub_id")["log_pop"].shift(-1)
    h = h.dropna(subset=["year_next", "log_pop_next"])
    h["year_next"] = h["year_next"].astype(int)
    h["dt"] = h["year_next"] - h["year"]
    h["pop_growth_ann"] = (h["log_pop_next"] - h["log_pop"]) / h["dt"]
    return h[["sub_id", "iso3", "year", "year_next", "dt",
              "log_pop", "pop_growth_ann"]]


def _attach_interval_climate(intervals: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Assign each annual (sub_id, year) row to its parent interval, then
    aggregate climate moments per (sub_id, interval_start)."""
    annual = annual.copy()
    annual["year"] = annual["year"].astype(np.int64)
    iv = intervals[["sub_id", "iso3", "year", "year_next", "dt",
                    "log_pop", "pop_growth_ann"]].copy()
    iv["year"] = iv["year"].astype(np.int64)
    iv["year_next"] = iv["year_next"].astype(np.int64)
    iv = iv.rename(columns={"year": "iv_start"})

    annual = annual.sort_values(["year", "sub_id"]).reset_index(drop=True)
    iv_sorted = iv.sort_values(["iv_start", "sub_id"]).reset_index(drop=True)

    merged = pd.merge_asof(
        annual, iv_sorted,
        left_on="year", right_on="iv_start",
        by="sub_id", direction="backward",
        suffixes=("", "_iv"),
    )
    merged = merged.dropna(subset=["iv_start", "year_next"])
    merged = merged[merged["year"] < merged["year_next"]]

    grouped = merged.groupby(["sub_id", "iso3", "iv_start", "year_next",
                                "dt", "log_pop", "pop_growth_ann"],
                              as_index=False).agg(
        t_mean_int=("t_c", "mean"), t_std_int=("t_c", "std"),
        p_mean_int=("p_mm", "mean"), p_std_int=("p_mm", "std"),
        n_yrs=("year", "count"),
    )
    grouped = grouped[grouped["n_yrs"] >= 3]
    grouped = grouped.rename(columns={"iv_start": "year"})
    return grouped


def _within_country_fe(d: pd.DataFrame, regs: list[str],
                        also_sub_fe: bool = False) -> dict:
    """Demean by country (and optionally sub-unit) before OLS with cluster SEs."""
    d = d.dropna(subset=regs + ["pop_growth_ann"]).copy()
    if len(d) < 50:
        return {}
    # Always demean by country
    g_c = d.groupby("iso3")
    for c in regs + ["pop_growth_ann"]:
        d[c] = d[c] - g_c[c].transform("mean")
    if also_sub_fe:
        g_s = d.groupby("sub_id")
        for c in regs + ["pop_growth_ann"]:
            d[c] = d[c] - g_s[c].transform("mean")
    X = sm.add_constant(d[regs])
    y = d["pop_growth_ann"]
    r = sm.OLS(y, X).fit(cov_type="cluster",
                          cov_kwds={"groups": d["iso3"].values})
    return {"params": r.params.to_dict(), "pvalues": r.pvalues.to_dict(),
            "se": r.bse.to_dict(), "n": int(r.nobs), "rsq": r.rsquared}


def main() -> None:
    print("Loading sub-national annual climate...", flush=True)
    annual = _annual_subnat_climate()
    print(f"  {len(annual):,} (sub_id, year) cells, "
          f"{annual['sub_id'].nunique()} sub-units")

    print("Loading sub-national HYDE intervals...", flush=True)
    iv = _hyde_pop_intervals()
    print(f"  {len(iv):,} (sub_id, interval) cells, "
          f"{iv['sub_id'].nunique()} sub-units")

    print("Attaching interval climate...", flush=True)
    df = _attach_interval_climate(iv, annual)
    print(f"  Final panel: {len(df):,} cells, "
          f"{df['sub_id'].nunique()} sub-units, "
          f"{df['iso3'].nunique()} countries", flush=True)
    df.to_parquet(DATA / "subnational_malthus_panel.parquet", index=False)

    # Density: pop / area. We don't have sub-unit area directly; proxy by
    # log_pop as the LHS-of-density variable and use log_pop level as control.
    # For now, regress pop_growth on log_pop (Malthusian density), t, p, sigma_v.
    df["t_anom"] = df["t_mean_int"] - df.groupby(["sub_id"])["t_mean_int"].transform("mean")
    df["p_anom"] = df["p_mean_int"] - df.groupby(["sub_id"])["p_mean_int"].transform("mean")

    # Country FE (within-country variation across sub-units + time)
    print("\n=== Sub-national Malthus: pooled, country FE ===")
    r = _within_country_fe(
        df, ["log_pop", "t_anom", "p_anom", "t_std_int", "p_std_int"], also_sub_fe=False)
    print(f"N = {r['n']:,}, R^2 = {r['rsq']:.4f}")
    for v in ["log_pop", "t_anom", "p_anom", "t_std_int", "p_std_int"]:
        b = r["params"][v]; s = r["se"][v]; p = r["pvalues"][v]
        print(f"  {v:>15s}: beta = {b:+.7f}  se = {s:.7f}  p = {p:.4g}")

    print("\n=== Sub-national Malthus: pooled, sub-unit FE ===")
    r = _within_country_fe(
        df, ["log_pop", "t_anom", "p_anom", "t_std_int", "p_std_int"], also_sub_fe=True)
    print(f"N = {r['n']:,}, R^2 = {r['rsq']:.4f}")
    for v in ["log_pop", "t_anom", "p_anom", "t_std_int", "p_std_int"]:
        b = r["params"][v]; s = r["se"][v]; p = r["pvalues"][v]
        print(f"  {v:>15s}: beta = {b:+.7f}  se = {s:.7f}  p = {p:.4g}")

    print("\n=== Subperiod stability ===")
    for label, mask in [("1500-1750", df["year"] < 1750),
                        ("1750-1900", df["year"].between(1750, 1899)),
                        ("1900-1950", df["year"] >= 1900)]:
        sub = df[mask]
        if len(sub) < 100: continue
        r = _within_country_fe(sub, ["log_pop", "t_anom", "p_anom", "t_std_int"])
        if not r: continue
        print(f"  {label}: N={r['n']:>6,}, "
              f"β_pop={r['params']['log_pop']:+.6f} (p={r['pvalues']['log_pop']:.3g}), "
              f"γ_T={r['params']['t_anom']:+.6f} (p={r['pvalues']['t_anom']:.3g}), "
              f"δ_T={r['params']['t_std_int']:+.6f} (p={r['pvalues']['t_std_int']:.3g})")


if __name__ == "__main__":
    main()
