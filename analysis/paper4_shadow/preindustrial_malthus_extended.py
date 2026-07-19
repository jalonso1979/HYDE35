"""Extend the pre-industrial Malthus regression from 1750 to 1950 using
HYDE decadal data + ModE-RA annual climate. Brings Tambora 1815 and
Krakatoa 1883 inside the regression sample.

Uses HYDE timesteps in 1421-1950:
    1500, 1600, 1700, 1710, 1720, ..., 1750, 1760, 1770, ..., 1940, 1950
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}

START_YEAR = 1421
END_YEAR = 1950


def _load_hyde_intervals_extended() -> pd.DataFrame:
    """Load HYDE country population at all decadal timesteps 1421-1950
    by aggregating sub-national subpop_4apr2025.csv."""
    sub = pd.read_csv(ROOT / "gbc2025_7apr_base" / "subpop_4apr2025.csv")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    sub = sub.dropna(subset=["isolink"]).copy()
    sub["iso_num"] = (sub["isolink"].astype(int) // 1000).astype(int)
    sub["iso3"] = sub["iso_num"].map(num_to_iso3)
    sub = sub.dropna(subset=["iso3"]).copy()

    # Sum sub-national rows to country
    year_cols = [c for c in sub.columns if c.startswith("y")]
    agg = sub.groupby("iso3", as_index=False)[year_cols].sum(min_count=1)

    # Reshape to long, then to intervals
    long = agg.melt(id_vars="iso3", value_vars=year_cols,
                    var_name="year_col", value_name="pop")
    long["year"] = long["year_col"].str.lstrip("y").astype(int)
    long = long[(long["year"] >= START_YEAR) & (long["year"] <= END_YEAR)]
    long = long[long["pop"] > 0]
    long = long.sort_values(["iso3", "year"])

    # Country areas
    area_map = iso_map.set_index("iso3")["land_area_km2"].to_dict()
    long["area_km2"] = long["iso3"].map(area_map)
    long["density"] = long["pop"] / long["area_km2"]
    long["log_pop"] = np.log(long["pop"])
    long["log_density"] = np.log(long["density"].clip(lower=1e-9))

    # Build intervals (consecutive HYDE timesteps)
    long["year_next"] = long.groupby("iso3")["year"].shift(-1)
    long["log_pop_next"] = long.groupby("iso3")["log_pop"].shift(-1)
    long = long.dropna(subset=["year_next", "log_pop_next"])
    long["year_next"] = long["year_next"].astype(int)
    long["dt"] = long["year_next"] - long["year"]
    long["pop_growth_ann"] = (long["log_pop_next"] - long["log_pop"]) / long["dt"]
    return long[["iso3", "year", "year_next", "dt", "log_density", "pop_growth_ann"]]


def _attach_interval_climate(intervals: pd.DataFrame) -> pd.DataFrame:
    annual = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    annual = annual[(annual["year"] >= START_YEAR) & (annual["year"] <= END_YEAR)]
    annual = annual[["iso3", "year", "t_c", "p_mm"]]

    out = []
    annual_by_iso = {k: v for k, v in annual.groupby("iso3")}
    for _, row in intervals.iterrows():
        if row["iso3"] not in annual_by_iso:
            continue
        sub = annual_by_iso[row["iso3"]]
        sub = sub[(sub["year"] >= row["year"]) & (sub["year"] < row["year_next"])]
        if len(sub) < 3:
            continue
        out.append({
            "iso3": row["iso3"], "year": int(row["year"]),
            "year_next": int(row["year_next"]), "dt": int(row["dt"]),
            "log_density": row["log_density"],
            "pop_growth_ann": row["pop_growth_ann"],
            "t_mean_int": sub["t_c"].mean(),
            "t_std_int": sub["t_c"].std(),
            "p_mean_int": sub["p_mm"].mean(),
            "p_std_int": sub["p_mm"].std(),
            "n_yrs": len(sub),
        })
    return pd.DataFrame(out)


def _attach_pathway(df: pd.DataFrame) -> pd.DataFrame:
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)
    clust["cluster"] = clust["cluster"].astype(int)
    df = df.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)
    return df


def _fe(d: pd.DataFrame, regs: list[str]) -> dict:
    d = d.dropna(subset=regs + ["pop_growth_ann"]).copy()
    if len(d) < 30:
        return {}
    g = d.groupby("iso3")
    for c in regs + ["pop_growth_ann"]:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[regs])
    y = d["pop_growth_ann"]
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    return {"params": r.params.to_dict(), "pvalues": r.pvalues.to_dict(),
            "se": r.bse.to_dict(), "n": int(r.nobs), "rsq": r.rsquared}


def main() -> None:
    print("Loading HYDE 1421-1950 intervals...")
    iv = _load_hyde_intervals_extended()
    print(f"  {len(iv):,} (country, interval) cells, "
          f"{iv['iso3'].nunique()} countries, "
          f"{iv['year'].min()}-{iv['year'].max()}")

    df = _attach_interval_climate(iv)
    df = _attach_pathway(df)
    df["t_anom_int"] = df["t_mean_int"] - df.groupby("iso3")["t_mean_int"].transform("mean")
    df["p_anom_int"] = df["p_mean_int"] - df.groupby("iso3")["p_mean_int"].transform("mean")
    print(f"  After climate attach: {len(df):,} rows, {df['iso3'].nunique()} countries")
    print()

    df.to_parquet(DATA / "preindustrial_malthus_panel_extended.parquet", index=False)

    print("=== Pooled FE Malthusian regression 1421-1950 ===")
    full = _fe(df, ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"])
    print(f"N = {full['n']}, R^2 = {full['rsq']:.4f}")
    for v in ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"]:
        b = full["params"][v]; s = full["se"][v]; p = full["pvalues"][v]
        print(f"  {v:>15s}: beta = {b:+.6f}  se = {s:.6f}  p = {p:.4g}")

    print("\n=== Pathway-stratified FE Malthus + climate, 1421-1950 ===")
    rows = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        if len(sub) < 30:
            continue
        r = _fe(sub, ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"])
        if not r: continue
        rows.append({
            "pathway": PATHWAY_NAMES[cl], "n": r["n"],
            "beta_density": r["params"]["log_density"], "p_density": r["pvalues"]["log_density"],
            "beta_T": r["params"]["t_anom_int"], "p_T": r["pvalues"]["t_anom_int"],
            "beta_P": r["params"]["p_anom_int"], "p_P": r["pvalues"]["p_anom_int"],
            "beta_Tstd": r["params"]["t_std_int"], "p_Tstd": r["pvalues"]["t_std_int"],
            "rsq": r["rsq"],
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_parquet(DATA / "preindustrial_malthus_extended_results.parquet", index=False)

    # Subperiod breakdown
    print("\n=== Subperiod stability ===")
    for label, mask in [("1421-1750", df["year"] <= 1750),
                        ("1750-1900", df["year"].between(1750, 1899)),
                        ("1900-1950", df["year"] >= 1900)]:
        sub = df[mask]
        if len(sub) < 50:
            continue
        r = _fe(sub, ["log_density", "t_anom_int", "p_anom_int", "t_std_int"])
        if not r: continue
        print(f"  {label}: N={r['n']:>4}, "
              f"β_density={r['params']['log_density']:+.5f} (p={r['pvalues']['log_density']:.3g}), "
              f"β_T={r['params']['t_anom_int']:+.5f} (p={r['pvalues']['t_anom_int']:.3g}), "
              f"β_Tstd={r['params']['t_std_int']:+.5f} (p={r['pvalues']['t_std_int']:.3g})")


if __name__ == "__main__":
    main()
