"""Continuous-forcing volcanic regression using Sigl-Toohey eVolv2k v4.

Sigl & Toohey (2024) provide a comprehensive ice-core record of volcanic
stratospheric sulfur injection (VSSI, in Tg SO2) for every detected eruption
from 500 BCE to 1900 CE. We aggregate to annual forcing and run pathway-
stratified regressions of HYDE decadal population growth on cumulative VSSI
exposure in each decade-long interval, plus pathway interactions.

This replaces the discrete 5-event design (whose Tambora-alone placebo was
at the 65th percentile) with continuous variation across ~600 years of
eruption forcing.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def _parse_sigl() -> pd.DataFrame:
    """Parse the eVolv2k Sigl-Toohey TAB file into a clean DataFrame."""
    with open(DATA / "eVolv2k_sigl_toohey_2024.tab") as f:
        lines = f.read().splitlines()
    data_start = next(i + 1 for i, l in enumerate(lines) if l.startswith("*/"))
    rows = []
    n_cols = 13
    for line in lines[data_start + 1:]:
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < n_cols: continue
        rows.append({
            "year_AD":  parts[0],
            "lat":      parts[4],
            "vssi_Tg":  parts[7],
            "location": parts[10],
        })
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year_AD"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["vssi_Tg"] = pd.to_numeric(df["vssi_Tg"], errors="coerce")
    return df[["year", "lat", "vssi_Tg", "location"]].dropna(subset=["year","vssi_Tg"])


def _hyde_pop_intervals_extended() -> pd.DataFrame:
    """Country-level decadal HYDE pop intervals 1500-1900 (within Sigl coverage)."""
    sub = pd.read_csv(ROOT / "gbc2025_7apr_base" / "subpop_4apr2025.csv")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    sub = sub.dropna(subset=["isolink"]).copy()
    sub["iso_num"] = (sub["isolink"].astype(int) // 1000).astype(int)
    sub["iso3"] = sub["iso_num"].map(num_to_iso3)
    sub = sub.dropna(subset=["iso3"]).copy()
    ycols = [c for c in sub.columns if c.startswith("y")]
    agg = sub.groupby("iso3", as_index=False)[ycols].sum(min_count=1)
    long = agg.melt(id_vars="iso3", value_vars=ycols,
                    var_name="year_col", value_name="pop")
    long["year"] = long["year_col"].str.lstrip("y").astype(int)
    long = long[(long["year"] >= 1500) & (long["year"] <= 1900) & (long["pop"] > 0)]
    long = long.sort_values(["iso3", "year"]).reset_index(drop=True)
    long["log_pop"] = np.log(long["pop"])
    long["year_next"] = long.groupby("iso3")["year"].shift(-1)
    long["log_pop_next"] = long.groupby("iso3")["log_pop"].shift(-1)
    long = long.dropna(subset=["year_next", "log_pop_next"])
    long["year_next"] = long["year_next"].astype(int)
    long["dt"] = long["year_next"] - long["year"]
    long["pop_growth_ann"] = (long["log_pop_next"] - long["log_pop"]) / long["dt"]
    return long[["iso3", "year", "year_next", "dt", "log_pop", "pop_growth_ann"]]


def main() -> None:
    print("Parsing Sigl eVolv2k forcing record...")
    sigl = _parse_sigl()
    print(f"  {len(sigl)} eruptions, years {int(sigl['year'].min())} to {int(sigl['year'].max())}")
    print(f"  Largest 5 by VSSI:")
    top = sigl.nlargest(5, "vssi_Tg")
    for _, r in top.iterrows():
        print(f"    year={int(r['year']):>5}  lat={r['lat']:+.1f}  VSSI={r['vssi_Tg']:.1f} Tg")

    # Build year-level VSSI series 1421-1900
    sigl = sigl[(sigl["year"] >= 1421) & (sigl["year"] <= 1900)].copy()
    sigl["year"] = sigl["year"].astype(int)
    annual = sigl.groupby("year", as_index=False)["vssi_Tg"].sum()
    print(f"\nAnnual VSSI series: {len(annual)} years with eruptions out of "
          f"{1900 - 1421 + 1} possible")

    # HYDE intervals
    print("\nLoading HYDE intervals 1500-1900...")
    iv = _hyde_pop_intervals_extended()
    print(f"  {len(iv):,} (country, interval) cells")

    # Compute interval-summed VSSI for each (country, interval). Eruptions in
    # the interval [year, year_next) contribute their VSSI.
    iv["vssi_int"] = 0.0
    iv["max_vssi_int"] = 0.0
    iv["n_eruptions"] = 0
    for i, r in iv.iterrows():
        mask = (annual["year"] >= r["year"]) & (annual["year"] < r["year_next"])
        if mask.any():
            iv.at[i, "vssi_int"] = annual.loc[mask, "vssi_Tg"].sum()
            iv.at[i, "max_vssi_int"] = annual.loc[mask, "vssi_Tg"].max()
            iv.at[i, "n_eruptions"] = int(mask.sum())

    # Attach pathway
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    df = iv.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)

    # Attach interval climate
    annual_clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    annual_clim = annual_clim[["iso3", "year", "t_c"]]
    df["t_anom_int"] = 0.0
    for i, r in df.iterrows():
        sub = annual_clim[(annual_clim["iso3"] == r["iso3"])
                          & (annual_clim["year"] >= r["year"])
                          & (annual_clim["year"] < r["year_next"])]
        if len(sub) >= 3:
            df.at[i, "t_anom_int"] = sub["t_c"].mean()
    # demean within country
    df["t_anom_int"] = df["t_anom_int"] - df.groupby("iso3")["t_anom_int"].transform("mean")

    print(f"\nFinal panel: {len(df):,} cells, {df['iso3'].nunique()} countries, "
          f"{df['cluster'].nunique()} pathways")
    df.to_parquet(DATA / "sigl_volcanic_panel.parquet", index=False)

    # ── Continuous-forcing regression ─────────────────────────────────────
    print("\n=== Pooled FE regression: pop growth ~ VSSI_interval + pathway interactions ===")
    d = df.dropna(subset=["pop_growth_ann", "vssi_int", "t_anom_int", "cluster"]).copy()
    g = d.groupby("iso3")
    for c in ["pop_growth_ann", "vssi_int", "t_anom_int"]:
        d[c] = d[c] - g[c].transform("mean")
    pw_dum = pd.get_dummies(d["pathway"], prefix="pw", drop_first=True).astype(float)
    pw_int = pw_dum.multiply(d["vssi_int"].values, axis=0)
    pw_int.columns = [c + "_x_VSSI" for c in pw_int.columns]
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[["vssi_int", "t_anom_int"]], pw_dum, pw_int], axis=1).astype(float)
    y = d["pop_growth_ann"].astype(float)
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
    print(f"N = {int(r.nobs)}, R² = {r.rsquared:.4f}")
    for col in ["vssi_int"] + list(pw_int.columns):
        if col in r.params:
            print(f"  {col:>40s}: beta = {r.params[col]:+.7f}  "
                  f"SE = {r.bse[col]:.7f}  p = {r.pvalues[col]:.4g}")

    # By-pathway slopes
    print("\n=== Slope on VSSI_int by pathway (country FE) ===")
    rows = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl].dropna(
            subset=["pop_growth_ann", "vssi_int", "t_anom_int"])
        if len(sub) < 30: continue
        gg = sub.groupby("iso3")
        for c in ["pop_growth_ann", "vssi_int", "t_anom_int"]:
            sub[c] = sub[c] - gg[c].transform("mean")
        X2 = sm.add_constant(sub[["vssi_int", "t_anom_int"]])
        y2 = sub["pop_growth_ann"]
        rr = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups": sub["iso3"]})
        rows.append({
            "pathway": PATHWAY_NAMES[cl], "n": int(rr.nobs),
            "beta_vssi": rr.params["vssi_int"],
            "se_vssi": rr.bse["vssi_int"],
            "p_vssi": rr.pvalues["vssi_int"],
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_parquet(DATA / "sigl_volcanic_pathway_slopes.parquet", index=False)

    # Figure
    if len(res):
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        order = res.sort_values("beta_vssi").reset_index(drop=True)
        yy = np.arange(len(order))
        ax.errorbar(order["beta_vssi"], yy, xerr=1.96 * order["se_vssi"],
                    fmt="o", color="#202020", markerfacecolor="white",
                    markeredgewidth=1, ecolor="#404040", elinewidth=0.8, capsize=2)
        for i, row in order.iterrows():
            mark = ("***" if row["p_vssi"] < 0.01
                    else "**" if row["p_vssi"] < 0.05
                    else "*" if row["p_vssi"] < 0.10 else "")
            ax.text(row["beta_vssi"], i + 0.22,
                    f"$N={row['n']}$  {mark}", ha="center", fontsize=8.5)
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_yticks(yy); ax.set_yticklabels(order["pathway"])
        ax.set_xlabel(r"Slope: ann.\ pop growth per Tg of decadal VSSI exposure")
        ax.set_title(r"Sigl-Toohey continuous-forcing volcanic regression, 1500--1900",
                     loc="left")
        plt.tight_layout()
        fig.savefig(FIG / "fig12_sigl_volcanic.pdf")
        fig.savefig(FIG / "fig12_sigl_volcanic.png")
        plt.close(fig)
        print(f"\nSaved {FIG / 'fig12_sigl_volcanic.pdf'}")


if __name__ == "__main__":
    main()
