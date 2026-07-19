"""Volcanic-forcing extension of the Sigl event study to 100--1500 CE.

The main-text Sigl exercise (`sigl_volcanic_forcing.py`) covers 1500--1900 because
that is where HYDE 3.5 has decadal pop estimates *and* where ModE-RA delivers
country-level annual climate.  Pre-1500 HYDE intervals are at century resolution
(y100, y200, ..., y1500), and ModE-RA does not extend below 1421.  We adapt the
design accordingly:

  - Outcome: annualised log-population growth across HYDE century intervals
    (y100→y200, …, y1400→y1500).  HYDE coverage extends to 10 kBCE, but the
    Sigl-Toohey eVolv2k record only starts at 500 BCE, and the country/iso3
    polygon match becomes increasingly tenuous before the Common Era — so we
    restrict to 100--1500 CE.
  - Treatment: century-summed stratospheric sulphur injection (VSSI, Tg) from
    eVolv2k v4.
  - Identification: country fixed effects on pop-growth and on VSSI exposure;
    pathway-by-VSSI interactions test the same prediction as the 1500--1900
    exercise (intensive systems are less harmed).

This is a coarser test than the decadal one — fourteen 100-year cells vs.
forty 10-year cells — but it covers an entirely separate slice of the historical
record, including the Samalas 1257 eruption (the largest of the past 2500 years)
and the 536--540 CE LALIA doublet that coincided with the Plague of Justinian.

Outputs:
    analysis/data/sigl_volcanic_panel_pre1500.parquet
    analysis/data/sigl_volcanic_pathway_slopes_pre1500.parquet
    analysis/figures/paper4_v2/figA1_sigl_pre1500.pdf / .png
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}

START_YEAR, END_YEAR = 100, 1500  # CE


def _parse_sigl() -> pd.DataFrame:
    with open(DATA / "eVolv2k_sigl_toohey_2024.tab") as f:
        lines = f.read().splitlines()
    data_start = next(i + 1 for i, l in enumerate(lines) if l.startswith("*/"))
    rows = []
    for line in lines[data_start + 1:]:
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < 13: continue
        rows.append({"year": parts[0], "lat": parts[4],
                     "vssi": parts[7], "loc": parts[10]})
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vssi"] = pd.to_numeric(df["vssi"], errors="coerce")
    return df.dropna(subset=["year", "vssi"])[["year", "vssi", "loc"]]


def _hyde_century_intervals() -> pd.DataFrame:
    """HYDE century-resolution pop intervals y100..y1500.

    Returns one row per (iso3, century_start) with pop_growth_ann to the next
    available century column.
    """
    sub = pd.read_csv(ROOT / "gbc2025_7apr_base" / "subpop_4apr2025.csv")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    sub = sub.dropna(subset=["isolink"]).copy()
    sub["iso_num"] = (sub["isolink"].astype(int) // 1000).astype(int)
    sub["iso3"] = sub["iso_num"].map(num_to_iso3)
    sub = sub.dropna(subset=["iso3"]).copy()

    century_cols = [f"y{y}" for y in range(START_YEAR, END_YEAR + 100, 100)]
    available = [c for c in century_cols if c in sub.columns]
    agg = sub.groupby("iso3", as_index=False)[available].sum(min_count=1)
    long = agg.melt(id_vars="iso3", value_vars=available,
                    var_name="ycol", value_name="pop")
    long["year"] = long["ycol"].str.lstrip("y").astype(int)
    long = long[long["pop"] > 0].sort_values(["iso3", "year"]).reset_index(drop=True)
    long["log_pop"] = np.log(long["pop"])
    long["year_next"] = long.groupby("iso3")["year"].shift(-1)
    long["log_pop_next"] = long.groupby("iso3")["log_pop"].shift(-1)
    long = long.dropna(subset=["year_next", "log_pop_next"])
    long["year_next"] = long["year_next"].astype(int)
    long["dt"] = long["year_next"] - long["year"]
    long["pop_growth_ann"] = (long["log_pop_next"] - long["log_pop"]) / long["dt"]
    return long[["iso3", "year", "year_next", "dt", "log_pop", "pop_growth_ann"]]


def main() -> None:
    print(f"=== Pre-1500 Sigl volcanic extension: {START_YEAR}–{END_YEAR} CE ===")
    sigl = _parse_sigl()
    annual = (sigl[(sigl["year"] >= START_YEAR) & (sigl["year"] < END_YEAR + 100)]
              .assign(year=lambda d: d["year"].astype(int))
              .groupby("year", as_index=False)["vssi"].sum())
    print(f"  eVolv2k years {START_YEAR}-{END_YEAR}: "
          f"{len(annual)} years with detected eruptions")

    iv = _hyde_century_intervals()
    print(f"  HYDE century intervals: {len(iv):,} (country, interval) cells "
          f"({iv['iso3'].nunique()} countries)")

    # Century-summed VSSI for each (country, interval)
    iv["vssi_int"] = 0.0
    iv["max_vssi_int"] = 0.0
    iv["n_eruptions"] = 0
    for i, r in iv.iterrows():
        mask = (annual["year"] >= r["year"]) & (annual["year"] < r["year_next"])
        if mask.any():
            iv.at[i, "vssi_int"] = annual.loc[mask, "vssi"].sum()
            iv.at[i, "max_vssi_int"] = annual.loc[mask, "vssi"].max()
            iv.at[i, "n_eruptions"] = int(mask.sum())

    print("\n  Century-summed VSSI distribution (top 8):")
    print(iv.nlargest(8, "vssi_int")[
        ["iso3", "year", "year_next", "vssi_int", "n_eruptions"]].to_string(index=False))

    # Attach pathway
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    df = iv.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)
    df.to_parquet(DATA / "sigl_volcanic_panel_pre1500.parquet", index=False)
    print(f"\n  Panel after pathway merge: {len(df):,} cells, "
          f"{df['iso3'].nunique()} countries, {df['cluster'].nunique()} pathways")

    # ── Pooled regression with country FE via within-transformation ──────────
    print("\n=== Pooled FE regression: pop growth ~ century VSSI × pathway ===")
    d = df.dropna(subset=["pop_growth_ann", "vssi_int", "cluster"]).copy()
    g = d.groupby("iso3")
    for c in ["pop_growth_ann", "vssi_int"]:
        d[c] = d[c] - g[c].transform("mean")
    pw_dum = pd.get_dummies(d["pathway"], prefix="pw", drop_first=True).astype(float)
    pw_int = pw_dum.multiply(d["vssi_int"].values, axis=0)
    pw_int.columns = [c + "_x_VSSI" for c in pw_int.columns]
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[["vssi_int"]], pw_dum, pw_int], axis=1).astype(float)
    y = d["pop_growth_ann"].astype(float)
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
    print(f"N = {int(r.nobs)}, R² = {r.rsquared:.4f}")
    for col in ["vssi_int"] + list(pw_int.columns):
        if col in r.params:
            print(f"  {col:>40s}: beta = {r.params[col]:+.7f}  "
                  f"SE = {r.bse[col]:.7f}  p = {r.pvalues[col]:.4g}")

    # ── By-pathway slopes ────────────────────────────────────────────────────
    print("\n=== Slope on century VSSI by pathway (country FE) ===")
    rows = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl].dropna(subset=["pop_growth_ann", "vssi_int"]).copy()
        if len(sub) < 15: continue
        gg = sub.groupby("iso3")
        for c in ["pop_growth_ann", "vssi_int"]:
            sub[c] = sub[c] - gg[c].transform("mean")
        X2 = sm.add_constant(sub[["vssi_int"]])
        rr = sm.OLS(sub["pop_growth_ann"], X2).fit(
            cov_type="cluster", cov_kwds={"groups": sub["iso3"]})
        rows.append({"pathway": PATHWAY_NAMES[cl], "n": int(rr.nobs),
                     "beta_vssi": rr.params["vssi_int"],
                     "se_vssi": rr.bse["vssi_int"],
                     "p_vssi": rr.pvalues["vssi_int"]})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_parquet(DATA / "sigl_volcanic_pathway_slopes_pre1500.parquet", index=False)

    # ── Figure ───────────────────────────────────────────────────────────────
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
        ax.set_xlabel(r"Slope: ann.\ pop growth per Tg of century-summed VSSI")
        ax.set_title(rf"Sigl-Toohey continuous-forcing volcanic regression, {START_YEAR}--{END_YEAR}~CE",
                     loc="left")
        plt.tight_layout()
        fig.savefig(FIG / "figA1_sigl_pre1500.pdf")
        fig.savefig(FIG / "figA1_sigl_pre1500.png")
        plt.close(fig)
        print(f"\n  Saved {FIG/'figA1_sigl_pre1500.pdf'}")


if __name__ == "__main__":
    main()
