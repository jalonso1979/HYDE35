"""European Malthusian regression on Allen-style real wages, 1421-1900.

Tests the canonical Malthusian prediction with a direct welfare measure:
real wages (Allen welfare ratios + grain-price-deflated nominal wages).
Pre-industrial wages should respond negatively to lagged density (Malthus
positive check); positively to climate shocks that destroy harvests.

Uses the parsed Allen wage panel (`allen_wage_panel.csv`) covering
~15 European city/region series. Merges with country-year climate from
the calibrated 1421-2025 panel.
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

# Map Allen regions to HYDE iso3 codes
REGION_TO_ISO3 = {
    "England": "GBR", "London": "GBR",
    "England (Winchester)": "GBR", "England (Cambridge)": "GBR",
    "England (Oxford)": "GBR",
    "Netherlands": "NLD", "Netherlands (Leiden)": "NLD",
    "Amsterdam": "NLD", "Antwerp": "BEL",
    "Strasbourg": "FRA", "Paris": "FRA",
    "Florence": "ITA", "Tuscany": "ITA", "Naples": "ITA",
    "Valencia": "ESP",
}


def _load_wage_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA / "allen_wage_panel.csv")
    df["iso3"] = df["region"].map(REGION_TO_ISO3)
    df = df.dropna(subset=["iso3", "year_CE"]).copy()
    df["year_CE"] = df["year_CE"].astype(int)
    # Compute log real wage where available
    df["log_real_wage"] = np.log(df["real_wage"].replace({0: np.nan}))
    # Where real wage is missing but nominal+price available, compute deflated
    mask = df["log_real_wage"].isna() & df["nominal_wage"].notna() & df["wheat_price"].notna()
    df.loc[mask, "log_real_wage"] = (np.log(df.loc[mask, "nominal_wage"]) -
                                       np.log(df.loc[mask, "wheat_price"]))
    df = df.dropna(subset=["log_real_wage"])
    # If multiple series per (iso3, year), average them (city-level → country)
    panel = df.groupby(["iso3", "year_CE"], as_index=False).agg(
        log_real_wage=("log_real_wage", "mean"),
        n_series=("source", "count"),
    )
    panel = panel.rename(columns={"year_CE": "year"})
    return panel.sort_values(["iso3", "year"])


def _attach_climate_and_density(wages: pd.DataFrame) -> pd.DataFrame:
    clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    clim = clim[["iso3", "year", "t_c", "t_c_anom_1971_2000",
                  "p_mm", "p_mm_anom_1971_2000"]]
    out = wages.merge(clim, on=["iso3", "year"], how="inner")

    # Load HYDE subnational pop (sum to country) at decadal timesteps; interpolate to annual
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
                    var_name="ycol", value_name="pop")
    long["year"] = long["ycol"].str.lstrip("y").astype(int)
    long = long[long["pop"] > 0]
    long = long[["iso3", "year", "pop"]].sort_values(["iso3", "year"])

    # Linear interpolation in log space to annual frequency
    pieces = []
    for iso, g in long.groupby("iso3"):
        g = g.sort_values("year")
        years = np.arange(g["year"].min(), g["year"].max() + 1)
        log_pop = np.interp(years, g["year"].values, np.log(g["pop"].values))
        pieces.append(pd.DataFrame({"iso3": iso, "year": years, "log_pop": log_pop}))
    pop_annual = pd.concat(pieces, ignore_index=True)

    out = out.merge(pop_annual, on=["iso3", "year"], how="inner")
    return out.sort_values(["iso3", "year"])


def main() -> None:
    print("Loading Allen wage panel...")
    wages = _load_wage_panel()
    print(f"  {len(wages):,} country-year wage observations, "
          f"{wages['iso3'].nunique()} countries, "
          f"years {wages['year'].min()}-{wages['year'].max()}")
    print("Coverage by country:")
    print(wages.groupby("iso3")["year"].agg(["count", "min", "max"]))

    print("\nAttaching climate and HYDE-interpolated population...")
    df = _attach_climate_and_density(wages)
    print(f"  Merged sample: {len(df):,} country-year cells")

    # Pre-industrial sample: 1421-1850 (before industrial wage compression)
    pre = df[df["year"].between(1421, 1850)].copy()
    print(f"\nPre-industrial 1421-1850 sample: {len(pre):,} country-years")

    # Country fixed effects via within-country demeaning
    g = pre.groupby("iso3")
    for c in ["log_real_wage", "log_pop", "t_c_anom_1971_2000", "p_mm_anom_1971_2000"]:
        pre[c + "_w"] = pre[c] - g[c].transform("mean")

    # ── Specification 1: Malthusian — wage ~ density (negative expected) ──
    print("\n=== Spec 1: Allen Malthusian — log_real_wage on lagged log_pop (country FE) ===")
    pre["log_pop_lag5_w"] = (pre.groupby("iso3")["log_pop_w"].shift(5))
    s1 = pre.dropna(subset=["log_real_wage_w", "log_pop_lag5_w"])
    X = sm.add_constant(s1[["log_pop_lag5_w"]])
    y = s1["log_real_wage_w"]
    r1 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": s1["iso3"]})
    print(f"N={int(r1.nobs):>6}, R²={r1.rsquared:.4f}, "
          f"β_log_pop = {r1.params['log_pop_lag5_w']:+.4f}  "
          f"SE = {r1.bse['log_pop_lag5_w']:.4f}  p = {r1.pvalues['log_pop_lag5_w']:.4g}")

    # ── Specification 2: climate response — wage on annual T anom ────────
    print("\n=== Spec 2: log_real_wage on contemporaneous T and P anomalies ===")
    s2 = pre.dropna(subset=["log_real_wage_w", "t_c_anom_1971_2000_w",
                              "p_mm_anom_1971_2000_w"])
    X = sm.add_constant(s2[["t_c_anom_1971_2000_w", "p_mm_anom_1971_2000_w"]])
    y = s2["log_real_wage_w"]
    r2 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": s2["iso3"]})
    for v in ["t_c_anom_1971_2000_w", "p_mm_anom_1971_2000_w"]:
        print(f"  {v[:30]:>30s}: β = {r2.params[v]:+.5f}  "
              f"SE = {r2.bse[v]:.5f}  p = {r2.pvalues[v]:.4g}")

    # ── Specification 3: joint Malthus + climate ─────────────────────────
    print("\n=== Spec 3: log_real_wage ~ lagged pop + T + P (country FE) ===")
    s3 = pre.dropna(subset=["log_real_wage_w", "log_pop_lag5_w",
                              "t_c_anom_1971_2000_w", "p_mm_anom_1971_2000_w"])
    X = sm.add_constant(s3[["log_pop_lag5_w", "t_c_anom_1971_2000_w", "p_mm_anom_1971_2000_w"]])
    y = s3["log_real_wage_w"]
    r3 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": s3["iso3"]})
    print(f"N={int(r3.nobs):>6}, R²={r3.rsquared:.4f}")
    for v in ["log_pop_lag5_w", "t_c_anom_1971_2000_w", "p_mm_anom_1971_2000_w"]:
        print(f"  {v[:30]:>30s}: β = {r3.params[v]:+.5f}  "
              f"SE = {r3.bse[v]:.5f}  p = {r3.pvalues[v]:.4g}")

    df.to_parquet(DATA / "allen_wage_climate_panel.parquet", index=False)

    # Figure: scatter for one country
    big_iso = wages["iso3"].value_counts().index[0]
    sub = pre[pre["iso3"] == big_iso].sort_values("year")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    ax = axes[0]
    ax.plot(sub["year"], sub["log_real_wage"], color="#202020", linewidth=0.9)
    ax.set_xlabel("Year"); ax.set_ylabel("log real wage")
    ax.set_title(f"(a) Allen real wage, {big_iso}", loc="left")

    ax = axes[1]
    ax.scatter(sub["log_pop"], sub["log_real_wage"], s=4, color="#404040", alpha=0.5)
    if len(sub) > 10:
        z = np.polyfit(sub["log_pop"], sub["log_real_wage"], 1)
        xs = np.array([sub["log_pop"].min(), sub["log_pop"].max()])
        ax.plot(xs, z[0]*xs + z[1], color="#000000", linewidth=1.2)
        ax.text(0.04, 0.96, fr"$\hat\beta = {z[0]:+.2f}$",
                transform=ax.transAxes, va="top")
    ax.set_xlabel(r"$\ln$ pop"); ax.set_ylabel("log real wage")
    ax.set_title(f"(b) Malthusian relationship, {big_iso}", loc="left")
    plt.tight_layout()
    fig.savefig(FIG / "fig13_allen_wages.pdf")
    fig.savefig(FIG / "fig13_allen_wages.png")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig13_allen_wages.pdf'}")


if __name__ == "__main__":
    main()
