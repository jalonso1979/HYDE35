"""European Malthusian regression on Allen-style real wages, 1421-1900,
using city-buffer climate from ModE-RA gridded fields rather than country
means.

The country-mean approach smears every grid cell from Brittany to Provence
into a single France series for Strasbourg's 1700 wage. We instead extract
a 1.5-degree buffer around each Allen city, average the ModE-RA gridded
t_anom and p_anom over those cells, and rebuild the regression at
city-year resolution with city fixed effects.

Output:
  analysis/data/allen_city_climate_panel.parquet
  analysis/figures/paper4_v2/fig13b_allen_citybuffer.{pdf,png}
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"
MODERA = Path("/Volumes/BIGDATA/MODERA/extracted")

TEMP_NC = MODERA / "ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
PRECIP_NC = MODERA / "ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc"

BUFFER_DEG = 1.5  # half-side of a square centered on each city

# Allen cities → (lat, lon, iso3). Country aggregates anchored on a
# representative centroid; English aggregate uses London + south-east England
# centroid, Netherlands aggregate uses Amsterdam.
CITY_COORDS: dict[str, tuple[float, float, str]] = {
    "London":               (51.51,  -0.13, "GBR"),
    "England":              (51.85,  -0.50, "GBR"),  # SE England centroid
    "England (Winchester)": (51.06,  -1.31, "GBR"),
    "England (Cambridge)":  (52.21,   0.12, "GBR"),
    "England (Oxford)":     (51.75,  -1.26, "GBR"),
    "Amsterdam":            (52.37,   4.90, "NLD"),
    "Netherlands":          (52.37,   4.90, "NLD"),
    "Netherlands (Leiden)": (52.16,   4.50, "NLD"),
    "Antwerp":              (51.22,   4.40, "BEL"),
    "Strasbourg":           (48.57,   7.75, "FRA"),
    "Paris":                (48.86,   2.35, "FRA"),
    "Florence":             (43.77,  11.26, "ITA"),
    "Tuscany":              (43.50,  11.00, "ITA"),
    "Naples":               (40.85,  14.27, "ITA"),
    "Valencia":             (39.47,  -0.38, "ESP"),
}


def _extract_city_buffer_monthly() -> pd.DataFrame:
    """Open ModE-RA monthly fields and build a city-year-month panel.

    Returns long-form: city, iso3, year, month, t_anom_c, p_anom_mm.
    """
    print("Opening ModE-RA temperature and precipitation NetCDFs...")
    t = xr.open_dataset(TEMP_NC, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    p = xr.open_dataset(PRECIP_NC, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))

    # ModE-RA uses 0..360 longitude. Detect and shift if needed.
    lon_min = float(t["longitude"].min())
    lon_max = float(t["longitude"].max())
    print(f"  t.lon range: [{lon_min:.2f}, {lon_max:.2f}], "
          f"shape t.temp2={t['temp2'].shape}, time={t['time'].size}")

    # Project lon to dataset's convention (ModE-RA is [-180,180])
    use_360 = lon_max > 180
    def to_modera_lon(lon: float) -> float:
        if use_360:
            return lon % 360
        return ((lon + 180) % 360) - 180

    pieces = []
    for city, (lat, lon, iso3) in CITY_COORDS.items():
        mlat0, mlat1 = lat - BUFFER_DEG, lat + BUFFER_DEG
        mlon = to_modera_lon(lon)
        mlon0, mlon1 = mlon - BUFFER_DEG, mlon + BUFFER_DEG
        lon_lo, lon_hi = (0, 360) if use_360 else (-180, 180)
        assert lon_lo <= mlon0 and mlon1 <= lon_hi, (
            f"city {city} crosses dateline: lon={mlon}, buffer=[{mlon0},{mlon1}]")

        t_box = t["temp2"].sel(
            latitude=slice(mlat1, mlat0) if t["latitude"][0] > t["latitude"][-1]
                                          else slice(mlat0, mlat1),
            longitude=slice(mlon0, mlon1),
        )
        p_box = p["totprec"].sel(
            latitude=slice(mlat1, mlat0) if p["latitude"][0] > p["latitude"][-1]
                                          else slice(mlat0, mlat1),
            longitude=slice(mlon0, mlon1),
        )
        # Spatial mean (latitude weight is small over 3 deg; use cos weighting)
        lat_w = np.cos(np.deg2rad(t_box["latitude"].values))
        lat_w = lat_w / lat_w.sum()
        t_series = (t_box * xr.DataArray(lat_w, dims=["latitude"],
                                          coords={"latitude": t_box["latitude"]})
                    ).sum(dim="latitude").mean(dim="longitude")
        lat_w_p = np.cos(np.deg2rad(p_box["latitude"].values))
        lat_w_p = lat_w_p / lat_w_p.sum()
        p_series = (p_box * xr.DataArray(lat_w_p, dims=["latitude"],
                                          coords={"latitude": p_box["latitude"]})
                    ).sum(dim="latitude").mean(dim="longitude")

        times = t_series["time"].values
        # cftime to (year, month)
        years = np.array([d.year for d in times], dtype=np.int32)
        months = np.array([d.month for d in times], dtype=np.int8)
        df = pd.DataFrame({
            "city": city, "iso3": iso3,
            "year": years, "month": months,
            "t_anom_c": t_series.values.astype(np.float32),
            "p_anom_mm": p_series.values.astype(np.float32),
            "n_cells": int(t_box["latitude"].size * t_box["longitude"].size),
        })
        pieces.append(df)
        print(f"  {city:24s} ({lat:+.2f},{lon:+.2f}) "
              f"buffer {t_box['latitude'].size}x{t_box['longitude'].size} cells, "
              f"N months = {len(df)}")
    out = pd.concat(pieces, ignore_index=True)
    t.close(); p.close()
    return out


def _aggregate_to_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Collapse month → annual. ModE-RA totprec is kg/m^2/s; convert to mm/month
    so Spec 4 is unit-comparable with the published country-mean Spec 3."""
    # Days-per-month is approximate (30.44 average) for the rate→cumulative conversion
    SEC_PER_DAY = 86400.0
    DAYS_PER_MONTH = 30.4375
    monthly = monthly.copy()
    monthly["p_anom_mm"] = (monthly["p_anom_mm"]
                             * SEC_PER_DAY * DAYS_PER_MONTH).astype(np.float32)
    annual = monthly.groupby(["city", "iso3", "year"], as_index=False).agg(
        t_anom_c=("t_anom_c", "mean"),
        p_anom_mm=("p_anom_mm", "mean"),  # mean mm/month anomaly
        n_cells=("n_cells", "first"),
    )
    return annual


def _attach_wages_and_pop(annual_clim: pd.DataFrame) -> pd.DataFrame:
    wage = pd.read_csv(DATA / "allen_wage_panel.csv")
    wage = wage[wage["region"].isin(CITY_COORDS.keys())].copy()
    wage = wage.dropna(subset=["year_CE"])
    wage["year_CE"] = wage["year_CE"].astype(int)

    wage["log_real_wage"] = np.log(wage["real_wage"].replace({0: np.nan}))
    mask = (wage["log_real_wage"].isna() & wage["nominal_wage"].notna()
            & wage["wheat_price"].notna())
    wage.loc[mask, "log_real_wage"] = (
        np.log(wage.loc[mask, "nominal_wage"]) - np.log(wage.loc[mask, "wheat_price"]))
    wage = wage.dropna(subset=["log_real_wage"])

    wage_yr = wage.groupby(["region", "year_CE"], as_index=False).agg(
        log_real_wage=("log_real_wage", "mean"),
        n_obs=("source", "count"),
    ).rename(columns={"region": "city", "year_CE": "year"})

    df = wage_yr.merge(annual_clim, on=["city", "year"], how="inner")

    # HYDE country population (interpolated to annual, same as country script)
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
    long = long[long["pop"] > 0][["iso3", "year", "pop"]].sort_values(["iso3", "year"])

    pieces = []
    for iso, g in long.groupby("iso3"):
        g = g.sort_values("year")
        years = np.arange(g["year"].min(), g["year"].max() + 1)
        log_pop = np.interp(years, g["year"].values, np.log(g["pop"].values))
        pieces.append(pd.DataFrame({"iso3": iso, "year": years, "log_pop": log_pop}))
    pop_annual = pd.concat(pieces, ignore_index=True)

    df = df.merge(pop_annual, on=["iso3", "year"], how="inner")
    return df.sort_values(["city", "year"])


def _run_specs(df: pd.DataFrame, fe_col: str, label: str) -> dict:
    print(f"\n=== Spec set with FE = {label} (group var: {fe_col}) ===")
    pre = df[df["year"].between(1421, 1850)].copy()
    print(f"  N pre-industrial city/country-years = {len(pre):,}; "
          f"{pre[fe_col].nunique()} {label} groups")
    g = pre.groupby(fe_col)
    for c in ["log_real_wage", "log_pop", "t_anom_c", "p_anom_mm"]:
        pre[c + "_w"] = pre[c] - g[c].transform("mean")
    pre["log_pop_lag5_w"] = pre.groupby(fe_col)["log_pop_w"].shift(5)

    results = {}

    # Spec 1: lagged pop
    s1 = pre.dropna(subset=["log_real_wage_w", "log_pop_lag5_w"])
    r1 = sm.OLS(s1["log_real_wage_w"],
                sm.add_constant(s1[["log_pop_lag5_w"]])
                ).fit(cov_type="cluster", cov_kwds={"groups": s1[fe_col]})
    results["spec1"] = r1
    print(f"  Spec 1 (pop only):   N={int(r1.nobs):>5}  "
          f"β_log_pop = {r1.params['log_pop_lag5_w']:+.4f}  "
          f"SE={r1.bse['log_pop_lag5_w']:.4f}  p={r1.pvalues['log_pop_lag5_w']:.4g}")

    # Spec 2: climate only
    s2 = pre.dropna(subset=["log_real_wage_w", "t_anom_c_w", "p_anom_mm_w"])
    r2 = sm.OLS(s2["log_real_wage_w"],
                sm.add_constant(s2[["t_anom_c_w", "p_anom_mm_w"]])
                ).fit(cov_type="cluster", cov_kwds={"groups": s2[fe_col]})
    results["spec2"] = r2
    print(f"  Spec 2 (climate):    N={int(r2.nobs):>5}  "
          f"β_T = {r2.params['t_anom_c_w']:+.5f} "
          f"(SE={r2.bse['t_anom_c_w']:.5f}, p={r2.pvalues['t_anom_c_w']:.4g})   "
          f"β_P = {r2.params['p_anom_mm_w']:+.5f} "
          f"(SE={r2.bse['p_anom_mm_w']:.5f}, p={r2.pvalues['p_anom_mm_w']:.4g})")

    # Spec 3: joint
    s3 = pre.dropna(subset=["log_real_wage_w", "log_pop_lag5_w",
                              "t_anom_c_w", "p_anom_mm_w"])
    r3 = sm.OLS(s3["log_real_wage_w"],
                sm.add_constant(s3[["log_pop_lag5_w", "t_anom_c_w", "p_anom_mm_w"]])
                ).fit(cov_type="cluster", cov_kwds={"groups": s3[fe_col]})
    results["spec3"] = r3
    print(f"  Spec 3 (joint):      N={int(r3.nobs):>5}  R²={r3.rsquared:.4f}")
    for v in ["log_pop_lag5_w", "t_anom_c_w", "p_anom_mm_w"]:
        print(f"     {v:<20s}  β={r3.params[v]:+.5f}  "
              f"SE={r3.bse[v]:.5f}  p={r3.pvalues[v]:.4g}")
    return results


def main() -> None:
    print("Step 1: extract ModE-RA monthly fields at city buffers...")
    monthly = _extract_city_buffer_monthly()
    monthly.to_parquet(DATA / "allen_city_climate_monthly.parquet", index=False)
    print(f"  Saved monthly: {len(monthly):,} rows, "
          f"{monthly['city'].nunique()} cities")

    print("\nStep 2: aggregate to annual...")
    annual = _aggregate_to_annual(monthly)
    print(f"  Annual: {len(annual):,} city-years")

    print("\nStep 3: merge wages + HYDE-interpolated population...")
    df = _attach_wages_and_pop(annual)
    print(f"  Merged panel: {len(df):,} city-years, "
          f"{df['city'].nunique()} cities, {df['iso3'].nunique()} iso3")
    df.to_parquet(DATA / "allen_city_climate_panel.parquet", index=False)

    print("\nStep 4: regressions with city fixed effects (city-buffer climate)")
    city_results = _run_specs(df, fe_col="city",
                              label="city FE (city-buffer climate)")

    print("\nStep 5: regressions with country fixed effects, same city-buffer "
           "climate (averaged within country if multiple cities per country)")
    # Aggregate city-buffer climate to country-year by simple city mean
    cy = df.groupby(["iso3", "year"], as_index=False).agg(
        log_real_wage=("log_real_wage", "mean"),
        t_anom_c=("t_anom_c", "mean"),
        p_anom_mm=("p_anom_mm", "mean"),
        log_pop=("log_pop", "first"),
    )
    country_results = _run_specs(cy, fe_col="iso3",
                                 label="country FE (city-buffer climate, averaged)")

    # ── Figure: side-by-side city-buffer T vs country-mean T for one country ──
    print("\nStep 6: figure — comparing city-buffer vs country-mean T for France")
    fra_cities = df[df["iso3"] == "FRA"][["city", "year", "t_anom_c"]]
    country_clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    fra_country = country_clim[(country_clim["iso3"] == "FRA")
                                 ][["year", "t_c_anom_1971_2000"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for city, g in fra_cities.groupby("city"):
        g = g.sort_values("year")
        # 11-yr smooth so the multi-city differences are visible
        rolling = g["t_anom_c"].rolling(11, center=True, min_periods=5).mean()
        ax.plot(g["year"], rolling, linewidth=0.9,
                label=f"{city} buffer")
    fra_country = fra_country.sort_values("year")
    rolling_c = fra_country["t_c_anom_1971_2000"].rolling(11, center=True, min_periods=5).mean()
    ax.plot(fra_country["year"], rolling_c, color="black",
            linewidth=1.4, linestyle="--", label="FRA country mean")
    ax.axhline(0, color="#606060", linewidth=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$T$ anomaly (°C, 11-yr rolling)")
    ax.set_title("City-buffer vs country-mean temperature: France",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_xlim(1421, 1900)
    plt.tight_layout()
    fig.savefig(FIG / "fig13b_allen_citybuffer.pdf")
    fig.savefig(FIG / "fig13b_allen_citybuffer.png")
    plt.close(fig)
    print(f"  Saved {FIG / 'fig13b_allen_citybuffer.pdf'}")

    # ── Summary print: city FE Spec 3 vs the published country FE Spec 3 ──
    r3 = city_results["spec3"]
    print("\n" + "=" * 70)
    print("HEADLINE — Spec 3 (city FE, city-buffer climate):")
    print(f"  N = {int(r3.nobs):,}, R² = {r3.rsquared:.4f}")
    print(f"  β_lagged_log_pop = {r3.params['log_pop_lag5_w']:+.4f}  "
          f"(p = {r3.pvalues['log_pop_lag5_w']:.4g})")
    print(f"  β_T_anom         = {r3.params['t_anom_c_w']:+.5f}  "
          f"(p = {r3.pvalues['t_anom_c_w']:.4g})")
    print(f"  β_P_anom         = {r3.params['p_anom_mm_w']:+.5f}  "
          f"(p = {r3.pvalues['p_anom_mm_w']:.4g})")
    print("Compare to published country-FE, country-mean climate Spec 3:")
    print("  β_log_pop = -0.285 (p=0.10), β_T = +0.032 (p<0.01), β_P = +0.0001 (p≈0.5)")


if __name__ == "__main__":
    main()
