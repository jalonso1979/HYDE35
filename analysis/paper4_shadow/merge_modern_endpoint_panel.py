"""Build a single modern-endpoint panel for Paper D long-shadow analyses by
merging ERA5 (climate covariates) + PRIMAP-hist v2.7 (emissions endpoint) +
HYDE 3.5 (land-use baseline).

Built 2026-05-20 as scaffolding for when Paper D drafting begins. Loads the
PRIMAP parquet built by ingest_primap_hist_v27.py and the ERA5 regional panel
built by ../update_era5_panel.py, and produces a country×year panel ready for
joining against HYDE land-use grids.

The resulting panel is intended as the *modern endpoint* of long-shadow
analyses that originate in pre-Columbian or medieval baselines — it does not
itself answer the long-shadow question, but it supplies the modern dependent
variables that those analyses regress on.

Run (after PRIMAP + ERA5 ingestions complete):
    cd /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow
    python3 merge_modern_endpoint_panel.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PRIMAP_LONG = DATA / "primap_hist_v27_long.parquet"
ERA5_PANEL = DATA / "era5_full_panel.parquet"
ERA5_COUNTRY = DATA / "era5_country_monthly.parquet"
OUT = DATA / "long_shadow_modern_endpoint_panel.parquet"


def load_primap_country_annual() -> pd.DataFrame:
    """Load PRIMAP CO2/CH4/N2O at country×year, total excluding LULUCF."""
    p = pd.read_parquet(PRIMAP_LONG)
    p = p[
        p.entity.isin(["CO2", "CH4", "N2O"])
        & (p.scenario == "HISTCR")
        & (p.category == "M.0.EL")
        & (p.iso3 != "EARTH")
        & (~p.iso3.str.startswith("ANNEX"))
        & (~p.iso3.str.startswith("EU"))
    ]
    wide = p.pivot_table(
        index=["iso3", "year"],
        columns="entity",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"CO2": "co2_kt", "CH4": "ch4_kt", "N2O": "n2o_kt"})
    return wide


def load_era5_country_annual() -> pd.DataFrame:
    """Aggregate ERA5 country-monthly to annual means."""
    if not ERA5_COUNTRY.exists():
        return pd.DataFrame()
    m = pd.read_parquet(ERA5_COUNTRY)
    keep = [c for c in m.columns if c in {"iso3", "country", "year", "month", "t2m_c", "tp_mm"}]
    m = m[keep]
    g = m.groupby([c for c in ("iso3", "country", "year") if c in m.columns]).agg(
        t2m_annual_c=("t2m_c", "mean"),
        tp_annual_mm=("tp_mm", "mean"),
    ).reset_index()
    return g


def main() -> None:
    print("Loading PRIMAP country×year …")
    primap = load_primap_country_annual()
    print(f"  rows: {len(primap):,}, countries: {primap.iso3.nunique()}, "
          f"years: {primap.year.min()}–{primap.year.max()}")

    print("Loading ERA5 country×year …")
    era5 = load_era5_country_annual()
    if era5.empty:
        print("  ERA5 country-monthly panel not yet built — re-run ../update_era5_panel.py")
        print("  Falling back to PRIMAP-only modern endpoint.")
        merged = primap.copy()
    else:
        print(f"  rows: {len(era5):,}, countries: {era5.iso3.nunique() if 'iso3' in era5.columns else 'n/a'}, "
              f"years: {era5.year.min()}–{era5.year.max()}")
        on = [c for c in ("iso3", "year") if c in era5.columns]
        merged = primap.merge(era5, on=on, how="left")

    merged.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}")
    print()
    print("Summary by year (every 50 years):")
    summary = merged[merged.year.isin([1750, 1800, 1850, 1900, 1950, 2000, 2020])]
    print(summary.groupby("year").agg(
        n_countries=("iso3", "nunique"),
        co2_total_kt=("co2_kt", "sum"),
        ch4_total_kt=("ch4_kt", "sum"),
    ).to_string())


if __name__ == "__main__":
    main()
