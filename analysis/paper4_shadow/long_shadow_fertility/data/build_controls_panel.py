"""Merge all control sources into one (iso3, year, controls...) panel."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
WAR = ROOT / "war_panel_country_year.parquet"
PAND = ROOT / "pandemic_panel_country_year.parquet"
EMDAT = ROOT / "emdat_panel_country_year.parquet"
EXTR = ROOT / "climate_extremes_country_year.parquet"
CLIM = ROOT / "country_climate_annual.parquet"
OUT = ROOT / "controls_panel_country_year.parquet"


def build_controls_panel(write: bool = False) -> pd.DataFrame:
    war = pd.read_parquet(WAR)
    pand = pd.read_parquet(PAND)
    emdat = pd.read_parquet(EMDAT)
    extr = pd.read_parquet(EXTR)
    clim = pd.read_parquet(CLIM)[["iso3", "year", "vol_t_10y", "vol_p_10y"]]
    df = (war
          .merge(pand, on=["iso3", "year"], how="outer")
          .merge(emdat[["iso3", "year", "disaster_count", "log_disaster_deaths"]],
                 on=["iso3", "year"], how="outer")
          .merge(extr, on=["iso3", "year"], how="outer")
          .merge(clim, on=["iso3", "year"], how="outer"))
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_controls_panel(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['iso3'].nunique()} countries")
