"""Merge fertility + climate + GDP + volcanic-event metadata into one panel."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
FERT = ROOT / "england_fertility_annual_1541_2020.parquet"
CLIM = ROOT / "england_climate_annual_1421_2008.parquet"
GDP = ROOT / "maddison_england_annual_1500_2022.parquet"
OUT = ROOT / "england_panel_1541_2020.parquet"

ERUPTIONS = [
    ("Huaynaputina", 1600),
    ("Parker",       1641),
    ("Tambora",      1815),
    ("Krakatoa",     1883),
    ("Pinatubo",     1991),
]


def _years_to_nearest(year: int, anchors: list[int]) -> int:
    return int(min(abs(year - a) for a in anchors))


def assemble_england_panel(write: bool = False) -> pd.DataFrame:
    fert = pd.read_parquet(FERT)
    clim = pd.read_parquet(CLIM)
    gdp = pd.read_parquet(GDP)
    df = fert.merge(clim, on="year", how="left").merge(gdp[["year", "gdppc", "log_gdppc"]], on="year", how="left")
    eruption_years = [y for _, y in ERUPTIONS]
    df["is_eruption_year"] = df["year"].isin(eruption_years).astype(int)
    df["years_to_nearest_eruption"] = df["year"].apply(lambda y: _years_to_nearest(y, eruption_years))
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = assemble_england_panel(write=True)
    print(f"wrote {OUT} ({len(df)} rows)")
