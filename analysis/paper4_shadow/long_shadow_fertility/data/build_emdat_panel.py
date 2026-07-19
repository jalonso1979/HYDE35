"""Country-year EMDAT disaster panel.

Aggregates monthly COUNTRY_LONG-M EMDAT indices (sheet COUNTRY_LONGM_IDX)
to annual country-year sums for our four-country sample (GBR, FRA, ITA,
SWE). The source begins in 1962 for most countries, with country-specific
coverage start dates (GBR: 1938-01, SWE: 1911-01, ITA: 1995-01) and ends
in 2025-12. FRA is absent from the source entirely and therefore has zero
counts/deaths throughout the panel.

The output is a balanced country-year panel spanning 1900..max_source_year,
zero-filled outside each country's source coverage so downstream merges
with fertility, climate, and war panels join cleanly without NaN gaps.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
SRC = FERTILITY / "data" / "PROCESSED" / "COUNTRY_LONG-M_with_EMDAT_disaster_indices.xlsx"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "emdat_panel_country_year.parquet")

COUNTRIES = ["GBR", "FRA", "ITA", "SWE"]
START_YEAR = 1900


def build_emdat_panel(write: bool = False) -> pd.DataFrame:
    raw = pd.read_excel(SRC, sheet_name="COUNTRY_LONGM_IDX")
    raw = raw.loc[raw["country"].isin(COUNTRIES)].copy()
    raw["year"] = pd.to_datetime(raw["time"].astype(str), format="%Y-%m").dt.year
    annual = (raw.groupby(["country", "year"], as_index=False)
              .agg(disaster_count=("count_all", "sum"),
                   disaster_count_natural=("count_natural", "sum"),
                   disaster_count_transport=("count_transport", "sum"),
                   disaster_deaths=("deaths_all", "sum")))
    annual = annual.rename(columns={"country": "iso3"})

    # Build a balanced grid 1900..max(source year, fallback 2025) for all
    # four countries, then left-merge the EMDAT aggregates onto it.
    max_year = int(raw["year"].max()) if len(raw) else 2025
    grid = pd.DataFrame(
        [(iso, y) for iso in COUNTRIES for y in range(START_YEAR, max_year + 1)],
        columns=["iso3", "year"],
    )
    df = grid.merge(annual, on=["iso3", "year"], how="left")
    count_cols = ["disaster_count", "disaster_count_natural",
                  "disaster_count_transport", "disaster_deaths"]
    df[count_cols] = df[count_cols].fillna(0)
    # Keep counts as integers (per spec, counts_nonneg test expects numeric).
    for c in ["disaster_count", "disaster_count_natural",
              "disaster_count_transport"]:
        df[c] = df[c].astype(int)
    df["log_disaster_deaths"] = np.log1p(df["disaster_deaths"])
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_emdat_panel(write=True)
    print(f"wrote {OUT}: {len(df)} rows, max year {df['year'].max()}")
