"""Ingest PRIMAP-hist v2.7 national emissions panel into long-format parquet
for paper4_shadow long-shadow analyses.

Built 2026-05-20 as part of the SETI data-sweep scaffolding for Paper D.

Source: SETI/Data/emissions_ipcc/Guetschow_et_al_2025a-PRIMAP-hist_v2.7_final_no_rounding_22-Aug-2025.csv
        (Zenodo 17090760)

Coverage:
  - Years 1750–2024 (275 cols)
  - Countries: ISO3 codes (193 + aggregates)
  - Entities: CO2, CH4, N2O, KYOTOGHG, F-gases, etc.
  - Scenarios: HISTCR (country-reported priority), HISTTP (third-party priority)
  - Categories: IPCC2006_PRIMAP sectors

This script reshapes the wide format (one row per entity × area × scenario ×
category, with one column per year) into long format suitable for joining
against HYDE land-use grids and the analysis stack.

Run:
    cd /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow
    python3 ingest_primap_hist_v27.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANDEMICS = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/"
    "My Drive/Pandemics"
)
PRIMAP_CSV = (
    PANDEMICS / "SETI" / "Data" / "emissions_ipcc"
    / "Guetschow_et_al_2025a-PRIMAP-hist_v2.7_final_no_rounding_22-Aug-2025.csv"
)
OUT_DIR = ROOT / "analysis" / "data"
OUT_LONG = OUT_DIR / "primap_hist_v27_long.parquet"
OUT_HEADLINE = OUT_DIR / "primap_hist_v27_co2_global_annual.parquet"


HEADLINE_ENTITIES = ("CO2", "CH4", "N2O", "KYOTOGHG (AR6GWP100)")
HEADLINE_SCENARIOS = ("HISTCR", "HISTTP")
HEADLINE_AREA = "EARTH"  # PRIMAP global aggregate


def main() -> None:
    print(f"Reading {PRIMAP_CSV.name} …")
    df = pd.read_csv(PRIMAP_CSV)
    print(f"  shape: {df.shape}")

    id_cols = [
        "source",
        "scenario (PRIMAP-hist)",
        "provenance",
        "area (ISO3)",
        "entity",
        "unit",
        "category (IPCC2006_PRIMAP)",
    ]
    year_cols = [c for c in df.columns if c.isdigit()]
    long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["value"])
    long = long.rename(columns={
        "scenario (PRIMAP-hist)": "scenario",
        "area (ISO3)": "iso3",
        "category (IPCC2006_PRIMAP)": "category",
    })
    print(f"  long-format rows: {len(long):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT_LONG, index=False)
    print(f"Wrote {OUT_LONG}")

    headline = long[
        long.entity.isin(HEADLINE_ENTITIES)
        & long.scenario.isin(HEADLINE_SCENARIOS)
        & (long.iso3 == HEADLINE_AREA)
        & (long.category == "M.0.EL")  # National total excluding LULUCF (PRIMAP convention)
    ]
    if headline.empty:
        # Fall back to total including LULUCF
        headline = long[
            long.entity.isin(HEADLINE_ENTITIES)
            & long.scenario.isin(HEADLINE_SCENARIOS)
            & (long.iso3 == HEADLINE_AREA)
        ]
    headline.to_parquet(OUT_HEADLINE, index=False)
    print(f"Wrote {OUT_HEADLINE} ({len(headline)} rows)")
    print()
    print(headline.groupby(["entity", "scenario"]).agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        n=("year", "count"),
    ).to_string())


if __name__ == "__main__":
    main()
