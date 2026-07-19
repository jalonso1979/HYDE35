"""Sigl-Toohey 2024 eVolv2k volcanic stratospheric sulfur injection panel.

Source: eVolv2k_sigl_toohey_2024.tab
Coverage: 500 BCE - 1900 CE (the dataset's declared range).
Post-1900 country-years are zero-filled (a known gap; would need Sato/GISS
volcanic AOD or similar for 20th-century IV).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis")
SRC = ROOT / "data" / "eVolv2k_sigl_toohey_2024.tab"
OUT = ROOT / "data" / "long_shadow_fertility" / "sigl_volcanic_panel.parquet"

COUNTRIES = ["GBR", "FRA", "ITA", "SWE"]

# Verified: header row is at line 49 (1-indexed), so skip 48 metadata lines.
HEADER_SKIPROWS = 48


def _parse_evolv2k() -> pd.DataFrame:
    raw = pd.read_csv(SRC, sep="\t", skiprows=HEADER_SKIPROWS, engine="python")
    # Column 0 is year, column 7 is VSSI [Tg]. Take by positional index for robustness.
    year_col = raw.columns[0]
    vssi_col = raw.columns[7]
    df = pd.DataFrame({
        "year": pd.to_numeric(raw[year_col], errors="coerce"),
        "vssi_tg_s": pd.to_numeric(raw[vssi_col], errors="coerce").fillna(0),
    }).dropna(subset=["year"]).astype({"year": int})
    # Sum across multiple eruptions in the same year
    df = df.groupby("year", as_index=False).agg(vssi_tg_s=("vssi_tg_s", "sum"))
    return df


def build_sigl_volcanic_panel(write: bool = False) -> pd.DataFrame:
    catalog = _parse_evolv2k()
    years = np.arange(1700, 2023)
    grid = pd.DataFrame(
        [(iso, y) for iso in COUNTRIES for y in years],
        columns=["iso3", "year"],
    )
    df = grid.merge(catalog, on="year", how="left")
    df["vssi_tg_s"] = df["vssi_tg_s"].fillna(0.0)
    df["log_vssi"] = np.log1p(df["vssi_tg_s"])
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_sigl_volcanic_panel(write=True)
    top = df.groupby("year")["vssi_tg_s"].first().nlargest(5)
    print(f"wrote {OUT}: {len(df)} rows; top VSSI years: {top.to_dict()}")
