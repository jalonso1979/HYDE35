"""Multi-country Maddison GDPpc panel (GBR + FRA + ITA + SWE).

Source: Bolt & van Zanden (2024). Maddison Project Database 2023.
Extends the England-only builder (build_maddison_england.py) to the four
core long-shadow countries used in Phase 2.

Output: data/long_shadow_fertility/maddison_multicountry_annual.parquet
Columns: iso3, year, gdppc (2011 international $), log_gdppc
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "maddison_multicountry_annual.parquet")

COUNTRIES = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"]


def build_maddison_multicountry(write: bool = False) -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"].isin(COUNTRIES), ["countrycode", "year", "gdppc"]].copy()
    sub = sub.rename(columns={"countrycode": "iso3"}).dropna(subset=["gdppc"])
    sub = sub.loc[sub["year"].between(1500, 2022)].sort_values(["iso3", "year"]).reset_index(drop=True)
    sub["log_gdppc"] = np.log(sub["gdppc"])
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(OUT, index=False)
    return sub


if __name__ == "__main__":
    df = build_maddison_multicountry(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['iso3'].nunique()} countries")
