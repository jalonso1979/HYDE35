"""Build Maddison Project England GDP-per-capita series.

Source: Bolt & van Zanden (2024). Maddison Project Database 2023.
We use the GBR country code; Maddison's GBR backcast is the Wrigley-Schofield
England series rescaled, so it is appropriate for the England-only panel.

Output: data/long_shadow_fertility/maddison_england_annual_1500_2022.parquet
Columns: year, gdppc (2011 international $), log_gdppc
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FERTILITY_ROOT = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
MADDISON_XLSX = FERTILITY_ROOT / "data" / "mpd2023_web.xlsx"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "maddison_england_annual_1500_2022.parquet")


def build_maddison_england(write: bool = False) -> pd.DataFrame:
    raw = pd.read_excel(MADDISON_XLSX, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    gbr = raw.loc[raw["countrycode"] == "GBR", ["year", "gdppc"]].copy()
    gbr = gbr.dropna(subset=["gdppc"])
    gbr = gbr.loc[gbr["year"].between(1500, 2022)].sort_values("year").reset_index(drop=True)
    gbr["log_gdppc"] = np.log(gbr["gdppc"])
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        gbr.to_parquet(OUT, index=False)
    return gbr


if __name__ == "__main__":
    df = build_maddison_england(write=True)
    print(f"wrote {OUT} ({len(df)} rows, {df['year'].min()}-{df['year'].max()})")
