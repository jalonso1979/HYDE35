"""Country-year pandemic indicator panel.

Sources:
- Pre-1500: conflict_pandemic_panel.parquet (city-year plague_active),
  aggregated to country-year via city->ISO map for our four countries.
- Black Death 1346-1353 forced active for all four (Europe-wide).
- Modern epidemics (1832+) hand-coded as Europe-wide events affecting all four.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis")
CONFLICT_PANDEMIC = ROOT / "data" / "conflict_pandemic_panel.parquet"
OUT = ROOT / "data" / "long_shadow_fertility" / "pandemic_panel_country_year.parquet"

COUNTRIES = ["GBR", "FRA", "ITA", "SWE"]

CITY_TO_ISO = {
    "london": "GBR", "york": "GBR",
    "paris": "FRA", "marseille": "FRA", "lyon": "FRA", "orleans": "FRA",
    "rome": "ITA", "venice": "ITA", "florence": "ITA", "milan": "ITA",
    "naples": "ITA", "genoa": "ITA", "siena": "ITA",
    "stockholm": "SWE",
}

MODERN_PANDEMICS = [
    (1832, 1834), (1849, 1854), (1893, 1894),
    (1918, 1919), (1957, 1957), (1968, 1968),
    (2009, 2009), (2020, 2022),
]
BLACK_DEATH = (1346, 1353)


def build_pandemic_panel(write: bool = False) -> pd.DataFrame:
    cp = pd.read_parquet(CONFLICT_PANDEMIC)
    cp["city_lc"] = cp["city"].str.lower()
    cp["iso3"] = cp["city_lc"].map(CITY_TO_ISO)
    cp = cp.dropna(subset=["iso3"])
    cp = cp.loc[cp["iso3"].isin(COUNTRIES)]
    pre = (cp.groupby(["iso3", "year"], as_index=False)
           .agg(plague_max=("plague_active", "max")))
    pre["pandemic_active"] = pre["plague_max"].astype(int)
    pre = pre[["iso3", "year", "pandemic_active"]]

    years = np.arange(100, 2023)
    grid = pd.DataFrame(
        [(iso, y) for iso in COUNTRIES for y in years],
        columns=["iso3", "year"],
    )
    df = grid.merge(pre, on=["iso3", "year"], how="left")
    df["pandemic_active"] = df["pandemic_active"].fillna(0).astype(int)

    # Black Death
    for iso in COUNTRIES:
        mask = (df["iso3"] == iso) & df["year"].between(*BLACK_DEATH)
        df.loc[mask, "pandemic_active"] = 1

    # Modern Europe-wide epidemics
    for iso in COUNTRIES:
        for (y0, y1) in MODERN_PANDEMICS:
            mask = (df["iso3"] == iso) & df["year"].between(y0, y1)
            df.loc[mask, "pandemic_active"] = 1

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_pandemic_panel(write=True)
    print(f"wrote {OUT}: {len(df)} rows; {df['pandemic_active'].sum()} pandemic country-years")
