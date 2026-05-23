"""Real-wage panel: Allen city-aggregated 1421-1913 + Maddison GDPpc proxy 1914+."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

BIGDATA = Path("/Volumes/BIGDATA/HYDE35/analysis")
FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
ALLEN = BIGDATA / "data" / "allen_wage_climate_panel.parquet"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = BIGDATA / "data" / "long_shadow_fertility" / "real_wage_panel.parquet"

COUNTRIES = ["GBR", "FRA", "ITA", "SWE"]


def _allen_part() -> pd.DataFrame:
    allen = pd.read_parquet(ALLEN)
    sub = allen.loc[allen["iso3"].isin(COUNTRIES), ["iso3", "year", "log_real_wage"]].copy()
    sub["source"] = "Allen_cropweighted"
    return sub


def _maddison_proxy() -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"].isin(COUNTRIES), ["countrycode", "year", "gdppc"]].copy()
    sub = sub.rename(columns={"countrycode": "iso3"}).dropna(subset=["gdppc"])
    sub = sub.loc[sub["year"] > 1913].copy()
    sub["log_real_wage"] = np.log(sub["gdppc"])
    sub["source"] = "Maddison_gdppc_proxy"
    return sub[["iso3", "year", "log_real_wage", "source"]]


def build_real_wage_panel(write: bool = False) -> pd.DataFrame:
    allen = _allen_part()
    mad = _maddison_proxy()
    df = pd.concat([allen, mad], ignore_index=True)
    df = df.sort_values(["iso3", "year"]).drop_duplicates(["iso3", "year"], keep="first").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_real_wage_panel(write=True)
    print(f"wrote {OUT}: {len(df)} rows; sources: {df['source'].value_counts().to_dict()}")
