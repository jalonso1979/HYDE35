"""Denmark country-year fertility from HMD DNK births + Maddison population.

Output
------
analysis/data/long_shadow_fertility/dnk_fertility_annual.parquet
Columns: year, iso3, births, population, cbr, log_cbr, source
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
HMD_BIRTHS = FERTILITY / "data" / "mortality" / "births" / "DNK.Births.txt"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "dnk_fertility_annual.parquet"
)

ISO3 = "DNK"


def _hmd_births() -> pd.DataFrame:
    df = pd.read_csv(HMD_BIRTHS, sep=r"\s+", skiprows=2, engine="python")
    df = df.loc[:, ["Year", "Total"]].rename(
        columns={"Year": "year", "Total": "births"}
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    df = df.dropna(subset=["year", "births"]).astype({"year": int})
    return df


def _maddison_pop(iso3: str) -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"] == iso3, ["year", "pop"]].dropna().copy()
    sub["population"] = sub["pop"].astype(float) * 1000.0
    sub["year"] = sub["year"].astype(int)
    return sub[["year", "population"]]


def build_dnk_fertility_annual(write: bool = False) -> pd.DataFrame:
    births = _hmd_births()
    pop = _maddison_pop(ISO3)
    df = births.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["iso3"] = ISO3
    df["source"] = f"HMD_{ISO3}_Maddison_pop"
    df = df[["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]]
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_dnk_fertility_annual(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}")
    print("\nCBR sanity (per 1000):")
    print(df["cbr"].describe())
