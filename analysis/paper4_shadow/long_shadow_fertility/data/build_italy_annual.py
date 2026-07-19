"""Italy country-year fertility from HMD ITA births + Maddison population.

Source choice
-------------
The Phase 2 plan originally called for "HFD totbirthsRR" annual Italy births,
but HFD's Italy coverage in the bulk file is limited and does not meet the
Phase 2 envelope requirements. We instead read HMD `ITA.Births.txt` (Human
Mortality Database), which provides a single continuous 1-year Italy total
births series 1862-2019 covering post-unification Italy. This mirrors the
substitution applied to France in Task 2.

Population denominator
----------------------
Maddison Project Database 2023 ("Full data" sheet), column `pop` is in
*thousands of persons*; we multiply by 1000 to get absolute persons.

Output
------
analysis/data/long_shadow_fertility/italy_fertility_annual.parquet
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
HMD_ITA = FERTILITY / "data" / "mortality" / "births" / "ITA.Births.txt"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "italy_fertility_annual.parquet"
)


def _hmd_italy_births() -> pd.DataFrame:
    """Read HMD ITA single-year total births (1862-2019).

    File layout: row 1 is a title line, row 2 is blank, row 3 holds the
    column headers `Year Female Male Total`, data follows from row 4.
    """
    df = pd.read_csv(HMD_ITA, sep=r"\s+", skiprows=2, engine="python")
    df = df.loc[:, ["Year", "Total"]].rename(
        columns={"Year": "year", "Total": "births"}
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    df = df.dropna(subset=["year", "births"]).astype({"year": int})
    return df


def _maddison_pop(iso3: str) -> pd.DataFrame:
    """Maddison `pop` (thousands of persons) for `iso3`, converted to persons."""
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"] == iso3, ["year", "pop"]].dropna().copy()
    sub["population"] = sub["pop"].astype(float) * 1000.0
    sub["year"] = sub["year"].astype(int)
    return sub[["year", "population"]]


def build_italy_annual(write: bool = False) -> pd.DataFrame:
    """Return Italy country-year fertility 1862-2019.

    Columns: year, iso3, births, population, cbr, log_cbr, source.
    """
    births = _hmd_italy_births()
    pop = _maddison_pop("ITA")
    df = births.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["iso3"] = "ITA"
    df["source"] = "HMD_ITA_Maddison_pop"
    df = df[
        ["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]
    ]
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_italy_annual(write=True)
    print(
        f"wrote {OUT}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}"
    )
    print("\nCBR sanity (per 1000):")
    print(df["cbr"].describe())
