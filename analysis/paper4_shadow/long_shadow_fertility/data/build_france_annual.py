"""France country-year fertility from HMD FRATNP births + Maddison population.

Source choice
-------------
The Phase 2 plan calls for "HFD FRATNP" annual France births. The HFD bulk
file `data/HFD/totbirthsRR.txt` only covers France from 1946 onward, which
does not meet the Phase 2 coverage target (long-shadow analysis requires the
pre-1820 / post-2018 envelope).

We instead read HMD `FRATNP.Births.txt` (Human Mortality Database, same
country code), which provides a single continuous 1-year France total births
series 1806-2020. HMD and HFD share the same FRATNP definition for total
births; HMD's series is simply longer because the underlying INSEE/SGF civil
registration files extend back to 1806.

Population denominator
----------------------
Maddison Project Database 2023 ("Full data" sheet), column `pop` is in
*thousands of persons*; we multiply by 1000 to get absolute persons. For
France 1820 the Maddison value is 31,250 (thousand) = 31.25 million, and
for 2018 it is ~67,347 (thousand) = ~67.35 million — both standard.

Output
------
data/long_shadow_fertility/france_fertility_annual.parquet
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
HFD = FERTILITY / "data" / "mortality" / "births" / "FRATNP.Births.txt"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "france_fertility_annual.parquet"
)


def _hfd_france_births() -> pd.DataFrame:
    """Read HMD FRATNP single-year total births (1806-2020).

    File layout: row 1 is a title line, row 2 is blank, row 3 holds the
    column headers `Year Female Male Total`, data follows from row 4.
    """
    df = pd.read_csv(HFD, sep=r"\s+", skiprows=2, engine="python")
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


def build_france_annual(write: bool = False) -> pd.DataFrame:
    """Return France country-year fertility 1806-2020.

    Columns: year, iso3, births, population, cbr, log_cbr, source.
    """
    births = _hfd_france_births()
    pop = _maddison_pop("FRA")
    df = births.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["iso3"] = "FRA"
    df["source"] = "HMD_FRATNP_Maddison_pop"
    df = df[
        ["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]
    ]
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_france_annual(write=True)
    print(
        f"wrote {OUT}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}"
    )
    print("\nCBR sanity (per 1000):")
    print(df["cbr"].describe())
