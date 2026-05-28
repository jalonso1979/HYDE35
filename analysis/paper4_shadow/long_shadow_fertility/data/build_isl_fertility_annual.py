"""Iceland country-year fertility from HMD ISL births + HMD exposures population.

HMD ISL births start 1838; Maddison ISL GDPpc starts only 1950, which leaves
112 years of births without a population denominator. We use the HMD
Exposures_1x1 file (sum across ages = mid-year person-years, a close
approximation to mid-year population) which covers the full 1838-2020 range.
Maddison pop is used for 2021+ if ISL HMD Births extend beyond 2020.

No Allen welfare-ratio series exists for Iceland — uses Maddison-only wage path
(same as Sweden in the wage panel).

Output
------
analysis/data/long_shadow_fertility/isl_fertility_annual.parquet
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
HMD_BIRTHS = FERTILITY / "data" / "mortality" / "births" / "ISL.Births.txt"
HMD_EXP = (FERTILITY / "data" / "mortality" / "exposures"
            / "Exposures_1x1" / "ISL.Exposures_1x1.txt")
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "isl_fertility_annual.parquet"
)

ISO3 = "ISL"


def _hmd_births() -> pd.DataFrame:
    df = pd.read_csv(HMD_BIRTHS, sep=r"\s+", skiprows=2, engine="python")
    df = df.loc[:, ["Year", "Total"]].rename(
        columns={"Year": "year", "Total": "births"}
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    df = df.dropna(subset=["year", "births"]).astype({"year": int})
    return df


def _hmd_exposures_pop() -> pd.DataFrame:
    """Sum all-age person-years from HMD Exposures_1x1 as population proxy."""
    df = pd.read_csv(HMD_EXP, sep=r"\s+", skiprows=2, engine="python")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Year", "Total"])
    annual = df.groupby("Year", as_index=False).agg(population=("Total", "sum"))
    annual = annual.rename(columns={"Year": "year"}).astype({"year": int})
    return annual


def _maddison_pop(iso3: str) -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"] == iso3, ["year", "pop"]].dropna().copy()
    sub["population"] = sub["pop"].astype(float) * 1000.0
    sub["year"] = sub["year"].astype(int)
    return sub[["year", "population"]]


def build_isl_fertility_annual(write: bool = False) -> pd.DataFrame:
    births = _hmd_births()
    # Primary: HMD exposures (full 1838-2020 coverage)
    exp_pop = _hmd_exposures_pop()
    # Fallback: Maddison for any years beyond HMD exposure range
    mad_pop = _maddison_pop(ISO3)
    # Combine: HMD exposures first, fill any remaining gaps from Maddison
    pop = (exp_pop
           .merge(mad_pop.rename(columns={"population": "pop_mad"}),
                  on="year", how="outer")
           .sort_values("year"))
    pop["population"] = pop["population"].fillna(pop["pop_mad"])
    pop = pop[["year", "population"]].dropna(subset=["population"])

    df = births.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["iso3"] = ISO3
    df["source"] = f"HMD_{ISO3}_exposures_pop"
    df = df[["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]]
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_isl_fertility_annual(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}")
    print(f"log_cbr non-null: {df['log_cbr'].notna().sum()}")
    print("\nCBR sanity (per 1000):")
    print(df["cbr"].describe())
