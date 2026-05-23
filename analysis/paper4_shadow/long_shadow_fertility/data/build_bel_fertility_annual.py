"""Belgium country-year fertility from HMD BEL + Maddison population (1840+)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
HMD_BEL = FERTILITY / "data" / "mortality" / "births" / "BEL.Births.txt"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "bel_fertility_annual.parquet")


def _hmd_bel_births() -> pd.DataFrame:
    df = pd.read_csv(HMD_BEL, sep=r"\s+", skiprows=2, engine="python")
    df = df.rename(columns={"Year": "year", "Total": "births"})
    return df[["year", "births"]].astype({"year": int, "births": float})


def _maddison_pop_bel() -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"] == "BEL", ["year", "pop"]].dropna()
    sub["population"] = sub["pop"].astype(float) * 1000.0
    return sub[["year", "population"]]


def build_bel_fertility_annual(write: bool = False) -> pd.DataFrame:
    births = _hmd_bel_births()
    pop = _maddison_pop_bel()
    df = births.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["iso3"] = "BEL"
    df["source"] = "HMD_BEL_Maddison_pop"
    df = df[["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]]
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_bel_fertility_annual(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}")
