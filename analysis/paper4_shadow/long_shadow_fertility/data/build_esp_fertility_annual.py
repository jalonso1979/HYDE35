"""Spain country-year fertility from HMD ESP + Maddison population (1908+)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
HMD_ESP = FERTILITY / "data" / "mortality" / "births" / "ESP.Births.txt"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "esp_fertility_annual.parquet")


def _hmd_esp_births() -> pd.DataFrame:
    df = pd.read_csv(HMD_ESP, sep=r"\s+", skiprows=2, engine="python")
    df = df.rename(columns={"Year": "year", "Total": "births"})
    return df[["year", "births"]].astype({"year": int, "births": float})


def _maddison_pop_esp() -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"] == "ESP", ["year", "pop"]].dropna()
    sub["population"] = sub["pop"].astype(float) * 1000.0
    return sub[["year", "population"]]


def build_esp_fertility_annual(write: bool = False) -> pd.DataFrame:
    births = _hmd_esp_births()
    pop = _maddison_pop_esp()
    df = births.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["iso3"] = "ESP"
    df["source"] = "HMD_ESP_Maddison_pop"
    df = df[["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]]
    df = df.sort_values("year").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_esp_fertility_annual(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['year'].min()}-{df['year'].max()}")
