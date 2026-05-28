"""Annual all-cause CDR per country, summed across ages from HMD Deaths_1x1."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
DEATHS_DIR = FERTILITY / "data" / "mortality" / "deaths" / "Deaths_1x1"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "country_mortality_annual.parquet")

ISO_MAP = {
    "GBRTENW": "GBR", "FRATNP": "FRA", "ITA": "ITA", "SWE": "SWE",
    "BEL": "BEL", "NLD": "NLD", "ESP": "ESP",
    "NOR": "NOR", "DNK": "DNK", "FIN": "FIN", "ISL": "ISL", "CHE": "CHE",
}


def _hmd_deaths_one(hmd_code: str) -> pd.DataFrame:
    path = DEATHS_DIR / f"{hmd_code}.Deaths_1x1.txt"
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, engine="python")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Year", "Total"])
    annual = df.groupby("Year", as_index=False).agg(deaths=("Total", "sum"))
    annual = annual.rename(columns={"Year": "year"}).astype({"year": int})
    return annual


def _maddison_pop(iso3: str) -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"] == iso3, ["year", "pop"]].dropna()
    sub["population"] = sub["pop"].astype(float) * 1000.0
    return sub[["year", "population"]]


def build_country_mortality_annual(write: bool = False) -> pd.DataFrame:
    parts = []
    for hmd_code, iso3 in ISO_MAP.items():
        deaths = _hmd_deaths_one(hmd_code)
        pop = _maddison_pop(iso3)
        d = deaths.merge(pop, on="year", how="left")
        d["iso3"] = iso3
        d["source"] = f"HMD_{hmd_code}_Maddison_pop"
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["cdr"] = df["deaths"] / df["population"] * 1000.0
    df["log_cdr"] = np.log(df["cdr"])
    df = df[["iso3", "year", "deaths", "population", "cdr", "log_cdr", "source"]]
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_country_mortality_annual(write=True)
    print(f"wrote {OUT}: {len(df)} rows; CDR mean per iso: {df.groupby('iso3')['cdr'].mean().to_dict()}")
