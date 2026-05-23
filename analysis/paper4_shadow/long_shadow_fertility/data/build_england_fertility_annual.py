"""Build annual England fertility series 1541-2020 — Phase 5 v2 (HMD splice).

Splice strategy:
    1541-1837: CamPOP 26-parish baptisms (anchored to W-S 1981 mean)
    1838-1840: GAP_civil_reg (3 years; civil registration started Jul 1837;
               HMD GBRTENW.Births.txt coverage starts 1841)
    1841-2020: HMD GBRTENW.Births.txt total births / BoE Millennium A18 population

Phase 5 fix: replaces HFD totbirthsRR.txt (starts 1938) with HMD GBRTENW
(starts 1841). Gap shrinks from 100 years to 3 years.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

BIGDATA = Path("/Volumes/BIGDATA/HYDE35/analysis")
FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
CAMPOP_CSV = BIGDATA / "data" / "wrigley_schofield" / "campop_england_annual.csv"
HMD_BIRTHS = FERTILITY / "data" / "mortality" / "births" / "GBRTENW.Births.txt"
BOE_XLSX = BIGDATA / "data" / "wrigley_schofield" / "boe_millennium.xlsx"
OUT = BIGDATA / "data" / "long_shadow_fertility" / "england_fertility_annual_1541_2020.parquet"

BOE_A18_POP_COL = "Wrigley (1997) to 1841, Census thereafter"
WS_1981_ANCHOR_CBR = 32.0
CAMPOP_LAST_YEAR = 1837
HMD_START_YEAR = 1841


def _load_campop() -> pd.DataFrame:
    df = pd.read_csv(CAMPOP_CSV)
    df = df.loc[df["year"].between(1541, CAMPOP_LAST_YEAR), ["year", "baptisms", "population", "cbr_26par"]]
    df = df.rename(columns={"baptisms": "births", "cbr_26par": "cbr_raw"})
    scale = WS_1981_ANCHOR_CBR / df["cbr_raw"].mean()
    df["cbr"] = df["cbr_raw"] * scale
    df = df[["year", "births", "population", "cbr"]]
    df["source"] = "CamPOP_26par_WS1981_anchored"
    return df


def _load_hmd_gbrtenw() -> pd.DataFrame:
    df = pd.read_csv(HMD_BIRTHS, sep=r"\s+", skiprows=2, engine="python")
    df = df.rename(columns={"Year": "year", "Total": "births"})
    df = df[["year", "births"]].astype({"year": int, "births": float})
    df = df.loc[df["year"].between(HMD_START_YEAR, 2022)]
    return df


def _load_boe_population() -> pd.DataFrame:
    raw = pd.read_excel(BOE_XLSX, sheet_name="A18. Population 1680+", skiprows=4)
    year_col = raw.columns[0]
    if BOE_A18_POP_COL in raw.columns:
        pop_col = BOE_A18_POP_COL
    else:
        matches = [c for c in raw.columns if "Wrigley" in str(c) and "Census" in str(c)]
        if not matches:
            raise KeyError(f"BoE A18 column {BOE_A18_POP_COL!r} missing; cols: {list(raw.columns)}")
        pop_col = matches[0]
    out = pd.DataFrame({
        "year": pd.to_numeric(raw[year_col], errors="coerce"),
        "population_boe": pd.to_numeric(raw[pop_col], errors="coerce"),
    }).dropna().astype({"year": int})
    return out


def build_england_fertility_annual(write: bool = False) -> pd.DataFrame:
    all_years = pd.DataFrame({"year": range(1541, 2023)})
    campop = _load_campop()
    hmd = _load_hmd_gbrtenw()
    boe_pop = _load_boe_population()

    out = all_years.merge(campop, on="year", how="left")
    out = out.merge(hmd, on="year", how="left", suffixes=("", "_hmd"))
    out = out.merge(boe_pop, on="year", how="left")

    hmd_mask = (out["year"] >= HMD_START_YEAR)
    out.loc[hmd_mask, "births"] = out.loc[hmd_mask, "births_hmd"]
    out.loc[hmd_mask, "population"] = out.loc[hmd_mask, "population_boe"] * 1000
    out.loc[hmd_mask, "cbr"] = out.loc[hmd_mask, "births"] / out.loc[hmd_mask, "population"] * 1000
    out.loc[hmd_mask, "source"] = "HMD_GBRTENW_BoE_A18"

    gap_mask = out["year"].between(1838, 1840)
    out.loc[gap_mask, "source"] = "GAP_civil_reg"
    out.loc[gap_mask, "population"] = out.loc[gap_mask, "population_boe"] * 1000

    out["log_cbr"] = np.log(out["cbr"])
    out = out[["year", "births", "population", "cbr", "log_cbr", "source"]]
    out = out.sort_values("year").reset_index(drop=True)

    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    df = build_england_fertility_annual(write=True)
    print(f"wrote {OUT} ({len(df)} rows)")
    print("Source counts:")
    print(df["source"].value_counts())
