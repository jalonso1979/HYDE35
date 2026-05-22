"""Build annual England fertility series 1541-2020 by splicing CamPOP + HFD.

Splice
------
1541-1837: CamPOP 26-parish baptisms scaled to a true national CBR using the
           Wrigley-Schofield 1981 published mean (~32 per 1000) as anchor.
           Source CSV produced by `paper4_shadow.build_campop_annual`.
1838-1937: GAP (CamPOP parish coverage collapses 1838+; no continuous source
           in this repo's data through to HFD).
1938-2022: HFD GBRTENW total live births / BoE Millennium A18 England &
           Wales population.

CamPOP coverage cutoff
----------------------
Annual CamPOP baptism counts hold steady at ~1000-1700 through 1837 then
collapse: 1838 -> 858, 1841 -> 583, 1842 -> 313, 1851 -> 129. This is a
known data artefact (parishes dropping out of the family-reconstitution
sample), not a real fertility collapse. Civil registration begins in
England in 1837, providing a natural cutoff. We treat 1838-1937 as the
industrial-era gap.

CamPOP scaling
--------------
`cbr_26par` in the CamPOP CSV is `baptisms / national_population * 1000` where
`baptisms` covers only 26 parishes (~0.5-1% of England) and `population` is
*national*. So `cbr_26par` ~0.17 per 1000 of national population. To recover a
true national CBR we anchor the era mean (1541-1837) to Wrigley-Schofield
1981's published England mean CBR of ~32 per 1000 (Wrigley & Schofield 1981,
Population History of England 1541-1871, Tables A3.1-A3.3, era mean
~31-33 per 1000). The per-year variation is preserved exactly; only the
multiplicative level is fixed.

Note
----
The 1838-1937 industrial-revolution gap is a Phase 1 limitation.
Phase 2 should plug this with ONS historical vital statistics or
Mitchell's British Historical Statistics.

Output
------
data/long_shadow_fertility/england_fertility_annual_1541_2020.parquet
Columns: year, births, population, cbr, log_cbr, source
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
HFD_BIRTHS = FERTILITY / "data" / "HFD" / "totbirthsRR.txt"
BOE_XLSX = BIGDATA / "data" / "wrigley_schofield" / "boe_millennium.xlsx"
OUT = (
    BIGDATA
    / "data"
    / "long_shadow_fertility"
    / "england_fertility_annual_1541_2020.parquet"
)

# BoE A18 column that gives a continuous England (or England & Wales)
# population in *thousands*, 1086-2016, anchored on Wrigley (1997) for the
# pre-Census era and on UK Census thereafter.
#
# Magnitudes verified at key years (in thousands):
#   1700: 5,196   1800:  8,620   1851: 16,820
#   1900: 30,253  1938: 38,749   1950: 41,421
#   2000: 49,233  2010: 52,642   2016: 55,219
#
# The plan-suggested column "Census estimates 1801+, retroplated with English
# population" turned out to hold a much smaller series (~350 in 1700, ~3,050 in
# 2010) — apparently a sub-component, not the headline population. The Wrigley
# column gives the canonical E&W magnitudes used throughout this paper.
BOE_A18_POP_COL = "Wrigley (1997) to 1841, Census thereafter"
BOE_A18_YEAR_COL = "Sources"  # first column header in the A18 sheet

# CamPOP coverage holds steady through 1837 then collapses. Civil registration
# begins in England in 1837 and our HFD source picks up in 1938 — the gap
# 1838-1937 is therefore the industrial-era gap flagged in the docstring.
CAMPOP_LAST_YEAR = 1837

# Wrigley-Schofield 1981 published mean CBR for England 1541-1837 used to
# anchor the level of the CamPOP-scaled series. See Wrigley & Schofield
# (1981), Tables A3.1-A3.3; era mean ~32 per 1000.
WS_1981_ANCHOR_CBR = 32.0


def _load_campop_raw() -> pd.DataFrame:
    df = pd.read_csv(CAMPOP_CSV)
    df = df.loc[
        df["year"].between(1541, CAMPOP_LAST_YEAR),
        ["year", "baptisms", "population", "cbr_26par"],
    ].copy()
    return df


def _load_campop() -> pd.DataFrame:
    """Return CamPOP 1541-1837 rows with births and a level-anchored cbr.

    The CamPOP CSV stores `cbr_26par = baptisms / national_population * 1000`
    where baptisms are 26-parish only. We rescale by a constant so the era
    mean matches the Wrigley-Schofield 1981 published value (~32 per 1000),
    preserving year-to-year variation.
    """
    df = _load_campop_raw()
    scale = WS_1981_ANCHOR_CBR / df["cbr_26par"].mean()
    df["cbr"] = df["cbr_26par"] * scale
    df = df.rename(columns={"baptisms": "births"})
    df["source"] = "CamPOP_26par_WS1981_anchored"
    return df[["year", "births", "population", "cbr", "source"]]


def _load_hfd_births() -> pd.DataFrame:
    df = pd.read_csv(HFD_BIRTHS, sep=r"\s+", skiprows=2, engine="python")
    df = df.loc[df["Code"] == "GBRTENW", ["Year", "Total"]]
    df = df.rename(columns={"Year": "year", "Total": "births"}).astype({"year": int})
    return df


def _load_boe_population() -> pd.DataFrame:
    """England/E&W population in absolute persons (converted from thousands).

    Source: BoE Millennium sheet A18, column BOE_A18_POP_COL.
    """
    raw = pd.read_excel(
        BOE_XLSX, sheet_name="A18. Population 1680+", skiprows=4
    )
    if BOE_A18_POP_COL not in raw.columns:
        raise KeyError(
            f"Expected BoE A18 column '{BOE_A18_POP_COL}' not found. "
            f"Available: {list(raw.columns)}"
        )
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(raw[BOE_A18_YEAR_COL], errors="coerce"),
            # A18 values are in thousands of persons → convert to absolute.
            "population_boe": pd.to_numeric(raw[BOE_A18_POP_COL], errors="coerce")
            * 1000.0,
        }
    )
    out = out.dropna(subset=["year"]).astype({"year": int})
    return out


def build_england_fertility_annual(write: bool = False) -> pd.DataFrame:
    """Splice CamPOP + HFD into a single annual England fertility table.

    Returns a DataFrame with columns
    ``year, births, population, cbr, log_cbr, source`` covering 1541-2022.
    Years 1838-1937 are an explicit gap (births/cbr/log_cbr NaN, population
    from BoE A18 where available, source = "GAP_industrial"). Years
    2017-2022 have HFD births but NaN population because the BoE
    Millennium A18 sheet only extends through 2016.
    """
    all_years = pd.DataFrame({"year": range(1541, 2023)})

    campop = _load_campop()
    hfd = _load_hfd_births()
    boe_pop = _load_boe_population()

    out = all_years.merge(campop, on="year", how="left")
    out = out.merge(hfd, on="year", how="left", suffixes=("", "_hfd"))
    out = out.merge(boe_pop, on="year", how="left")

    # HFD era 1938-2022: take births from HFD, population from BoE A18.
    hfd_mask = out["year"] >= 1938
    out.loc[hfd_mask, "births"] = out.loc[hfd_mask, "births_hfd"]
    out.loc[hfd_mask, "population"] = out.loc[hfd_mask, "population_boe"]
    out.loc[hfd_mask, "cbr"] = (
        out.loc[hfd_mask, "births"] / out.loc[hfd_mask, "population"] * 1000.0
    )
    out.loc[hfd_mask, "source"] = "HFD_GBRTENW_BoE_A18"

    # Industrial gap 1838-1937: births/cbr NaN, population from BoE.
    gap_mask = out["year"].between(CAMPOP_LAST_YEAR + 1, 1937)
    out.loc[gap_mask, "births"] = np.nan
    out.loc[gap_mask, "cbr"] = np.nan
    out.loc[gap_mask, "population"] = out.loc[gap_mask, "population_boe"]
    out.loc[gap_mask, "source"] = "GAP_industrial"

    out["log_cbr"] = np.log(out["cbr"])
    out = out[["year", "births", "population", "cbr", "log_cbr", "source"]]
    out = out.sort_values("year").reset_index(drop=True)

    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    df = build_england_fertility_annual(write=True)
    print(
        f"wrote {OUT} ({len(df)} rows, {df['year'].min()}-{df['year'].max()})"
    )
    print("Source counts:")
    print(df["source"].value_counts(dropna=False))
    print("\nCBR sanity (per 1000):")
    print(df["cbr"].describe())
