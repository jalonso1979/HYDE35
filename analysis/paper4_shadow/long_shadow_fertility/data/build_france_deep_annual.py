"""Deepen the France vital series back to 1740 by splicing Blayo (1975) onto HMD.

Mirrors the England CamPOP->HMD splice in ``build_england_fertility_annual.py``:
the pre-modern national reconstruction (Blayo for France, just as CamPOP/Wrigley-
Schofield is for England) supplies the early years; the HMD-based civil-
registration pipeline supplies the modern years.

Splice
------
    1740-1805 : Blayo (1975) "Mouvement naturel de la population francaise de
                1740 a 1829", *Population* 30(HS):71-122, ENSEMBLE (1954-territory)
                annual national births/deaths, in THOUSANDS. Per-decade-anchored
                annual reconstruction; the canonical pre-1806 France series.
    1806+     : existing HMD pipeline, UNCHANGED
                - fertility: france_fertility_annual.parquet (HMD FRATNP births)
                - mortality: country_mortality_annual.parquet, iso3==FRA
                             (HMD FRATNP Deaths_1x1)

Why 1806 is the HMD floor: the official HMD FRATNP series begins in 1806 (HMD
France methods protocol, mortality.org/FRATNP) because pre-1806 vital counts
come from Blayo's reconstruction rather than from the civil-registration input
files HMD ingests. Blayo's annual series is therefore a genuine extension *below*
HMD, not a duplication of it.

Population denominator (same source as the existing CDR builder: Maddison FRA)
-----------------------------------------------------------------------------
Maddison Project 2023 has France ``pop`` only at sparse benchmarks before 1820
(...1700=21,471k, then annual from 1820). For consistency with the rest of the
panel we use Maddison FRA pop, log-linearly interpolated to annual values across
the 1700->1820 benchmark gap (standard treatment of Maddison population between
benchmarks). This is the SAME source the existing CDR builder uses; we only add
the annual interpolation the pre-1820 gap requires.

Death under-registration (pre-1793 parish-register era) -- reported BOTH ways
-----------------------------------------------------------------------------
Pre-1792 French parish registers undercount deaths, concentrated among children
under 5 (Spagnoli 1997, *Social Science History* 22(4):425-461, citing Blayo
1975 and the INED reconstitution; the 1792 move to civil registration sharply
improved completeness). We expose two CDR variants:

    raw         : Blayo ENSEMBLE deaths, uncorrected.
    corrected   : pre-1793 deaths multiplied by PRE1793_DEATH_CORRECTION (default
                  1.10, i.e. +10%), a standard upward adjustment for under-
                  registration of infant/child deaths in the parish-register era
                  (Spagnoli 1997; Blayo 1975). Births are left RAW: the literature
                  treats the pre-1793 birth series as relatively complete, with a
                  definitional (not coverage) change at 1793.

The 1793 break (civil registration replaces parish registers, Sept 1792) is
flagged in the ``break_1793`` column.

Harmonization at the 1806 join -- reported BOTH ways
----------------------------------------------------
On the 1820-1829 overlap (both Blayo and HMD present) HMD = 0.964 x Blayo for
births and 0.975 x Blayo for deaths -- a small (~3%) level offset from
territorial/definitional differences. England anchors CamPOP to an EXTERNAL
benchmark (Wrigley-Schofield), not to its HMD join; the truest mirror here is to
use Blayo raw (it is itself the canonical external benchmark for pre-1806
France). We additionally expose a ``harmonized`` variant that scales Blayo to the
HMD level using the overlap ratios, so the join is seamless and the sensitivity
is transparent.

Outputs
-------
- data/long_shadow_fertility/france_fertility_annual.parquet   (OVERWRITES with
  deepened series; baseline backed up to /tmp/ls_baseline_p14 by the caller)
- data/long_shadow_fertility/country_mortality_annual.parquet  (deepened FRA rows
  spliced in; other 11 countries untouched)
- data/long_shadow_fertility/_deep_sources/france_deepened_1740_2008.csv (audit)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
BLAYO_CSV = ROOT / "_deep_sources" / "france_blayo_1740_1829.csv"
FERT_PARQUET = ROOT / "france_fertility_annual.parquet"          # HMD fertility (in/out)
MORT_PARQUET = ROOT / "country_mortality_annual.parquet"         # all-country mortality (in/out)
DEEP_CSV = ROOT / "_deep_sources" / "france_deepened_1740_2008.csv"

BLAYO_FIRST = 1740
BLAYO_LAST = 1805            # splice boundary: Blayo <= 1805, HMD >= 1806
HMD_FIRST = 1806
BREAK_YEAR = 1793           # parish-register -> civil-registration completeness break

# Pre-1793 death under-registration correction (Spagnoli 1997; Blayo 1975).
PRE1793_DEATH_CORRECTION = 1.10

# Blayo->HMD level ratios on the 1820-1829 overlap (computed once; see module docstring).
HARMONIZE_BIRTHS = 0.9639
HARMONIZE_DEATHS = 0.9753


# ---------------------------------------------------------------------------
# Population denominator (Maddison FRA, log-linear annual interpolation)
# ---------------------------------------------------------------------------
def maddison_fra_pop_annual(years: range | list[int]) -> pd.DataFrame:
    """Maddison FRA ``pop`` (persons), log-linearly interpolated to annual ``years``."""
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    fr = (raw.loc[raw["countrycode"] == "FRA", ["year", "pop"]]
          .dropna().astype({"year": int}).sort_values("year"))
    lo, hi = min(years), max(years)
    grid = pd.DataFrame({"year": range(min(lo, int(fr["year"].min())),
                                       max(hi, int(fr["year"].max())) + 1)})
    g = grid.merge(fr, on="year", how="left")
    # log-linear interpolation between benchmarks (standard for Maddison pop)
    g["population"] = np.exp(np.log(g["pop"]).interpolate(method="linear")) * 1000.0
    return g.loc[g["year"].isin(list(years)), ["year", "population"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Blayo pre-1806 series -> CBR / CDR
# ---------------------------------------------------------------------------
def blayo_france_rates(
    death_correction: bool = False,
    harmonize: bool = False,
) -> pd.DataFrame:
    """Blayo-derived France CBR/CDR for 1740-1805.

    Columns: year, iso3, births, deaths, population, cbr, cdr, log_cbr, log_cdr,
             break_1793, source.
    """
    bl = pd.read_csv(BLAYO_CSV)
    bl = bl.loc[bl["year"].between(BLAYO_FIRST, BLAYO_LAST),
                ["year", "births_ensemble", "deaths_ensemble"]].copy()
    bl["births"] = bl["births_ensemble"] * 1000.0          # thousands -> persons
    bl["deaths"] = bl["deaths_ensemble"] * 1000.0

    if harmonize:
        bl["births"] *= HARMONIZE_BIRTHS
        bl["deaths"] *= HARMONIZE_DEATHS

    bl["break_1793"] = (bl["year"] < BREAK_YEAR).astype(int)  # 1 = pre-break parish era
    if death_correction:
        pre = bl["year"] < BREAK_YEAR
        bl.loc[pre, "deaths"] *= PRE1793_DEATH_CORRECTION

    pop = maddison_fra_pop_annual(range(BLAYO_FIRST, BLAYO_LAST + 1))
    df = bl.merge(pop, on="year", how="left")
    df["cbr"] = df["births"] / df["population"] * 1000.0
    df["cdr"] = df["deaths"] / df["population"] * 1000.0
    df["log_cbr"] = np.log(df["cbr"])
    df["log_cdr"] = np.log(df["cdr"])
    df["iso3"] = "FRA"
    tag = []
    tag.append("Blayo1975_ensemble")
    tag.append("harmonized" if harmonize else "raw")
    tag.append("deathcorr1.10" if death_correction else "deathraw")
    tag.append("Maddison_pop_loglin")
    df["source"] = "_".join(tag)
    return df[["year", "iso3", "births", "deaths", "population",
               "cbr", "cdr", "log_cbr", "log_cdr", "break_1793", "source"]]


# ---------------------------------------------------------------------------
# Splice builders (write the parquets the panel/phases read)
# ---------------------------------------------------------------------------
def build_deepened_fertility(
    death_correction: bool = False,
    harmonize: bool = False,
    write: bool = False,
) -> pd.DataFrame:
    """France fertility = Blayo 1740-1805 + existing HMD parquet 1806+."""
    hmd = pd.read_parquet(FERT_PARQUET)
    hmd = hmd.loc[hmd["year"] >= HMD_FIRST].copy()           # keep 1806+ unchanged
    bl = blayo_france_rates(death_correction=death_correction, harmonize=harmonize)
    bl_f = bl[["year", "iso3", "births", "population", "cbr", "log_cbr", "source"]]
    out = (pd.concat([bl_f, hmd[bl_f.columns]], ignore_index=True)
           .sort_values("year").reset_index(drop=True))
    if write:
        out.to_parquet(FERT_PARQUET, index=False)
    return out


def build_deepened_mortality(
    death_correction: bool = False,
    harmonize: bool = False,
    write: bool = False,
) -> pd.DataFrame:
    """All-country mortality parquet with FRA rows deepened to 1740 (others intact)."""
    mort = pd.read_parquet(MORT_PARQUET)
    other = mort.loc[mort["iso3"] != "FRA"].copy()
    fra_hmd = mort.loc[(mort["iso3"] == "FRA") & (mort["year"] >= HMD_FIRST)].copy()
    bl = blayo_france_rates(death_correction=death_correction, harmonize=harmonize)
    bl_m = bl[["iso3", "year", "deaths", "population", "cdr", "log_cdr", "source"]]
    fra = pd.concat([bl_m, fra_hmd[bl_m.columns]], ignore_index=True)
    out = (pd.concat([other[bl_m.columns], fra], ignore_index=True)
           .sort_values(["iso3", "year"]).reset_index(drop=True))
    if write:
        out.to_parquet(MORT_PARQUET, index=False)
    return out


def write_deep_csv(death_correction_for_csv: bool = False) -> Path:
    """Persist a full audit CSV: Blayo 1740-1805 + HMD-derived 1806-2008, both
    death-correction CDR variants side by side."""
    fert_raw = build_deepened_fertility(death_correction=False, harmonize=False)
    bl_raw = blayo_france_rates(death_correction=False, harmonize=False)
    bl_cor = blayo_france_rates(death_correction=True, harmonize=False)

    # mortality 1806+ (HMD) for CDR continuation in the CSV
    mort = pd.read_parquet(MORT_PARQUET)
    fra_hmd = mort.loc[(mort["iso3"] == "FRA") & (mort["year"].between(HMD_FIRST, 2008)),
                       ["year", "deaths", "population", "cdr", "log_cdr"]].copy()

    fert = fert_raw.loc[fert_raw["year"] <= 2008,
                        ["year", "births", "population", "cbr", "log_cbr", "source"]].copy()
    cdr_pre = bl_raw[["year", "cdr"]].rename(columns={"cdr": "cdr_raw"}).merge(
        bl_cor[["year", "cdr", "break_1793"]].rename(columns={"cdr": "cdr_corrected"}),
        on="year",
    )
    out = fert.merge(cdr_pre, on="year", how="left")
    hmd_cdr = fra_hmd.rename(columns={"cdr": "cdr_hmd"})[["year", "cdr_hmd"]]
    out = out.merge(hmd_cdr, on="year", how="left")
    # unified cdr column: pre-1806 raw Blayo, 1806+ HMD
    out["cdr"] = out["cdr_raw"].where(out["year"] < HMD_FIRST, out["cdr_hmd"])
    out["iso3"] = "FRA"
    out = out[["iso3", "year", "births", "population", "cbr", "log_cbr",
               "cdr", "cdr_raw", "cdr_corrected", "cdr_hmd", "break_1793", "source"]]
    DEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEEP_CSV, index=False)
    return DEEP_CSV


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--death-correction", action="store_true",
                    help="apply +10%% pre-1793 death under-registration correction")
    ap.add_argument("--harmonize", action="store_true",
                    help="scale Blayo to HMD level using 1820-1829 overlap ratios")
    ap.add_argument("--write", action="store_true",
                    help="overwrite france_fertility_annual + country_mortality_annual parquets")
    args = ap.parse_args()

    fert = build_deepened_fertility(args.death_correction, args.harmonize, write=args.write)
    mort = build_deepened_mortality(args.death_correction, args.harmonize, write=args.write)
    csv_path = write_deep_csv()

    bl = blayo_france_rates(args.death_correction, args.harmonize)
    print(f"death_correction={args.death_correction}  harmonize={args.harmonize}  write={args.write}")
    print(f"Blayo 1740-1805 France-years added: {len(bl)}")
    print("  CBR mean (per 1000):  %.2f   [sanity target 38-40]" % bl["cbr"].mean())
    print("  CDR mean (per 1000):  %.2f   [sanity target 33-38]" % bl["cdr"].mean())
    print("  CBR range: %.1f - %.1f ; CDR range: %.1f - %.1f"
          % (bl["cbr"].min(), bl["cbr"].max(), bl["cdr"].min(), bl["cdr"].max()))
    print(f"  pre-1793 CDR mean={bl.loc[bl['break_1793']==1,'cdr'].mean():.2f}  "
          f"post-1793(<=1805) CDR mean={bl.loc[bl['break_1793']==0,'cdr'].mean():.2f}")
    print(f"Deepened fertility parquet: {len(fert)} rows, {fert['year'].min()}-{fert['year'].max()}")
    print(f"Deepened mortality parquet FRA: "
          f"{mort.loc[mort['iso3']=='FRA','year'].min()}-{mort.loc[mort['iso3']=='FRA','year'].max()}")
    print(f"Audit CSV: {csv_path}")
