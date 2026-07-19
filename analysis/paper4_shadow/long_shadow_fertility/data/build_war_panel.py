"""Country-year war indicator + log fatalities from Brecke conflict catalogs.

Output
------
analysis/data/long_shadow_fertility/war_panel_country_year.parquet
Columns: iso3, year, war_active, log_war_fatalities

Coverage: 1400-2022 for GBR, FRA, ITA, SWE (panel countries), filled with
zeros where no Brecke conflict mentions the country.

Sources
-------
- Conflict-Catalog-18-vars.xlsx  : Brecke post-1400 global catalog. Columns
  include Common Name, Name, StartYear, EndYear, MilFatalities,
  TotalFatalities. The Name field encodes participants ("England-Scotland,
  1400-02", "France, Britain-Germany (Cameroon), 1914-16", etc.). Major
  20th-century wars use 3-letter abbreviations in one summary row
  ("Ger, Ita, Hung-Pol, Brit, Fra, Nor, ..." for WWII).
- Brecke-Pre-1400-European-Conflicts.xlsx : Pre-1400 European catalog with
  columns Conflict, StartYear, EndYear, Fatalities.

Approach
--------
1. Parse both catalogs. Build a participant list per conflict row by
   keyword-matching the conflict description against ISO_KEYWORDS (full
   country names AND short tokens such as "fra"/"brit"/"ita").
2. Special-case the two World War summary rows ("First World War" and
   "Second World War in Europe") to attribute participation to all four
   panel countries across the full conflict span, since the Name field of
   WWI omits country lists and WWII uses ambiguous abbreviations.
3. Aggregate to (iso3, year): war_active = 1 if any conflict touched the
   country that year, fatalities summed (then log1p).
4. Outer-join with a full (iso3, year) grid 1400-2022; missing -> 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis")
BRECKE_POST = ROOT / "data" / "brecke" / "Conflict-Catalog-18-vars.xlsx"
BRECKE_PRE = ROOT / "data" / "brecke" / "Brecke-Pre-1400-European-Conflicts.xlsx"
OUT = ROOT / "data" / "long_shadow_fertility" / "war_panel_country_year.parquet"

COUNTRIES = ["GBR", "FRA", "ITA", "SWE"]
YEAR_MIN, YEAR_MAX = 1400, 2022

# Keyword tokens (matched against lowercased conflict description). Each token
# is matched as a whole word using a regex word boundary, except where the
# token already contains hyphens or other punctuation handled directly.
ISO_KEYWORDS = {
    "GBR": (
        "england", "english", "scotland", "scottish", "wales", "welsh",
        "britain", "british", "britian", "brit", "uk", "u.k.",
    ),
    "FRA": ("france", "french", "fra"),
    "ITA": (
        "italy", "italian", "ita", "papal", "venice", "venetian", "florence",
        "florentine", "sicily", "sicilian", "naples", "neapolitan", "milan",
        "milanese", "sardinia", "sardinian", "tuscany", "tuscan", "rome",
        "roman", "savoy", "savoyard", "genoa", "genoese", "modena", "parma",
    ),
    "SWE": ("sweden", "swedish", "swede", "swe"),
}

# Compile patterns with word boundaries so "fra" does not match "fragile" and
# "uk" does not match "uksunja", etc.
_PATTERNS = {
    iso: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b")
    for iso, kws in ISO_KEYWORDS.items()
}

# Two large 20th-century catalog rows in which the Brecke "Name" field either
# omits country names entirely ("First World War, 1914-18") or uses ambiguous
# 3-letter abbreviations. We treat these as touching all four panel countries
# across the conflict span.
WORLD_WAR_FORCED = {
    "first world war": ("GBR", "FRA", "ITA"),  # SWE neutral in WWI
    "second world war": ("GBR", "FRA", "ITA"),  # SWE neutral in WWII
}


def _row_to_iso3(description: str) -> set[str]:
    """Return ISO3 codes whose keywords appear as whole words in description."""
    if not isinstance(description, str):
        return set()
    text = description.lower()
    return {iso for iso, pat in _PATTERNS.items() if pat.search(text)}


def _force_world_war(common_name: str, name: str) -> set[str] | None:
    """Return the forced ISO3 set for a WWI/WWII summary row, else None."""
    blob = f"{common_name or ''} {name or ''}".lower()
    for key, isos in WORLD_WAR_FORCED.items():
        if key in blob:
            return set(isos)
    return None


def _expand_post1400(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the post-1400 Brecke catalog into (iso3, year, fatalities) rows."""
    rows: list[dict] = []
    for _, r in df.iterrows():
        y0 = r.get("StartYear")
        y1 = r.get("EndYear", y0)
        if pd.isna(y0):
            continue
        if pd.isna(y1):
            y1 = y0
        try:
            y0i = int(y0)
            y1i = int(y1)
        except (TypeError, ValueError):
            continue
        if y1i < y0i:
            y0i, y1i = y1i, y0i

        common = r.get("Common Name")
        name = r.get("Name")

        forced = _force_world_war(common, name)
        if forced is not None:
            isos = forced
        else:
            # Try Common Name first (often cleaner), then Name; union both.
            isos = _row_to_iso3(common) | _row_to_iso3(name)

        if not isos:
            continue

        # Fatalities: prefer TotalFatalities, fall back to MilFatalities.
        fat = r.get("TotalFatalities")
        if pd.isna(fat):
            fat = r.get("MilFatalities")
        try:
            fat_val = float(fat) if pd.notna(fat) else 0.0
        except (TypeError, ValueError):
            fat_val = 0.0

        years = list(range(y0i, y1i + 1))
        # Spread fatalities evenly across conflict years so a 4-year war
        # doesn't quadruple-count fatalities at the country-year level.
        n_years = len(years)
        per_year_fat = fat_val / n_years if n_years else 0.0

        for iso in isos:
            for y in years:
                rows.append({"iso3": iso, "year": y, "fatalities": per_year_fat})
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["iso3", "year", "fatalities"])
    )


def _expand_pre1400(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the pre-1400 Brecke catalog (single Conflict-text column)."""
    rows: list[dict] = []
    for _, r in df.iterrows():
        y0 = r.get("StartYear")
        y1 = r.get("EndYear", y0)
        if pd.isna(y0):
            continue
        if pd.isna(y1):
            y1 = y0
        try:
            y0i = int(y0)
            y1i = int(y1)
        except (TypeError, ValueError):
            continue
        if y1i < y0i:
            y0i, y1i = y1i, y0i

        isos = _row_to_iso3(r.get("Conflict", ""))
        if not isos:
            continue

        fat = r.get("Fatalities")
        try:
            fat_val = float(fat) if pd.notna(fat) else 0.0
        except (TypeError, ValueError):
            fat_val = 0.0

        years = list(range(y0i, y1i + 1))
        n_years = len(years)
        per_year_fat = fat_val / n_years if n_years else 0.0

        for iso in isos:
            for y in years:
                rows.append({"iso3": iso, "year": y, "fatalities": per_year_fat})
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["iso3", "year", "fatalities"])
    )


def build_war_panel(write: bool = False) -> pd.DataFrame:
    """Build the (iso3, year, war_active, log_war_fatalities) panel."""
    post = pd.read_excel(BRECKE_POST)
    pre = pd.read_excel(BRECKE_PRE)

    conflicts_post = _expand_post1400(post)
    conflicts_pre = _expand_pre1400(pre)
    conflicts = pd.concat([conflicts_post, conflicts_pre], ignore_index=True)

    # Restrict to panel countries and target year range.
    conflicts = conflicts[
        conflicts["iso3"].isin(COUNTRIES)
        & (conflicts["year"] >= YEAR_MIN)
        & (conflicts["year"] <= YEAR_MAX)
    ]

    agg = (
        conflicts.groupby(["iso3", "year"], as_index=False)
        .agg(fatalities=("fatalities", "sum"))
    )
    agg["war_active"] = 1

    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    grid = pd.DataFrame(
        [(iso, int(y)) for iso in COUNTRIES for y in years],
        columns=["iso3", "year"],
    )

    df = grid.merge(agg, on=["iso3", "year"], how="left")
    df["war_active"] = df["war_active"].fillna(0).astype(int)
    df["fatalities"] = df["fatalities"].fillna(0.0)
    df["log_war_fatalities"] = np.log1p(df["fatalities"])
    df = df[["iso3", "year", "war_active", "log_war_fatalities"]]
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_war_panel(write=True)
    print(
        f"wrote {OUT}: {len(df)} rows, "
        f"{df['war_active'].sum()} war-years across "
        f"{df['iso3'].nunique()} countries"
    )
    print("\nWar-years per country (1400-2022):")
    print(df.groupby("iso3")["war_active"].sum().to_string())
    print("\nlog_war_fatalities summary (war_active==1 only):")
    print(df.loc[df["war_active"] == 1, "log_war_fatalities"].describe())
