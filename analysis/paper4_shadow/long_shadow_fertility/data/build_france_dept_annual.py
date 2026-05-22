"""France dept-year fertility + ModE-RA climate panel (Phase 2 Task 5).

Cassini historical dept-level births merged with ModE-RA cropland-weighted
annual climate at dept centroids. Output keyed by (dep, year) with iso3="FRA"
appended for unified-panel compatibility.

Source coverage
---------------
- ModE-RA dept-annual (1421-2008, 96 depts including Corsica split 2A/2B):
    columns dep, year, temp_mean, temp_anom_mean, precip_anom_mean.
    Provides the panel spine.
- Cassini dept-level births (1851-1897, 91 historical departments coded as
    integer 1..91 with Corsica = 20): columns region, year, births.
    Cassini coverage is sparse pre-1885 (single-year snapshots for 1853,
    1871) and continuous 1885-1897; many years between 1898 and 2008 have
    no birth observation in this builder.

Department code harmonisation
-----------------------------
Cassini stores `region` as a 1-91 integer. ModE-RA stores `dep` as a 2-char
zero-padded string ("01".."95", with Corsica split into "2A"/"2B" instead of
"20"). We map Cassini integers to zero-padded strings ("1" -> "01"). Region
20 in Cassini corresponds historically to Corsica before the 1975 split into
2A/2B; we route Cassini "20" to "2A" so the Corsican rows merge into one of
the ModE-RA Corsica entries (rather than dropping). This is an
approximation; the alternative would be to leave Corsica unmatched.
Departments 92-95 (post-1968 Paris-ring divisions) and 2B exist in ModE-RA
but never in Cassini, so they appear in the output with NaN births.

Output
------
analysis/data/long_shadow_fertility/france_dept_fertility_annual.parquet

Columns
-------
- dep       : zero-padded 2-char dept code matching ModE-RA convention
- year      : int
- iso3      : "FRA"
- births    : Cassini total live births (NaN where Cassini has no row)
- log_cbr   : log(births) — Cassini provides no population denominator at
              dept level, so we use log-births as a fertility-level proxy
              (variable name kept `log_cbr` for cross-country panel
              consistency); NaN where births is NaN or zero.
- t_growing : ModE-RA annual temperature anomaly (renamed from
              temp_anom_mean); the "growing" suffix matches the convention
              used in other long_shadow_fertility builders even though the
              underlying ModE-RA aggregate is calendar-year, not
              growing-season.
- p_growing : ModE-RA annual precipitation anomaly (renamed from
              precip_anom_mean).
- temp_mean : ModE-RA absolute mean annual temperature (degrees Celsius),
              passed through unchanged for diagnostic use.
- source    : "Cassini_ModE-RA" where both sources present, "ModE-RA_only"
              otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
CASSINI = FERTILITY / "data" / "PROCESSED" / "cassini_fra_dept_births_historical.csv"
MODERA = FERTILITY / "data" / "PROCESSED" / "modera_france_dept_annual_1421_2008.csv"
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "france_dept_fertility_annual.parquet"
)


def _load_cassini() -> pd.DataFrame:
    """Cassini dept-year births with region remapped to ModE-RA `dep` string."""
    df = pd.read_csv(CASSINI)
    df = df.loc[:, ["region", "year", "births"]].copy()
    df["region"] = df["region"].astype(int)
    df["dep"] = df["region"].apply(lambda r: "2A" if r == 20 else f"{r:02d}")
    df = df.drop(columns="region")
    df["year"] = df["year"].astype(int)
    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    return df[["dep", "year", "births"]]


def _load_modera() -> pd.DataFrame:
    """ModE-RA dept-annual climate (1421-2008) renamed to panel convention."""
    df = pd.read_csv(MODERA, dtype={"dep": str})
    df = df.rename(
        columns={
            "temp_anom_mean": "t_growing",
            "precip_anom_mean": "p_growing",
        }
    )
    keep = [c for c in ("dep", "year", "t_growing", "p_growing", "temp_mean") if c in df.columns]
    return df[keep]


def build_france_dept_annual(write: bool = False) -> pd.DataFrame:
    """Return France dept-year fertility + climate panel keyed by (dep, year)."""
    clim = _load_modera()
    births = _load_cassini()

    df = clim.merge(births, on=["dep", "year"], how="left")

    df["log_cbr"] = np.log(df["births"].where(df["births"] > 0))
    df["iso3"] = "FRA"
    df["source"] = np.where(df["births"].notna(), "Cassini_ModE-RA", "ModE-RA_only")

    cols = ["dep", "year", "iso3", "births", "log_cbr",
            "t_growing", "p_growing", "temp_mean", "source"]
    df = df[cols].sort_values(["dep", "year"]).reset_index(drop=True)

    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_france_dept_annual(write=True)
    print(
        f"wrote {OUT}: {len(df)} rows, "
        f"{df['dep'].nunique()} depts, "
        f"{df['year'].min()}-{df['year'].max()}"
    )
    print("\nSource counts:")
    print(df["source"].value_counts(dropna=False))
    print("\nbirths sanity (Cassini era only):")
    print(df["births"].dropna().describe())
