"""Build GS-decomposed climate extension to the horserace panel.

Reads:
  analysis/data/deep_determinants_horserace.parquet  (existing panel)
  analysis/data/country_seasonality_gs_preindustrial.parquet
    (built by long_shadow Task 1/2; has cropw-weighted GS columns)

Merges on iso3 and writes:
  analysis/data/deep_determinants_horserace_gs.parquet

New columns added:
  sigma_v_T_gs_pre1750_cropw    — GS-restricted T volatility
  sigma_v_T_nongs_pre1750_cropw — non-GS T volatility
  sigma_v_P_gs_pre1750_cropw    — GS P volatility
  n_gs_months_cropw             — number of growing-season months (diagnostic)

The existing panel is left unmodified; this file is the input for the
growing-season Shapley exercise (exercise1_climate_gs_decomposition.py).
"""
from pathlib import Path

import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
GS_DATA = ROOT / "analysis/data/country_seasonality_gs_preindustrial.parquet"
OUT = ROOT / "analysis/data/deep_determinants_horserace_gs.parquet"

GS_COLS = [
    "iso3",
    "sigma_v_T_gs_pre1750_cropw",
    "sigma_v_T_nongs_pre1750_cropw",
    "sigma_v_P_gs_pre1750_cropw",
    "n_gs_months_cropw",
]


def main() -> None:
    df = pd.read_parquet(PANEL)
    gs = pd.read_parquet(GS_DATA)[GS_COLS]

    print(f"Horserace panel: {len(df)} rows, {df.shape[1]} columns")
    print(f"GS data: {len(gs)} rows")
    print(f"GS non-null counts:\n{gs.notnull().sum()}")

    # Check overlap before merge
    overlap = set(df["iso3"]) & set(gs["iso3"])
    only_hr = set(df["iso3"]) - set(gs["iso3"])
    only_gs = set(gs["iso3"]) - set(df["iso3"])
    print(f"\niso3 overlap: {len(overlap)} countries")
    if only_hr:
        print(f"Only in horserace (will have NaN GS columns): {sorted(only_hr)}")
    if only_gs:
        print(f"Only in GS data (will be dropped): {sorted(only_gs)}")

    merged = df.merge(gs, on="iso3", how="left")
    assert len(merged) == len(df), "Row count changed after merge — check iso3 uniqueness"

    # Diagnostic: how many non-null GS observations in the merged panel?
    for col in GS_COLS[1:]:
        n = merged[col].notnull().sum()
        print(f"  {col}: {n} non-null of {len(merged)}")

    merged.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(merged)} rows, {merged.shape[1]} columns)")


if __name__ == "__main__":
    main()
