"""ModE-RA + ERA5 spliced country-year climate panel (Phase 9 Pillar A3)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.build_era5_country_annual_v2 import (
    build_era5_country_annual_v2,
)

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
MODERA_PATH = ROOT / "country_climate_annual.parquet"
OUT_PATH = ROOT / "country_climate_spliced.parquet"
SPLICE_YEAR = 1950
OVERLAP = (1950, 2008)


def build_country_climate_spliced(write: bool = False) -> pd.DataFrame:
    """ModE-RA + ERA5 spliced country-year climate panel (Phase 9 Pillar A3).

    Layout (data-driven — ERA5 is used wherever the v2 builder has data):
    - 1421-1949: ModE-RA
    - 1950-2025: ERA5 v2 (mean-shifted to ModE-RA's 1950-2008 overlap
      baseline). With the raw archive complete (July 2026) the v2 panel
      covers all of 1950-2025, so the former 1968-2008 ModE-RA gap-fill
      branch below no longer contributes rows; it is kept as a fallback for
      partial rebuilds.

    Output schema matches country_climate_annual.parquet for drop-in use.
    """
    modera = pd.read_parquet(MODERA_PATH)
    era5 = build_era5_country_annual_v2(write=False)

    pre = modera.loc[modera["year"] < SPLICE_YEAR, ["iso3", "year", "t_growing", "p_growing"]].copy()
    pre["source"] = "ModE-RA"

    # ERA5 window: 1950 onward, but only where ERA5 v2 has data
    era5_years_per_iso = {iso: set(sub["year"].tolist()) for iso, sub in era5.groupby("iso3")}

    post_era5_parts = []
    for iso, sub in era5.groupby("iso3"):
        sub = sub.loc[sub["year"] >= SPLICE_YEAR].copy()
        ovp_modera = modera.loc[
            (modera["iso3"] == iso) & modera["year"].between(*OVERLAP),
            ["year", "t_growing", "p_growing"]
        ].set_index("year")
        ovp_era5 = sub.loc[sub["year"].between(*OVERLAP),
                            ["year", "t_growing_era5", "p_growing_era5"]].set_index("year")
        common = ovp_modera.index.intersection(ovp_era5.index)
        if len(common) >= 10:
            t_shift = float(ovp_modera.loc[common, "t_growing"].mean()
                            - ovp_era5.loc[common, "t_growing_era5"].mean())
            p_shift = float(ovp_modera.loc[common, "p_growing"].mean()
                            - ovp_era5.loc[common, "p_growing_era5"].mean())
        else:
            t_shift, p_shift = 0.0, 0.0
        sub["t_growing"] = sub["t_growing_era5"] + t_shift
        sub["p_growing"] = sub["p_growing_era5"] + p_shift
        sub["source"] = "ERA5_v2_shifted"
        post_era5_parts.append(sub[["iso3", "year", "t_growing", "p_growing", "source"]])
    post_era5 = pd.concat(post_era5_parts, ignore_index=True) if post_era5_parts else pd.DataFrame(columns=["iso3", "year", "t_growing", "p_growing", "source"])

    # ModE-RA tail: years >= SPLICE_YEAR that ERA5 v2 doesn't cover
    post_modera_rows = []
    for iso, sub in modera.groupby("iso3"):
        era5_yrs = era5_years_per_iso.get(iso, set())
        gap = sub.loc[(sub["year"] >= SPLICE_YEAR) & (~sub["year"].isin(era5_yrs)),
                       ["iso3", "year", "t_growing", "p_growing"]].copy()
        gap["source"] = "ModE-RA"
        post_modera_rows.append(gap)
    post_modera = pd.concat(post_modera_rows, ignore_index=True) if post_modera_rows else pd.DataFrame(columns=["iso3", "year", "t_growing", "p_growing", "source"])

    cols = ["iso3", "year", "t_growing", "p_growing", "source"]
    spliced = pd.concat([pre[cols], post_era5[cols], post_modera[cols]], ignore_index=True)
    spliced = (spliced.sort_values(["iso3", "year"])
                .drop_duplicates(["iso3", "year"], keep="first").reset_index(drop=True))

    if write:
        spliced.to_parquet(OUT_PATH, index=False)
    return spliced


if __name__ == "__main__":
    df = build_country_climate_spliced(write=True)
    print(f"wrote {OUT_PATH}: {len(df)} rows; "
          f"sources: {df['source'].value_counts().to_dict()}")
