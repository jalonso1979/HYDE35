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
    modera = pd.read_parquet(MODERA_PATH)
    era5 = build_era5_country_annual_v2(write=False)

    pre = modera.loc[modera["year"] < SPLICE_YEAR, ["iso3", "year", "t_growing", "p_growing"]].copy()
    pre["source"] = "ModE-RA"

    post_parts = []
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
        post_parts.append(sub[["iso3", "year", "t_growing", "p_growing", "source"]])
    post = pd.concat(post_parts, ignore_index=True)

    cols = ["iso3", "year", "t_growing", "p_growing", "source"]
    spliced = pd.concat([pre[cols], post[cols]], ignore_index=True)
    spliced = (spliced.sort_values(["iso3", "year"])
                .drop_duplicates(["iso3", "year"]).reset_index(drop=True))

    if write:
        spliced.to_parquet(OUT_PATH, index=False)
    return spliced


if __name__ == "__main__":
    df = build_country_climate_spliced(write=True)
    print(f"wrote {OUT_PATH}: {len(df)} rows; "
          f"sources: {df['source'].value_counts().to_dict()}")
