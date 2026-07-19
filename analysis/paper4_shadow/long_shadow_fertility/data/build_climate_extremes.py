"""Heat-extreme and drought indicators per (iso3, year), thresholds set on
the unit's own pre-1900 distribution to remove modern-warming bias."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
CLIM_IN = ROOT / "country_climate_annual.parquet"
OUT = ROOT / "climate_extremes_country_year.parquet"


def build_climate_extremes(write: bool = False) -> pd.DataFrame:
    clim = pd.read_parquet(CLIM_IN)
    out_parts = []
    for iso, sub in clim.groupby("iso3"):
        sub = sub.copy()
        pre = sub.loc[sub["year"] < 1900]
        t_thresh = pre["t_growing"].quantile(0.95)
        p_thresh = pre["p_growing"].quantile(0.05)
        sub["heat_extreme"] = (sub["t_growing"] > t_thresh).astype(int)
        sub["drought"] = (sub["p_growing"] < p_thresh).astype(int)
        out_parts.append(sub[["iso3", "year", "heat_extreme", "drought"]])
    df = pd.concat(out_parts, ignore_index=True).sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_climate_extremes(write=True)
    print(f"wrote {OUT}: {len(df)} rows; heat shares per ISO: "
          f"{df.groupby('iso3')['heat_extreme'].mean().to_dict()}")
