"""Rebuild the ERA5 regional panel from the raw monthly archive.

Reads hourly t2m (temperature) and tp (precipitation) from every raw monthly
file (zip container or plain merged netCDF4 — both formats handled by
analysis.shared.build_era5_compact.open_month), computes annual region-level
means, and saves to analysis/data/era5_full_panel.parquet.

Semantics match the original _extracted/-based builder: each month
contributes the mean over all hours and grid cells of the full tile (ocean
included); the annual value is the mean of the available monthly means.
``precipitation_m`` is therefore the mean *hourly accumulation* in metres
(``precipitation_mm`` = x1000), not a monthly or annual total.

The original read the ``_extracted/`` cache, which for years >=1968 held
only the last-extracted month — those annual values were single-month means.
This version reads the per-month files directly.
"""
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT_PATH = ROOT / "analysis" / "data" / "era5_full_panel.parquet"


def _month_means(task):
    region, year, month, path = task
    from analysis.shared.build_era5_compact import open_month
    ds = open_month(Path(path))
    if ds is None or "t2m" not in ds.data_vars:
        return None
    t2m = float(ds["t2m"].mean().values)
    tp = float(ds["tp"].mean().values) if "tp" in ds.data_vars else np.nan
    ds.close()
    return region, year, t2m, tp


def main(workers: int = 6) -> None:
    from analysis.shared.build_era5_compact import iter_raw_files

    tasks = [(r, y, m, str(p)) for r, y, m, p in iter_raw_files()]
    print(f"Scanning {len(tasks)} raw monthly files with {workers} workers ...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = [r for r in pool.imap_unordered(_month_means, tasks,
                                                  chunksize=8) if r]
    df = pd.DataFrame(results, columns=["region", "year", "t2m_k", "tp_m"])
    panel = df.groupby(["region", "year"], as_index=False).agg(
        temperature_k=("t2m_k", "mean"), precipitation_m=("tp_m", "mean"))
    panel["temperature_c"] = panel["temperature_k"] - 273.15
    panel["precipitation_mm"] = panel["precipitation_m"] * 1000.0
    panel = panel[["region", "year", "temperature_k", "precipitation_m",
                   "temperature_c", "precipitation_mm"]]
    panel = panel.sort_values(["region", "year"]).reset_index(drop=True)
    panel.to_parquet(OUT_PATH, index=False)

    print(f"\nERA5 panel: {panel.shape}")
    print(f"Regions: {sorted(panel['region'].unique())}")
    print(f"Years: {panel['year'].min()} - {panel['year'].max()}")
    print("\nTemperature (C) summary:")
    print(panel["temperature_c"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
