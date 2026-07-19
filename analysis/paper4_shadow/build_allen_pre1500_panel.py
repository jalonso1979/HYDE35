"""Parse the raw Allen-Nuffield city xls files into a long city-year panel of
wheat prices in grams of silver per hectoliter.

The existing `analysis/data/allen_wage_panel.csv` was built from a subset of
sources and is sparse pre-1421 (Tuscany is the only city with substantial
wheat-price coverage before 1320).  Allen's canonical 17-city dataset reaches
back to 1259 (London) and provides silver-content-normalised wheat prices
directly in section A2 of each city xls.  This builder extracts the silver-
gram-per-hectoliter wheat-price column for each city.

Sources: https://www.nuffield.ox.ac.uk/people/sites/allen-research-pages/
         (downloaded 2026-05 by `scripts/`-equivalent ad-hoc curl;
          one xls per city, file name pattern allen_<city>.xls).

Output: analysis/data/allen_silver_wheat_prices_1259_1914.parquet
        columns = [city, year, silver_g_per_hl, log_price]
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import re

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
RAW = DATA / "gpih_raw"


def _find_silver_wheat_col(df: pd.DataFrame) -> int | None:
    silver_start = None
    for c in range(df.shape[1]):
        v = df.iloc[1, c]
        if pd.notna(v) and ("Silver" in str(v) or "A2" in str(v)):
            silver_start = c; break
    if silver_start is None:
        return None
    for c in range(silver_start, df.shape[1]):
        v = df.iloc[6, c]
        if pd.notna(v) and str(v).strip().lower() == "wheat":
            return c
    return None


def _parse_city(path: Path) -> pd.DataFrame:
    city = path.stem.replace("allen_", "").replace("-", " ").title()
    df = pd.read_excel(path, sheet_name="Prices", header=None)
    col = _find_silver_wheat_col(df)
    if col is None:
        return pd.DataFrame(columns=["city", "year", "silver_g_per_hl"])
    # Years are in column 0 from row 8 onward; some files use row 7 as data start.
    # We accept any rows where col 0 is numeric in [0, 2025] and col is numeric.
    years = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(df.iloc[:, col], errors="coerce")
    valid = years.between(1, 2025) & vals.notna() & (vals > 0)
    out = pd.DataFrame({"city": city,
                        "year": years[valid].astype(int).values,
                        "silver_g_per_hl": vals[valid].astype(float).values})
    # Deduplicate (some sheets have repeated years across sub-tables)
    out = out.groupby(["city", "year"], as_index=False)["silver_g_per_hl"].mean()
    return out


def main() -> None:
    print("Parsing Allen-Nuffield city xls files for silver-gram wheat prices…")
    rows = []
    for path in sorted(RAW.glob("allen_*.xls")):
        sub = _parse_city(path)
        if sub.empty:
            print(f"  {path.name:30s} — no wheat-silver column found")
            continue
        rows.append(sub)
        print(f"  {path.name:30s}  N={len(sub):>4}  years {sub.year.min()}–{sub.year.max()}")
    if not rows:
        print("No city panels parsed; aborting.")
        return
    df = pd.concat(rows, ignore_index=True)
    df["log_price"] = np.log(df["silver_g_per_hl"])
    out_path = DATA / "allen_silver_wheat_prices_1259_1914.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nTotal panel: {len(df):,} city-year obs across {df['city'].nunique()} cities")
    print(f"Pre-1500 obs: {(df['year'] < 1500).sum():,} "
          f"({df[df['year']<1500]['city'].nunique()} cities)")
    print(f"\nPre-1500 obs by city:")
    print(df[df["year"] < 1500].groupby("city").agg(
        n=("year", "count"), ymin=("year", "min"), ymax=("year", "max")
    ).sort_values("ymin").to_string())
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
