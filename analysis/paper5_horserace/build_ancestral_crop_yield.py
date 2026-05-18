"""Build Galor-Özak ancestral crop-yield potential substrate.

Source: Galor & Özak (2016, AER) "The Agricultural Origins of Time Preference."
        American Economic Review 106(10): 3064-3103.
        Caloric Suitability Index data from ozak/Caloric-Suitability-Index.
        Zenodo archive: https://zenodo.org/records/14714917

Variable used: pre15002AverageCalories0mean
  pre1500 = pre-Columbian crop set (before the Columbian Exchange)
  2       = excludes Asian crop varieties in Africa (conservative Old-World definition)
  Average = mean of single-crop caloric yields across crops suitable for each cell
  0       = statistics exclude cells with zero yield (rather than treating zero as data)
  mean    = population-unweighted mean across all grid cells within the country polygon

This is the country-level average caloric yield potential under pre-1500 CE crop
availability, which is the conceptually appropriate measure for the Galor-Özak
(2016 AER) analysis of the deep agricultural origins of time preference.

Download:
    cd analysis/data/deep_determinants/_raw/galor_ozak_2016
    curl -L -o country_Calories_stats_web.csv \\
        "https://zenodo.org/records/14714917/files/country_Calories_stats_web.csv?download=1"

Output: analysis/data/deep_determinants/ancestral_crop_yield.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
RAW = ROOT / "analysis/data/deep_determinants/_raw/galor_ozak_2016"
OUT = ROOT / "analysis/data/deep_determinants/ancestral_crop_yield.parquet"

# Primary variable: pre-1500 CE, Old-World-restricted caloric suitability index,
# average caloric yield excluding zero-yield cells, mean across country pixels.
YIELD_COL = "pre15002AverageCalories0mean"

# ISO-3 column name in the CSI dataset
ISO_COL = "ISO_A3"


def main() -> None:
    # --- 1. locate raw file ---
    candidates = (
        list(RAW.glob("*.dta"))
        + list(RAW.glob("*.csv"))
        + list(RAW.glob("country_Calories*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Galor-Özak data file in {RAW}.\n"
            "Download with:\n"
            "  curl -L -o analysis/data/deep_determinants/_raw/galor_ozak_2016/"
            "country_Calories_stats_web.csv "
            "'https://zenodo.org/records/14714917/files/"
            "country_Calories_stats_web.csv?download=1'"
        )

    # Prefer the CSV; fall back to DTA
    csv_candidates = [f for f in candidates if f.suffix == ".csv"]
    raw = csv_candidates[0] if csv_candidates else candidates[0]
    print(f"Reading: {raw}")

    # --- 2. load ---
    if raw.suffix == ".dta":
        df = pd.read_stata(str(raw))
    else:
        df = pd.read_csv(raw, encoding="latin-1")

    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- 3. validate required columns ---
    missing = {ISO_COL, YIELD_COL} - set(df.columns)
    if missing:
        raise KeyError(
            f"Expected columns not found: {missing}.\n"
            f"Available columns (first 30): {list(df.columns[:30])}"
        )

    # --- 4. select and rename ---
    df = df[[ISO_COL, YIELD_COL]].copy()
    df = df.rename(columns={ISO_COL: "iso3", YIELD_COL: "ancestral_yield"})

    # --- 5. clean ISO-3 codes ---
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    valid_iso = df["iso3"].str.match(r"^[A-Z]{3}$")
    n_dropped = (~valid_iso).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} rows with non-ISO-3 codes: "
              f"{df.loc[~valid_iso, 'iso3'].unique()[:10].tolist()}")
    df = df[valid_iso]

    # --- 6. drop rows with null yield ---
    n_null = df["ancestral_yield"].isna().sum()
    if n_null > 0:
        print(f"  Dropping {n_null} rows with null ancestral_yield")
    df = df.dropna(subset=["ancestral_yield"])

    # --- 7. enforce non-negativity ---
    n_neg = (df["ancestral_yield"] < 0).sum()
    if n_neg > 0:
        print(f"  WARNING: {n_neg} rows have negative ancestral_yield; clamping to 0")
        df["ancestral_yield"] = df["ancestral_yield"].clip(lower=0.0)

    # --- 8. log transform (log1p so zero-yield countries get log1p(0)=0) ---
    df["ancestral_yield_log"] = np.log1p(df["ancestral_yield"])

    # --- 9. source label ---
    df["source"] = (
        "Galor-Özak (2016 AER) Caloric Suitability Index; "
        "variable pre15002AverageCalories0mean from "
        "ozak/Caloric-Suitability-Index (Zenodo 14714917)"
    )

    # --- 10. deduplicate and sort ---
    n_before = len(df)
    df = df.drop_duplicates(subset=["iso3"])
    n_dup = n_before - len(df)
    if n_dup > 0:
        print(f"  Dropped {n_dup} duplicate iso3 rows")
    df = df.sort_values("iso3").reset_index(drop=True)

    # --- 11. write ---
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep = ["iso3", "ancestral_yield", "ancestral_yield_log", "source"]
    df[keep].to_parquet(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")

    # --- 12. quick sanity check ---
    vals = df.set_index("iso3")["ancestral_yield"]
    for pair in [("FRA", "SAU"), ("IND", "ISL"), ("ITA", "IND")]:
        hi, lo = pair
        if hi in vals.index and lo in vals.index:
            assert vals[hi] > vals[lo], f"Sanity check failed: {hi} > {lo}"
            print(f"  Sanity OK: {hi}={vals[hi]:.0f} > {lo}={vals[lo]:.0f}")


if __name__ == "__main__":
    main()
