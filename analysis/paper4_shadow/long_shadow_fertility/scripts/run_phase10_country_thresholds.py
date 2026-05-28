"""Phase 10 Pillar D2: Country-heterogeneous Hansen threshold driver.

Runs fit_country_specific_thresholds (n_boot=500) for each country
in the 7-country panel, runs the pooled Hansen for reference, and
computes a Phillips-Sul-style heterogeneity test.

Output:
    /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_country_thresholds.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Allow running as a script (rootdir = /Volumes/BIGDATA/HYDE35)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from analysis.paper4_shadow.long_shadow_fertility.estimators.country_heterogeneous_threshold import (
    fit_country_specific_thresholds,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression import (
    fit_threshold_regression,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PANEL = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
WAGE = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_country_thresholds.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(o):
    """Make numpy/pandas scalars JSON-serialisable."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    t0 = time.time()

    # --- Load and merge panel ---
    print("Loading panel...")
    df = pd.read_parquet(PANEL)
    print(f"  Main panel shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    print("Loading real wage panel...")
    wage = pd.read_parquet(WAGE)[["iso3", "year", "log_real_wage"]]
    df = df.merge(wage, on=["iso3", "year"], how="left")
    print(f"  log_real_wage non-null after merge: {df['log_real_wage'].notna().sum()}")
    print(f"  iso3 values: {sorted(df['iso3'].unique())}")

    # Report key columns
    for col in ["log_cbr", "t_growing", "log_real_wage"]:
        n = df[col].notna().sum() if col in df.columns else "N/A (missing)"
        print(f"  {col}: {n} non-null")
    print()

    # --- Pooled Hansen ---
    print("Running pooled Hansen threshold regression (n_boot=500)...")
    t1 = time.time()
    pooled = fit_threshold_regression(
        df, y="log_cbr", x="t_growing", z="log_real_wage",
        unit_col="iso3", n_boot=500, seed=0,
    )
    print(f"  Done in {time.time() - t1:.1f}s")
    print(f"  c_hat = {pooled['c_hat']:.4f}")
    print(f"  beta_M = {pooled['beta_M']:.4f}, beta_T = {pooled['beta_T']:.4f}")
    print(f"  sup-Wald p = {pooled['sup_wald_pvalue']:.4f}")
    print()

    # --- Per-country Hansen ---
    print("Running per-country Hansen threshold regression (n_boot=500, min_obs=80)...")
    print("  (This step takes ~20-40 min; one country at a time)")
    t2 = time.time()
    by_country = fit_country_specific_thresholds(
        df, y="log_cbr", x="t_growing", z="log_real_wage",
        unit_col="iso3", n_boot=500, seed=0, min_obs=80,
    )
    print(f"  Done in {time.time() - t2:.1f}s")
    print()

    print("=== By country ===")
    for iso3, res in sorted(by_country.items()):
        if isinstance(res, dict) and "c_hat" in res:
            print(
                f"  {iso3}: c={res['c_hat']:.4f}  "
                f"beta_M={res['beta_M']:.4f}  beta_T={res['beta_T']:.4f}  "
                f"p={res['sup_wald_pvalue']:.4f}  n={res['n']}"
            )
        else:
            print(f"  {iso3}: {res}")
    print()

    # --- Heterogeneity diagnostic ---
    chats = np.array(
        [r["c_hat"] for r in by_country.values()
         if isinstance(r, dict) and "c_hat" in r]
    )
    if len(chats) >= 2:
        mean_c = float(chats.mean())
        sd_c = float(chats.std(ddof=1))
        z_stat = float((mean_c - pooled["c_hat"]) / (sd_c / np.sqrt(len(chats))))
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        het = {
            "chats": chats.tolist(),
            "mean": mean_c,
            "sd": sd_c,
            "n_countries": int(len(chats)),
            "pooled_c": float(pooled["c_hat"]),
            "z_vs_pooled": z_stat,
            "p_value": p_val,
        }
    else:
        het = {
            "chats": chats.tolist(),
            "n_countries": int(len(chats)),
            "note": "insufficient countries for heterogeneity test",
        }

    print("=== Heterogeneity diagnostic ===")
    print(f"  Mean c_hat = {het.get('mean', 'n/a'):.4f}")
    print(f"  SD c_hat   = {het.get('sd', 'n/a'):.4f}")
    print(f"  Pooled c   = {het.get('pooled_c', 'n/a'):.4f}")
    print(f"  z vs pooled = {het.get('z_vs_pooled', 'n/a'):.4f}")
    print(f"  p-value    = {het.get('p_value', 'n/a'):.4f}")
    print()

    # --- Write output ---
    out = {
        "pooled": pooled,
        "by_country": by_country,
        "heterogeneity": het,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(out), indent=2))
    print(f"Wrote {OUT}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
