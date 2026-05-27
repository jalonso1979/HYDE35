"""Phase 10 Pillar B2: Hansen threshold grid for multiple development proxies.

Runs fit_threshold_grid on the 7-country panel with four candidate threshold
variables: log_real_wage, log_cdr, log_gdppc, and log_tfr (if available).

Output: /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json

Sanity check: log_real_wage c_hat should reproduce Phase 9 finding (≈ 9.97).

Column notes (as of 2026-05-27):
- log_real_wage: merged from real_wage_panel_v2.parquet (not in main panel)
- log_cdr:       merged from country_mortality_annual.parquet (not in main panel)
- log_gdppc:     present in main panel (panel_multi_country_year.parquet)
- log_tfr:       not found in any panel — DROPPED from z_candidates
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script or as a module
# pyproject.toml sets rootdir to /Volumes/BIGDATA/HYDE35; match that for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # -> /Volumes/BIGDATA/HYDE35

from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression_grid import (
    fit_threshold_grid,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PANEL_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
REAL_WAGE_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
MORTALITY_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/country_mortality_annual.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build merged panel
# ---------------------------------------------------------------------------
print("Loading panel...")
df = pd.read_parquet(PANEL_PATH)
print(f"  Main panel columns: {list(df.columns)}")
print(f"  Main panel shape: {df.shape}")

# Merge real wage
rw = pd.read_parquet(REAL_WAGE_PATH)[["iso3", "year", "log_real_wage"]]
df = df.merge(rw, on=["iso3", "year"], how="left")
print(f"  log_real_wage non-null after merge: {df['log_real_wage'].notna().sum()}")

# Merge mortality (log_cdr)
mort = pd.read_parquet(MORTALITY_PATH)[["iso3", "year", "log_cdr"]]
df = df.merge(mort, on=["iso3", "year"], how="left")
print(f"  log_cdr non-null after merge: {df['log_cdr'].notna().sum()}")

# Derive log_tfr if possible (not expected to exist)
for src, dst in [("cdr", "log_cdr_alt"), ("tfr", "log_tfr")]:
    if src in df.columns and dst not in df.columns:
        s = df[src]
        df[dst] = s.where(s > 0).pipe(np.log)
        print(f"  Derived {dst} from {src}")

# Report available columns
print(f"\nMerged panel shape: {df.shape}")
print(f"iso3 values: {sorted(df['iso3'].unique())}")
for col in ["log_cbr", "t_growing", "log_real_wage", "log_cdr", "log_gdppc", "log_tfr"]:
    n = df[col].notna().sum() if col in df.columns else "N/A (col missing)"
    print(f"  {col}: {n} non-null")

# ---------------------------------------------------------------------------
# Determine z_candidates (drop unavailable)
# ---------------------------------------------------------------------------
z_wanted = ["log_real_wage", "log_cdr", "log_gdppc", "log_tfr"]
z_candidates = [z for z in z_wanted if z in df.columns]
z_dropped = [z for z in z_wanted if z not in df.columns]
if z_dropped:
    print(f"\nDropped z_candidates (not in panel): {z_dropped}")
print(f"Running grid for: {z_candidates}")

# ---------------------------------------------------------------------------
# Run threshold grid
# ---------------------------------------------------------------------------
print("\nRunning fit_threshold_grid (n_boot=500)...")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=z_candidates,
        n_boot=500,
        seed=0,
    )
    for w in caught:
        print(f"  WARNING: {w.message}")

# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------
print("\n=== THRESHOLD GRID RESULTS ===")
for z, res in out.items():
    print(
        f"  {z:20s}: c_hat={res['c_hat']:.4f}  "
        f"CI=[{res['c_ci_lo']:.3f}, {res['c_ci_hi']:.3f}]  "
        f"p={res['sup_wald_pvalue']:.3f}  "
        f"beta_M={res['beta_M']:.4f}  beta_T={res['beta_T']:.4f}  "
        f"n={res['n']}"
    )

# Sanity check
if "log_real_wage" in out:
    c_hat_wage = out["log_real_wage"]["c_hat"]
    phase9_ref = 9.97
    deviation = abs(c_hat_wage - phase9_ref)
    status = "PASS" if deviation < 0.05 else ("WARN" if deviation < 0.2 else "FAIL")
    print(
        f"\nSanity check log_real_wage c_hat={c_hat_wage:.4f} "
        f"vs Phase9={phase9_ref} → deviation={deviation:.4f} [{status}]"
    )

# ---------------------------------------------------------------------------
# Serialise to JSON
# ---------------------------------------------------------------------------
def _clean(o):
    """Make numpy/pandas scalars JSON-serialisable."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    return o


OUT.write_text(json.dumps(_clean(out), indent=2))
print(f"\nWrote {OUT}")
print(f"log_real_wage c_hat: {out.get('log_real_wage', {}).get('c_hat')}")
