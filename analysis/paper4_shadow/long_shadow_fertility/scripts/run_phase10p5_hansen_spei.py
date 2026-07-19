"""Phase 10 Pillar #5: Hansen threshold regression with SPEI as shock.

Runs fit_threshold_regression with:
  - y = log_cbr  (log crude birth rate)
  - x = spei_growing  (growing-season SPEI — the headline Pillar 5 shock)
  - z = log_real_wage  (threshold variable — same as Phase 9 Pillar B)

Also runs the temperature comparator with x = t_growing for a direct
comparison on the identical sample.

Output: phase10p5_hansen_spei.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Volumes/BIGDATA/HYDE35")

from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression import (
    fit_threshold_regression,
)

PANEL = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
WAGE = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10p5_hansen_spei.json"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> None:
    print("Loading panel...")
    df = pd.read_parquet(PANEL)
    wage = pd.read_parquet(WAGE)[["iso3", "year", "log_real_wage"]]
    df = df.merge(wage, on=["iso3", "year"], how="left")

    # Restrict to rows where both SPEI and wage are available
    spei_avail = df[["iso3", "year", "log_cbr", "spei_growing",
                      "t_growing", "log_real_wage"]].dropna()
    print(f"  Joint sample (SPEI + wage + log_cbr): {len(spei_avail)} rows, "
          f"{spei_avail['iso3'].nunique()} countries")
    print(f"  Year range: {spei_avail['year'].min()} - {spei_avail['year'].max()}")

    # --- SPEI Hansen ---
    print("\nFitting Hansen threshold: SPEI as shock...")
    res_spei = fit_threshold_regression(
        spei_avail,
        y="log_cbr",
        x="spei_growing",
        z="log_real_wage",
        unit_col="iso3",
        n_boot=500,
        seed=0,
    )
    print(f"  c_hat = {res_spei['c_hat']:.3f}")
    print(f"  p     = {res_spei['sup_wald_pvalue']:.3f}")
    print(f"  beta_M = {res_spei['beta_M']:.4f}  (below threshold, Malthusian)")
    print(f"  beta_T = {res_spei['beta_T']:.4f}  (above threshold, transitional)")
    print(f"  n      = {res_spei['n']}")

    # --- Temperature comparator on identical sample ---
    print("\nFitting Hansen threshold: temperature as shock (comparator)...")
    res_temp = fit_threshold_regression(
        spei_avail,
        y="log_cbr",
        x="t_growing",
        z="log_real_wage",
        unit_col="iso3",
        n_boot=500,
        seed=0,
    )
    print(f"  c_hat = {res_temp['c_hat']:.3f}")
    print(f"  p     = {res_temp['sup_wald_pvalue']:.3f}")
    print(f"  beta_M = {res_temp['beta_M']:.4f}  (below threshold, Malthusian)")
    print(f"  beta_T = {res_temp['beta_T']:.4f}  (above threshold, transitional)")
    print(f"  n      = {res_temp['n']}")

    out = {
        "spei": res_spei,
        "temperature_comparator": res_temp,
        "meta": {
            "n_obs": int(len(spei_avail)),
            "n_countries": int(spei_avail["iso3"].nunique()),
            "year_min": int(spei_avail["year"].min()),
            "year_max": int(spei_avail["year"].max()),
            "note": (
                "SPEI = growing-season (Apr-Sep) Thornthwaite SPEI, standardised "
                "per country; Path A using CRU 1901-1950 baseline + ModE-RA anomalies. "
                "Temperature comparator uses identical sample for comparability."
            ),
        },
    }
    OUT.write_text(json.dumps(clean(out), indent=2))
    print(f"\nWrote {OUT}")

    print("\n=== SUMMARY ===")
    print(f"SPEI:        c_hat={res_spei['c_hat']:.3f}, p={res_spei['sup_wald_pvalue']:.3f}, "
          f"beta_M={res_spei['beta_M']:.4f}, beta_T={res_spei['beta_T']:.4f}")
    print(f"Temperature: c_hat={res_temp['c_hat']:.3f}, p={res_temp['sup_wald_pvalue']:.3f}, "
          f"beta_M={res_temp['beta_M']:.4f}, beta_T={res_temp['beta_T']:.4f}")

    # Interpret
    ratio_spei = res_spei["beta_T"] / res_spei["beta_M"] if res_spei["beta_M"] != 0 else float("nan")
    ratio_temp = res_temp["beta_T"] / res_temp["beta_M"] if res_temp["beta_M"] != 0 else float("nan")
    print(f"\nElasticity amplification (beta_T / beta_M):")
    print(f"  SPEI:        {ratio_spei:.2f}x")
    print(f"  Temperature: {ratio_temp:.2f}x")


if __name__ == "__main__":
    main()
