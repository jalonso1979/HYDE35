"""Phase 10 Pillar #5 channel decomposition: Hansen threshold regression
with precipitation (p_growing) as the shock variable.

Runs fit_threshold_regression with:
  - y = log_cbr  (log crude birth rate)
  - x = p_growing  (growing-season precipitation anomaly, mm)
  - z = log_real_wage  (threshold variable)

Uses identical sample restriction as the SPEI + temperature comparators
(rows where spei_growing, p_growing, and log_real_wage are all present)
so all three shock variants are directly comparable.

Output: phase10p5_hansen_precip.json
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

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
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10p5_hansen_precip.json"
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

    # Restrict to rows where all three shock variables + wage are present,
    # so comparisons against SPEI and temperature are on identical samples.
    joint = df[["iso3", "year", "log_cbr", "spei_growing",
                "t_growing", "p_growing", "log_real_wage"]].dropna()
    print(f"  Joint sample (SPEI+T+P+wage+log_cbr): {len(joint)} rows, "
          f"{joint['iso3'].nunique()} countries")
    print(f"  Year range: {joint['year'].min()} - {joint['year'].max()}")

    # --- Precipitation Hansen ---
    print("\nFitting Hansen threshold: precipitation (p_growing) as shock...")
    res_precip = fit_threshold_regression(
        joint,
        y="log_cbr",
        x="p_growing",
        z="log_real_wage",
        unit_col="iso3",
        n_boot=500,
        seed=0,
    )
    print(f"  c_hat = {res_precip['c_hat']:.3f}")
    print(f"  p     = {res_precip['sup_wald_pvalue']:.3f}")
    print(f"  beta_M = {res_precip['beta_M']:.4f}  (below threshold, Malthusian)")
    print(f"  beta_T = {res_precip['beta_T']:.4f}  (above threshold, transitional)")
    print(f"  n      = {res_precip['n']}")

    # --- Temperature comparator on identical joint sample ---
    print("\nFitting Hansen threshold: temperature (t_growing) comparator on joint sample...")
    res_temp = fit_threshold_regression(
        joint,
        y="log_cbr",
        x="t_growing",
        z="log_real_wage",
        unit_col="iso3",
        n_boot=500,
        seed=0,
    )
    print(f"  c_hat = {res_temp['c_hat']:.3f}")
    print(f"  p     = {res_temp['sup_wald_pvalue']:.3f}")
    print(f"  beta_M = {res_temp['beta_M']:.4f}")
    print(f"  beta_T = {res_temp['beta_T']:.4f}")
    print(f"  n      = {res_temp['n']}")

    # --- SPEI comparator on identical joint sample ---
    print("\nFitting Hansen threshold: SPEI comparator on joint sample...")
    res_spei = fit_threshold_regression(
        joint,
        y="log_cbr",
        x="spei_growing",
        z="log_real_wage",
        unit_col="iso3",
        n_boot=500,
        seed=0,
    )
    print(f"  c_hat = {res_spei['c_hat']:.3f}")
    print(f"  p     = {res_spei['sup_wald_pvalue']:.3f}")
    print(f"  beta_M = {res_spei['beta_M']:.4f}")
    print(f"  beta_T = {res_spei['beta_T']:.4f}")
    print(f"  n      = {res_spei['n']}")

    out = {
        "precipitation": res_precip,
        "temperature_comparator": res_temp,
        "spei_comparator": res_spei,
        "meta": {
            "n_obs": int(len(joint)),
            "n_countries": int(joint["iso3"].nunique()),
            "year_min": int(joint["year"].min()),
            "year_max": int(joint["year"].max()),
            "note": (
                "All three shock variants (precipitation p_growing, temperature t_growing, "
                "SPEI spei_growing) estimated on the identical joint sample (all four "
                "variables non-null) for direct comparability. "
                "Threshold variable: harmonized log real wage (Allen-Maddison). "
                "n_boot=500 wild-cluster bootstrap for sup-Wald p-value."
            ),
        },
    }
    OUT.write_text(json.dumps(clean(out), indent=2))
    print(f"\nWrote {OUT}")

    print("\n=== SUMMARY ===")
    print(f"Precipitation: c_hat={res_precip['c_hat']:.3f}, p={res_precip['sup_wald_pvalue']:.3f}, "
          f"beta_M={res_precip['beta_M']:.4f}, beta_T={res_precip['beta_T']:.4f}")
    print(f"Temperature:   c_hat={res_temp['c_hat']:.3f}, p={res_temp['sup_wald_pvalue']:.3f}, "
          f"beta_M={res_temp['beta_M']:.4f}, beta_T={res_temp['beta_T']:.4f}")
    print(f"SPEI:          c_hat={res_spei['c_hat']:.3f}, p={res_spei['sup_wald_pvalue']:.3f}, "
          f"beta_M={res_spei['beta_M']:.4f}, beta_T={res_spei['beta_T']:.4f}")

    # Amplification ratios
    for label, res in [("Precipitation", res_precip), ("Temperature", res_temp), ("SPEI", res_spei)]:
        ratio = res["beta_T"] / res["beta_M"] if res["beta_M"] != 0 else float("nan")
        print(f"  {label} amplification (beta_T / beta_M): {ratio:.2f}x")


if __name__ == "__main__":
    main()
