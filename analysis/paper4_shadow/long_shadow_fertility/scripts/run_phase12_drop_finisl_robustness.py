"""Phase 12 robustness: drop FIN and ISL from the 12-country panel.

FIN and ISL entered with Maddison-native wages (no Allen series).
This script verifies the headline numbers (Hansen wage threshold + regime-FEVD core)
are stable when those two countries are excluded, leaving 10 countries.

Output:
    /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase12_drop_finisl_robustness.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # -> /Volumes/BIGDATA/HYDE35

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression_grid import (
    fit_threshold_grid,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DROP_COUNTRIES = ["FIN", "ISL"]
FULL12_THRESHOLD_JSON = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json"
)
FULL12_FEVD_JSON = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase11_regime_fevd.json"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase12_drop_finisl_robustness.json"
)

HANSEN_THRESHOLD = 9.97
HORIZONS = list(range(0, 16))
P_LAGS = 2
CORE_VARS = ["spei_growing", "log_real_wage", "log_cdr", "log_cbr"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def build_panel_10() -> pd.DataFrame:
    """Build 10-country panel (drop FIN and ISL)."""
    panel = assemble_panel_multi()
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = (
        panel
        .merge(mort, on=["iso3", "year"], how="left")
        .merge(wage, on=["iso3", "year"], how="left")
    )
    df = df[~df["iso3"].isin(DROP_COUNTRIES)].copy()
    df["regime"] = (df["log_real_wage"] > HANSEN_THRESHOLD).astype(int)
    print(f"10-country panel shape: {df.shape}")
    print(f"Countries: {sorted(df['iso3'].unique())}")
    return df


def run_hansen_10(df: pd.DataFrame) -> dict:
    """Hansen threshold grid on 10-country panel (wage only for speed)."""
    print("\nRunning Hansen threshold grid (wage only, n_boot=500)…")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = fit_threshold_grid(
            df, y="log_cbr", x="t_growing",
            z_candidates=["log_real_wage"],
            n_boot=500,
            seed=0,
        )
    res = out.get("log_real_wage", {})
    print(f"  c_hat={res.get('c_hat'):.4f}  p={res.get('sup_wald_pvalue'):.3f}  "
          f"beta_M={res.get('beta_M'):.4f}  beta_T={res.get('beta_T'):.4f}  "
          f"n={res.get('n')}")
    return res


def run_regime_fevd_10(df: pd.DataFrame) -> dict:
    """Core 4-var regime-FEVD on 10-country panel."""
    print("\nRunning regime-FEVD core (10 countries)…")
    cc = df.dropna(subset=CORE_VARS).copy()
    out = {}
    for regime, label in ((0, "malthusian"), (1, "modern")):
        sub = cc[cc["regime"] == regime].copy()
        n = len(sub)
        print(f"  {label}: N={n}")
        res = fit_system_lp_fevd(sub, variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS)
        variables = res["variables"]
        fevd = res["fevd"]["log_cbr"]
        shares_h15 = {v: float(fevd[j][-1]) for j, v in enumerate(variables)}
        out[label] = {"n": int(n), "fevd_log_cbr_h15": shares_h15}
        print(f"    h=15 shares: {', '.join(f'{k}={v:.3f}' for k, v in shares_h15.items())}")
    return out


def compare_with_full12(hansen_10: dict, fevd_10: dict) -> dict:
    """Load full-12 results and build a side-by-side comparison."""
    full12_t = json.loads(FULL12_THRESHOLD_JSON.read_text()).get("log_real_wage", {})
    full12_f = json.loads(FULL12_FEVD_JSON.read_text())
    core12 = full12_f.get("core_by_regime", {})

    comparison = {
        "hansen_wage": {
            "full12": {
                "c_hat": full12_t.get("c_hat"),
                "p":     full12_t.get("sup_wald_pvalue"),
                "beta_M": full12_t.get("beta_M"),
                "beta_T": full12_t.get("beta_T"),
                "n":     full12_t.get("n"),
            },
            "drop_finisl": {
                "c_hat": hansen_10.get("c_hat"),
                "p":     hansen_10.get("sup_wald_pvalue"),
                "beta_M": hansen_10.get("beta_M"),
                "beta_T": hansen_10.get("beta_T"),
                "n":     hansen_10.get("n"),
            },
        },
        "regime_fevd_h15": {},
    }

    for label in ("malthusian", "modern"):
        shares12 = core12.get(label, {}).get("fevd_log_cbr_h15", {})
        shares10 = fevd_10.get(label, {}).get("fevd_log_cbr_h15", {})
        n12 = core12.get(label, {}).get("n")
        n10 = fevd_10.get(label, {}).get("n")
        comparison["regime_fevd_h15"][label] = {
            "full12_n": n12,
            "drop_finisl_n": n10,
            "full12_shares": shares12,
            "drop_finisl_shares": shares10,
            "max_abs_delta": float(max(
                abs(shares10.get(k, 0) - shares12.get(k, 0))
                for k in shares12
            )) if shares12 and shares10 else None,
        }

    # Verdict
    c_delta = abs(hansen_10.get("c_hat", 0) - (full12_t.get("c_hat") or 0))
    max_fevd_delta = max(
        comparison["regime_fevd_h15"].get(lab, {}).get("max_abs_delta") or 0
        for lab in ("malthusian", "modern")
    )
    stable = c_delta < 0.15 and max_fevd_delta < 0.05
    comparison["verdict"] = {
        "c_hat_delta": float(c_delta),
        "max_fevd_share_delta": float(max_fevd_delta),
        "headline_stable": bool(stable),
        "note": (
            "Stable (c_hat within 0.15, FEVD shares within 5pp)" if stable
            else "CONCERN: c_hat or FEVD shares shifted more than tolerance"
        ),
    }
    return comparison


def main():
    df10 = build_panel_10()
    hansen_10 = run_hansen_10(df10)
    fevd_10 = run_regime_fevd_10(df10)
    comparison = compare_with_full12(hansen_10, fevd_10)

    print("\n=== COMPARISON: 12-country vs 10-country (drop FIN+ISL) ===")
    print("\nHansen wage threshold:")
    for k, v in comparison["hansen_wage"].items():
        print(f"  {k}: c={v.get('c_hat'):.4f}  p={v.get('p'):.3f}  "
              f"betaT={v.get('beta_T'):.4f}  n={v.get('n')}")

    print("\nRegime-FEVD h=15 fertility shares:")
    for label in ("malthusian", "modern"):
        entry = comparison["regime_fevd_h15"][label]
        print(f"  {label.upper()}: 12-cty N={entry['full12_n']}  10-cty N={entry['drop_finisl_n']}")
        print(f"    Full12  : {entry['full12_shares']}")
        print(f"    Drop10  : {entry['drop_finisl_shares']}")
        print(f"    Max |Δ| : {entry['max_abs_delta']:.4f}")

    v = comparison["verdict"]
    print(f"\nVerdict: {v['note']}")
    print(f"  Δc_hat={v['c_hat_delta']:.4f}  max FEVD Δ={v['max_fevd_share_delta']:.4f}  stable={v['headline_stable']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(comparison), indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
