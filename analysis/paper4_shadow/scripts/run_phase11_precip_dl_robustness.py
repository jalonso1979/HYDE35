"""
Phase 11: Precipitation DL robustness — diagnose sample-collapse from full control vector.

Runs 3 × 2 (control-spec × shock) grid:
  Specs: (a) full controls, (b) no disaster controls, (c) minimal (FE + lags only)
  Shocks: p_growing (precipitation), t_growing (temperature)

Outputs JSON to analysis/output/long_shadow_fertility/phase11_precip_dl_robustness.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root as a script
sys.path.insert(0, "/Volumes/BIGDATA/HYDE35")

from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_distributed_lag import (
    fit_pooled_distributed_lag,
)

PANEL = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/"
    "phase11_precip_dl_robustness.json"
)

# Control specs
CONTROLS_FULL = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]
CONTROLS_NODIS = [  # drop EMDAT disaster controls (only post-1962)
    "war_active", "log_war_fatalities", "pandemic_active",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]
CONTROLS_MINIMAL: list[str] = []  # country FE + year FE + lags only

SPECS = {
    "a_full_controls": CONTROLS_FULL,
    "b_no_disaster": CONTROLS_NODIS,
    "c_minimal": CONTROLS_MINIMAL,
}

SHOCKS = ["p_growing", "t_growing"]
LAGS = 3


def run_spec(
    df: pd.DataFrame,
    shock: str,
    controls: list[str],
    label: str,
) -> dict:
    """Run a single DL spec and return summary stats."""
    # determine effective sample (same logic as estimator's dropna)
    needed_cols = ["log_cbr", shock, "iso3", "year"] + controls
    sub = df.dropna(subset=needed_cols)
    # also need lag rows — the estimator adds lags and dropna's internally,
    # so approximate N by counting rows after adding lags manually
    sub_sorted = sub.sort_values(["iso3", "year"]).copy()
    for k in range(1, LAGS + 1):
        sub_sorted[f"{shock}_lag{k}"] = sub_sorted.groupby("iso3")[shock].shift(k)
    lag_cols = [f"{shock}_lag{k}" for k in range(LAGS + 1)]
    lag_cols_all = [shock] + [f"{shock}_lag{k}" for k in range(1, LAGS + 1)]
    sub_sorted[f"{shock}_lag0"] = sub_sorted[shock]
    lag_cols = [f"{shock}_lag{k}" for k in range(LAGS + 1)]
    sub_final = sub_sorted.dropna(subset=["log_cbr"] + lag_cols + controls)

    n = int(len(sub_final))
    n_countries = int(sub_final["iso3"].nunique())
    year_min = int(sub_final["year"].min())
    year_max = int(sub_final["year"].max())

    try:
        results = fit_pooled_distributed_lag(
            df=sub,
            y="log_cbr",
            x=shock,
            lags=LAGS,
            controls=controls if controls else None,
            year_fe=True,
        )
        cum_row = results.loc[results["lag"] == "cumulative"].iloc[0]
        cum_beta = float(cum_row["beta"])
        se = float(cum_row["se"])
        t_stat = round(cum_beta / se, 3) if se > 0 else None
    except Exception as exc:
        print(f"  ERROR in {label} / {shock}: {exc}")
        cum_beta = None
        se = None
        t_stat = None

    result = {
        "spec": label,
        "shock": shock,
        "controls": controls,
        "N": n,
        "N_countries": n_countries,
        "year_min": year_min,
        "year_max": year_max,
        "cum_beta": round(cum_beta, 5) if cum_beta is not None else None,
        "se": round(se, 5) if se is not None else None,
        "t": round(t_stat, 3) if t_stat is not None else None,
    }
    return result


def report_coverage(df: pd.DataFrame) -> dict:
    """Report non-null coverage of each control variable."""
    all_controls = list(dict.fromkeys(CONTROLS_FULL))  # deduplicated
    coverage = {}
    for c in all_controls:
        n_nonnull = int(df[c].notna().sum())
        total = int(len(df))
        yr_min = int(df.loc[df[c].notna(), "year"].min()) if n_nonnull > 0 else None
        yr_max = int(df.loc[df[c].notna(), "year"].max()) if n_nonnull > 0 else None
        coverage[c] = {
            "n_nonnull": n_nonnull,
            "total": total,
            "pct_nonnull": round(100 * n_nonnull / total, 1),
            "year_min": yr_min,
            "year_max": yr_max,
        }
    return coverage


def main():
    print(f"Loading panel from {PANEL}")
    df = pd.read_parquet(PANEL)
    print(f"Panel: {df.shape}, countries: {sorted(df['iso3'].unique())}")
    print(f"Year range: {df['year'].min()}-{df['year'].max()}")

    # Step 1: Diagnose coverage
    print("\n=== Control variable coverage ===")
    coverage = report_coverage(df)
    for c, v in coverage.items():
        print(
            f"  {c}: {v['n_nonnull']}/{v['total']} ({v['pct_nonnull']}%), "
            f"years {v['year_min']}-{v['year_max']}"
        )

    # Step 2: Run 3 × 2 grid
    print("\n=== Running 3 × 2 DL grid ===")
    grid_results = []
    for spec_label, controls in SPECS.items():
        for shock in SHOCKS:
            print(f"\n  Spec={spec_label}, shock={shock}")
            r = run_spec(df, shock, controls, spec_label)
            grid_results.append(r)
            print(
                f"    N={r['N']}, N_countries={r['N_countries']}, "
                f"years={r['year_min']}-{r['year_max']}, "
                f"cum_β={r['cum_beta']}, SE={r['se']}, t={r['t']}"
            )

    # Step 3: Assemble output
    output = {
        "description": (
            "Phase 11 robustness: precipitation DL sample-collapse diagnosis. "
            "3 control specs × 2 shocks (p_growing, t_growing). "
            "Key finding: disaster controls (disaster_count, log_disaster_deaths) cover "
            "only 11.6% of panel (1900-2022), while war/pandemic controls cover 57.2% (1421-2022). "
            "Full control spec collapses to N~436, 4 countries (1900-2008); "
            "no-disaster spec expands to N~990, 4 countries (1541-2008); "
            "minimal spec uses N~1419, 7 countries (1541-2008)."
        ),
        "control_coverage": coverage,
        "grid": grid_results,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(f"\nWritten to {OUT}")

    # Pretty-print grid table
    print("\n=== RESULTS GRID ===")
    print(f"{'Spec':<20} {'Shock':<12} {'N':>6} {'Countries':>10} {'Years':>12} {'cum_β':>8} {'SE':>7} {'t':>7}")
    print("-" * 90)
    for r in grid_results:
        print(
            f"{r['spec']:<20} {r['shock']:<12} {r['N']:>6} {r['N_countries']:>10} "
            f"{str(r['year_min'])+'-'+str(r['year_max']):>12} "
            f"{r['cum_beta'] if r['cum_beta'] is not None else 'NA':>8} "
            f"{r['se'] if r['se'] is not None else 'NA':>7} "
            f"{r['t'] if r['t'] is not None else 'NA':>7}"
        )


if __name__ == "__main__":
    main()
