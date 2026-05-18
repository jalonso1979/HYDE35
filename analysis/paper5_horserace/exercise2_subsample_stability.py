# analysis/paper5_horserace/exercise2_subsample_stability.py
"""Re-run Exercise 2 mediation across the SUBSAMPLE_STAGES sub-samples.

For each of the 5 sequential sub-sample stages, run the full 4×4 mediation
analysis (4 outcomes × 4 substrates) with n_boot=500 bootstrap replications.

5 stages × 16 cells × 500 bootstraps = 40,000 regressions.

Output: analysis/data/deep_determinants/subsample_stability.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.mediation import mediation_share_with_ci
from analysis.paper5_horserace.subsamples import SUBSAMPLE_STAGES

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/subsample_stability.parquet"

SUBSTRATES = [
    "sigma_v_T_pre1750",
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]
OUTCOMES = [
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]
CONTROLS = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]


def main() -> None:
    df = pd.read_parquet(PANEL)

    # Identify pathway dummies in panel (skip pathway_0, which is the reference)
    pathway_cols = sorted(
        [c for c in df.columns if c.startswith("pathway_") and c != "pathway_0"]
    )
    print(f"Pathway dummies: {pathway_cols}")

    rows = []
    total_stages = len(SUBSAMPLE_STAGES)
    for stage_idx, (stage, drop_list) in enumerate(SUBSAMPLE_STAGES):
        sub = df[~df["iso3"].isin(drop_list)].copy()
        n_countries = len(sub)
        print(
            f"\n[Stage {stage_idx + 1}/{total_stages}] {stage}  "
            f"(n={n_countries}, dropped={len(set(drop_list))} codes)"
        )
        for outcome in OUTCOMES:
            for substrate in SUBSTRATES:
                # Derive a reproducible seed from stage + cell
                seed = (44 + abs(hash((stage, outcome, substrate)))) % (2**31)
                r = mediation_share_with_ci(
                    sub,
                    y_col=outcome,
                    substrate=substrate,
                    pathway_dummies=pathway_cols,
                    controls=CONTROLS,
                    n_boot=500,
                    seed=seed,
                )
                rows.append(
                    {
                        "stage": stage,
                        "stage_idx": stage_idx,
                        "outcome": outcome,
                        "substrate": substrate,
                        "n_countries": n_countries,
                        **r,
                    }
                )
                ms = r["mediation_share"]
                ci = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
                print(f"  {outcome[:25]:25s} ~ {substrate[:25]:25s}  "
                      f"med_share={ms:+.3f}  CI={ci}  n_obs={r['n_obs']}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")

    # Quick summary: median |mediation share| by stage
    print("\nMedian |mediation share| by stage:")
    for stage, grp in out_df.groupby("stage"):
        med_abs = grp["mediation_share"].dropna().abs().median()
        in_unit = ((grp["mediation_share"] >= 0) &
                   (grp["mediation_share"] <= 1)).mean()
        print(f"  {stage:35s}  |med|={med_abs:.3f}  frac_in_[0,1]={in_unit:.2f}")


if __name__ == "__main__":
    main()
