# analysis/paper5_horserace/exercise2_subsample_stability.py
"""Re-run Exercise 2 mediation across the SUBSAMPLE_STAGES sub-samples.

For each of the 5 sequential sub-sample stages, run the full 4×6 mediation
analysis (4 substrates × 6 outcomes) with n_boot=500 bootstrap replications.

Climate bundle uses partial-R² mediation share; H, A, Π use β-attenuation.

5 stages × 24 cells × 500 bootstraps = 60,000 regressions.

Output: analysis/data/deep_determinants/subsample_stability.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.mediation import (
    mediation_share_with_ci,
    mediation_share_partial_r2_with_ci,
)
from analysis.paper5_horserace.exercise1_shapley import (
    CLIMATE_BUNDLE, OUTCOMES, CONTROLS,
)
from analysis.paper5_horserace.subsamples import SUBSAMPLE_STAGES

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/subsample_stability.parquet"

SCALAR_SUBSTRATES = [
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]


def main() -> None:
    df = pd.read_parquet(PANEL)

    pathway_cols = sorted(
        [c for c in df.columns if c.startswith("pathway_") and c != "pathway_0"]
    )
    print(f"Pathway dummies: {pathway_cols}")

    cells = [("climate_bundle", list(CLIMATE_BUNDLE), "partial_r2")] + [
        (s, s, "beta_attenuation") for s in SCALAR_SUBSTRATES
    ]

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
            for substrate_key, sub_arg, method in cells:
                seed = (44 + abs(hash((stage, outcome, substrate_key)))) % (2**31)
                if method == "partial_r2":
                    r = mediation_share_partial_r2_with_ci(
                        sub, y_col=outcome, substrate=sub_arg,
                        pathway_dummies=pathway_cols,
                        controls=CONTROLS, n_boot=500, seed=seed,
                    )
                else:
                    r = mediation_share_with_ci(
                        sub, y_col=outcome, substrate=sub_arg,
                        pathway_dummies=pathway_cols,
                        controls=CONTROLS, n_boot=500, seed=seed,
                    )
                rows.append(
                    {
                        "stage": stage,
                        "stage_idx": stage_idx,
                        "outcome": outcome,
                        "substrate": substrate_key,
                        "method": method,
                        "n_countries": n_countries,
                        **r,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")

    print("\nMedian |mediation share| by stage:")
    for stage, grp in out_df.groupby("stage"):
        med_abs = grp["mediation_share"].dropna().abs().median()
        in_unit = ((grp["mediation_share"] >= 0) &
                   (grp["mediation_share"] <= 1)).mean()
        print(f"  {stage:35s}  |med|={med_abs:.3f}  frac_in_[0,1]={in_unit:.2f}")


if __name__ == "__main__":
    main()
