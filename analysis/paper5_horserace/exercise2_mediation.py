# analysis/paper5_horserace/exercise2_mediation.py
"""Exercise 2: mediation-share decomposition by pathway, for each
(outcome, substrate) pair.

Output: analysis/data/deep_determinants/exercise2_mediation_results.parquet
"""
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.mediation import mediation_share_with_ci

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet"

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
# pathway_0 is the reference category (omitted to avoid perfect collinearity)
PATHWAY_DUMMIES = ["pathway_1", "pathway_2", "pathway_3", "pathway_4"]


def main() -> None:
    df = pd.read_parquet(PANEL)

    rows = []
    total = len(OUTCOMES) * len(SUBSTRATES)
    i = 0
    for outcome in OUTCOMES:
        for substrate in SUBSTRATES:
            i += 1
            seed = abs(hash((outcome, substrate))) % (2**31)
            print(f"[{i:02d}/{total}] {outcome} ~ {substrate}  (seed={seed})")
            r = mediation_share_with_ci(
                df,
                y_col=outcome,
                substrate=substrate,
                pathway_dummies=PATHWAY_DUMMIES,
                controls=CONTROLS,
                n_boot=1000,
                seed=seed,
            )
            rows.append({"outcome": outcome, "substrate": substrate, **r})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")

    pivot = out_df.pivot(index="substrate", columns="outcome", values="mediation_share")
    print("\nMediation shares (4 substrates × 4 outcomes):")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
