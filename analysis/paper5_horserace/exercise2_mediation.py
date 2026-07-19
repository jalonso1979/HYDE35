# analysis/paper5_horserace/exercise2_mediation.py
"""Exercise 2: mediation-share decomposition by pathway, for each
(outcome, substrate) pair.

Four substrates, six outcomes => 24 cells:
  - climate bundle (T̄, P̄, σᵥᵀ, σᵥᴾ): partial-R² mediation share
  - 3 individual substrates (H, A, Π): β-attenuation mediation share

Output: analysis/data/deep_determinants/exercise2_mediation_results.parquet
"""
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.mediation import (
    mediation_share_with_ci,
    mediation_share_partial_r2_with_ci,
    stable_seed,
)
from analysis.paper5_horserace.exercise1_shapley import (
    CLIMATE_BUNDLE, FUNCTIONAL_BUNDLE, OUTCOMES, CONTROLS,
)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet"

SCALAR_SUBSTRATES = [
    "neolithic_frac",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]

PATHWAY_DUMMIES = ["pathway_1", "pathway_2", "pathway_3", "pathway_4"]


def main() -> None:
    df = pd.read_parquet(PANEL)

    rows = []
    cells = [
        ("climate_bundle", list(CLIMATE_BUNDLE), "partial_r2"),
        ("functional_alleles", list(FUNCTIONAL_BUNDLE), "partial_r2"),
    ] + [(s, s, "beta_attenuation") for s in SCALAR_SUBSTRATES]
    total = len(OUTCOMES) * len(cells)
    i = 0
    for outcome in OUTCOMES:
        for substrate_key, sub_arg, method in cells:
            i += 1
            seed = stable_seed("mediation", outcome, substrate_key)
            print(f"[{i:02d}/{total}] {outcome} ~ {substrate_key}  ({method})  seed={seed}")
            if method == "partial_r2":
                r = mediation_share_partial_r2_with_ci(
                    df,
                    y_col=outcome,
                    substrate=sub_arg,
                    pathway_dummies=PATHWAY_DUMMIES,
                    controls=CONTROLS,
                    n_boot=1000,
                    seed=seed,
                )
            else:
                r = mediation_share_with_ci(
                    df,
                    y_col=outcome,
                    substrate=sub_arg,
                    pathway_dummies=PATHWAY_DUMMIES,
                    controls=CONTROLS,
                    n_boot=1000,
                    seed=seed,
                )
            rows.append({"outcome": outcome, "substrate": substrate_key,
                         "method": method, **r})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")

    pivot = out_df.pivot(index="substrate", columns="outcome", values="mediation_share")
    pivot = pivot.reindex(
        index=["climate_bundle", "functional_alleles"] + SCALAR_SUBSTRATES,
        columns=OUTCOMES,
    )
    print("\nMediation shares (5 substrates × 6 outcomes):")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
