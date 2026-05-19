# analysis/paper5_horserace/exercise3_climate_clusters.py
"""Exercise 3: re-run mediation using HYDE-features-based pathway dummies
as the mediator, with the same 4-substrate × 6-outcome matrix as Exercise 2.

Climate bundle uses partial-R² mediation; H, A, Π use β-attenuation.

Output: analysis/data/deep_determinants/exercise3_climate_cluster_results.parquet
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

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise3_climate_cluster_results.parquet"

SCALAR_SUBSTRATES = [
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]

# hyde_pathway_0 is the reference category (omitted); use 1, 3, 4
HYDE_PATHWAY_DUMMIES = ["hyde_pathway_1", "hyde_pathway_3", "hyde_pathway_4"]


def main() -> None:
    df = pd.read_parquet(PANEL)
    missing = [c for c in HYDE_PATHWAY_DUMMIES if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing HYDE pathway dummy columns: {missing}. "
            "Run: python -m analysis.paper5_horserace.build_climate_pathway_dummies"
        )
    n_hyde = df[HYDE_PATHWAY_DUMMIES[0]].notna().sum()
    print(f"=== Exercise 3: HYDE-pathway mediation (N≈{n_hyde}) ===\n")

    cells = [("climate_bundle", list(CLIMATE_BUNDLE), "partial_r2")] + [
        (s, s, "beta_attenuation") for s in SCALAR_SUBSTRATES
    ]

    rows = []
    total = len(OUTCOMES) * len(cells)
    i = 0
    for outcome in OUTCOMES:
        for skey, sub_arg, method in cells:
            i += 1
            seed = abs(hash((outcome, skey, "hyde"))) % (2**31)
            print(f"[{i:02d}/{total}] {outcome} ~ {skey}  ({method})")
            if method == "partial_r2":
                r = mediation_share_partial_r2_with_ci(
                    df, y_col=outcome, substrate=sub_arg,
                    pathway_dummies=HYDE_PATHWAY_DUMMIES,
                    controls=CONTROLS, n_boot=1000, seed=seed,
                )
            else:
                r = mediation_share_with_ci(
                    df, y_col=outcome, substrate=sub_arg,
                    pathway_dummies=HYDE_PATHWAY_DUMMIES,
                    controls=CONTROLS, n_boot=1000, seed=seed,
                )
            rows.append({"outcome": outcome, "substrate": skey,
                         "mediator": "hyde_features", "method": method, **r})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")

    pivot = out_df.pivot(index="substrate", columns="outcome", values="mediation_share")
    pivot = pivot.reindex(
        index=["climate_bundle"] + SCALAR_SUBSTRATES,
        columns=OUTCOMES,
    )
    print("\nHYDE-clustered pathway mediation shares:")
    print(pivot.round(3).to_string())

    outside = ((out_df["mediation_share"] < 0) | (out_df["mediation_share"] > 1))
    print(f"\nFraction outside [0,1]: {outside.sum()}/{len(out_df)} = {outside.mean():.1%}")


if __name__ == "__main__":
    main()
