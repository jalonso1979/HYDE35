# analysis/paper5_horserace/exercise3_climate_clusters.py
"""Exercise 3: re-run mediation analysis using HYDE-features-based pathway
dummies as the mediator, instead of the climate-only-clustered pathways used
in Exercise 2.

Disambiguation:
  - Exercise 2 used pathway_0..4 (from climate_pathways_country.parquet),
    which are K=5 KMeans clusters built on pre-industrial climate primitives:
    productive_months, sigma_s, sigma_p, t_mean, p_mean, t_volatility (1421-1750).
  - Exercise 3 uses hyde_pathway_0..4 (from paper1_clustered_features.parquet),
    built on HYDE-trajectory features: peak_ag_expansion_year,
    peak_pop_growth_year, max_density, density/crop/urban shares in 1750, etc.
    These are downstream outcomes of the same processes the model explains,
    providing an orthogonal robustness check.

The substantive question: does the suppressor structure (mediation shares
outside [0,1]) persist under HYDE-clustered pathways, or does one clustering
clean it up?

Output: analysis/data/deep_determinants/exercise3_climate_cluster_results.parquet
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.mediation import mediation_share_with_ci

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise3_climate_cluster_results.parquet"

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

# hyde_pathway_0 is the reference category (omitted); use 1, 3, 4
# (cluster 2 was a singleton and was dropped in build_climate_pathway_dummies)
HYDE_PATHWAY_DUMMIES = ["hyde_pathway_1", "hyde_pathway_3", "hyde_pathway_4"]


def main() -> None:
    df = pd.read_parquet(PANEL)

    # Verify HYDE pathway dummies exist
    missing = [c for c in HYDE_PATHWAY_DUMMIES if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing HYDE pathway dummy columns: {missing}. "
            "Run: python -m analysis.paper5_horserace.build_climate_pathway_dummies"
        )

    n_hyde = df[HYDE_PATHWAY_DUMMIES[0]].notna().sum()
    print(f"=== Exercise 3: HYDE-pathway mediation (N≈{n_hyde} countries with HYDE clusters) ===\n")

    rows = []
    total = len(OUTCOMES) * len(SUBSTRATES)
    i = 0
    for outcome in OUTCOMES:
        for substrate in SUBSTRATES:
            i += 1
            seed = abs(hash((outcome, substrate, "hyde"))) % (2**31)
            print(f"[{i:02d}/{total}] {outcome} ~ {substrate}  (seed={seed})")
            r = mediation_share_with_ci(
                df,
                y_col=outcome,
                substrate=substrate,
                pathway_dummies=HYDE_PATHWAY_DUMMIES,
                controls=CONTROLS,
                n_boot=1000,
                seed=seed,
            )
            rows.append({
                "outcome": outcome,
                "substrate": substrate,
                "mediator": "hyde_features",
                **r,
            })
            print(f"       mediation_share={r['mediation_share']:+.4f}  "
                  f"CI=[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  "
                  f"N={r['n_obs']}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out_df)} rows)")

    pivot = out_df.pivot(index="substrate", columns="outcome", values="mediation_share")
    print("\n=== Exercise 3 mediation shares (HYDE-clustered pathways) ===")
    print(pivot.round(3).to_string())

    # Summary comparison
    outside = ((out_df["mediation_share"] < 0) | (out_df["mediation_share"] > 1))
    print(f"\nFraction outside [0,1]: {outside.sum()}/{len(out_df)} = {outside.mean():.1%}")
    print(f"Median |mediation_share|: {out_df['mediation_share'].abs().median():.4f}")
    print(f"Mean   |mediation_share|: {out_df['mediation_share'].abs().mean():.4f}")


if __name__ == "__main__":
    main()
