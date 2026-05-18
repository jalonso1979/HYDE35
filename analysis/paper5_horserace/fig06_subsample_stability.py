# analysis/paper5_horserace/fig06_subsample_stability.py
"""Figure 6: sub-sample stability ribbon for the headline mediation-share
estimates.

Layout: 4 rows (outcomes) × 4 columns (substrates).
Each panel: line + percentile-bootstrap CI band across the 5 sub-sample stages.
X-axis: stage labels. Y-axis: mediation share (shared across panels).
Reference lines at 0 and 1.

Output: analysis/figures/paper5_horserace/fig06_subsample_stability.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/subsample_stability.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig06_subsample_stability.pdf"

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
SUBSTRATE_LABELS = {
    "sigma_v_T_pre1750": r"$\sigma_v^T$ (climate vol.)",
    "H_pred_pwadj": "Predicted Het.",
    "ancestral_yield_log": "Anc. crop yield",
    "pandemic_intensity_norm": "Pre-1500 pandemic",
}
OUTCOME_LABELS = {
    "log_pop_growth_1950_2025": r"$\Delta\log P$",
    "urban_change_1950_2025": r"$\Delta$ Urban",
    "log_gdppc_2015": r"$\log\!GDPpc$",
    "dt_timing_year": "DT timing",
}

STAGES = [
    "baseline",
    "drop_ajr_colonial",
    "drop_small_island",
    "drop_americas_post_1492",
    "drop_ag_imputed",
]
STAGE_LABELS = [
    "Baseline",
    r"$-$AJR" + "\ncolonial",
    r"$-$island",
    r"$-$Americas",
    r"$-$AG-imp.",
]


def main() -> None:
    df = pd.read_parquet(DATA)

    fig, axes = plt.subplots(
        4, 4,
        figsize=(13, 11),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        "Subsample stability of mediation shares (95% bootstrap CI)",
        fontsize=12,
        y=1.01,
    )

    x = np.arange(len(STAGES))

    for i, outcome in enumerate(OUTCOMES):
        for j, substrate in enumerate(SUBSTRATES):
            ax = axes[i, j]

            # Filter and reorder rows to match STAGES order
            sub = df[(df["outcome"] == outcome) & (df["substrate"] == substrate)]
            sub = sub.set_index("stage").reindex(STAGES)

            y = sub["mediation_share"].values
            lo = sub["ci_lower"].values
            hi = sub["ci_upper"].values

            ax.fill_between(x, lo, hi, alpha=0.20, color="C0", label="95% CI")
            ax.plot(x, y, "o-", color="C0", linewidth=1.5,
                    markersize=4, label="Med. share")

            # Reference lines
            ax.axhline(0, color="black", linewidth=0.7, linestyle="-")
            ax.axhline(1, color="gray", linewidth=0.7, linestyle="--")

            ax.set_xticks(x)
            if i == 3:
                ax.set_xticklabels(STAGE_LABELS, rotation=35, ha="right",
                                   fontsize=7.5)
            else:
                ax.set_xticklabels([])

            # Column header (substrate) on top row
            if i == 0:
                ax.set_title(SUBSTRATE_LABELS[substrate], fontsize=9.5,
                             pad=4)

            # Row header (outcome) on leftmost column
            if j == 0:
                ax.set_ylabel(OUTCOME_LABELS[outcome], fontsize=9)

            # Light grid
            ax.yaxis.grid(True, linewidth=0.3, linestyle=":", color="gray")
            ax.set_axisbelow(True)

            # Annotate n at baseline stage
            n_baseline = sub.loc["baseline", "n_obs"] if "n_obs" in sub.columns else None
            if n_baseline is not None and not np.isnan(n_baseline):
                ax.annotate(
                    f"n={int(n_baseline)}",
                    xy=(0, ax.get_ylim()[0]),
                    xytext=(0.03, 0.04),
                    textcoords="axes fraction",
                    fontsize=6.5,
                    color="gray",
                )

    # Shared y-axis label
    fig.text(0.02, 0.5, "Mediation share", va="center",
             rotation="vertical", fontsize=10)

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, bbox_inches="tight", dpi=150)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
