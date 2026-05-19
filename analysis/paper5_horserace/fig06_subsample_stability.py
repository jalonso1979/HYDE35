# analysis/paper5_horserace/fig06_subsample_stability.py
"""Figure 6: sub-sample stability ribbon for the headline mediation-share
estimates.

Layout: 4 rows (outcomes) × 4 columns (substrates).
Each panel: line + percentile-bootstrap CI band across the 5 sub-sample stages.
X-axis: stage labels. Y-axis: mediation share (shared across panels).
Reference lines at 0 and 1.

Output: analysis/figures/paper5_horserace/fig06_subsample_stability.pdf

Run:
    python -m analysis.paper5_horserace.fig06_subsample_stability          # colour
    python -m analysis.paper5_horserace.fig06_subsample_stability --bw     # grayscale B&W
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# B&W toggle — set to True to produce grayscale output
# ---------------------------------------------------------------------------
BW = False  # default; overridden by --bw CLI flag

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/subsample_stability.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig06_subsample_stability.pdf"
FIG_BW = ROOT / "analysis/figures/paper5_horserace/fig06_subsample_stability_bw.pdf"

SUBSTRATES = [
    "climate_bundle",
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]
OUTCOMES = [
    "log_popd_1500",
    "log_popd_2025",
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]
SUBSTRATE_LABELS = {
    "climate_bundle": "Climate bundle",
    "H_pred_pwadj": "Predicted Het.",
    "ancestral_yield_log": "Anc. crop yield",
    "pandemic_intensity_norm": "Pre-1500 pandemic",
}
OUTCOME_LABELS = {
    "log_popd_1500": r"$\log D_{1500}$",
    "log_popd_2025": r"$\log D_{2025}$",
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
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw
    print(f"Mode: {'B&W grayscale' if BW else 'colour'}")

    df = pd.read_parquet(DATA)

    # B&W style: single dark line with hatched fill band
    line_color = "black" if BW else "C0"
    fill_color = "0.6" if BW else "C0"
    fill_alpha = 0.30 if BW else 0.20
    fill_hatch = "///" if BW else None

    out = FIG_BW if BW else FIG

    fig, axes = plt.subplots(
        len(OUTCOMES), len(SUBSTRATES),
        figsize=(13, 2.5 * len(OUTCOMES) + 1),
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

            # Ribbon (filled CI band)
            fill_kw = dict(alpha=fill_alpha, color=fill_color, label="95% CI")
            if fill_hatch:
                fill_kw["hatch"] = fill_hatch
                fill_kw["edgecolor"] = "0.5"
                fill_kw["facecolor"] = "none"
                fill_kw.pop("color")
                fill_kw.pop("alpha")
            ax.fill_between(x, lo, hi, **fill_kw)
            ax.plot(x, y, "o-", color=line_color, linewidth=1.5,
                    markersize=4, label="Med. share")

            # Reference lines
            ax.axhline(0, color="black", linewidth=0.7, linestyle="-")
            ax.axhline(1, color="0.45" if BW else "gray",
                       linewidth=0.7, linestyle="--")

            ax.set_xticks(x)
            if i == len(OUTCOMES) - 1:
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
                    color="0.4" if BW else "gray",
                )

    # Shared y-axis label
    fig.text(0.02, 0.5, "Mediation share", va="center",
             rotation="vertical", fontsize=10)

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
