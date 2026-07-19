# analysis/paper5_horserace/fig03_shapley_heatmap.py
"""Figure 3: Shapley R² heatmap (substrates × outcomes) + Table 4 (LaTeX).

Run:
    python -m analysis.paper5_horserace.fig03_shapley_heatmap          # colour
    python -m analysis.paper5_horserace.fig03_shapley_heatmap --bw     # grayscale B&W
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# B&W toggle — set to True to produce grayscale output
# ---------------------------------------------------------------------------
BW = False  # default; overridden by --bw CLI flag

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/exercise1_shapley_results.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig03_shapley_heatmap.pdf"
FIG_BW = ROOT / "analysis/figures/paper5_horserace/fig03_shapley_heatmap_bw.pdf"
TAB = ROOT / "analysis/figures/paper5_horserace/tab04_shapley_table.tex"

SUBSTRATE_LABELS = {
    "climate_bundle": r"Climate bundle",
    "functional_alleles": r"Functional alleles",
    "neolithic_frac": r"Neolithic frac.",
    "ancestral_yield_log": r"Anc.\ crop yield",
    "pandemic_intensity_norm": r"Pre-1500 pandemic int.",
}
SUBSTRATE_ORDER = ["climate_bundle", "functional_alleles",
                   "neolithic_frac", "ancestral_yield_log",
                   "pandemic_intensity_norm"]
OUTCOME_LABELS = {
    "log_popd_1500": r"$\log D_{1500}$",
    "log_popd_2025": r"$\log D_{2025}$",
    "log_pop_growth_1950_2025": r"$\Delta\log\!P_{50\!-\!25}$",
    "urban_change_1950_2025": r"$\Delta$ Urban$_{50\!-\!25}$",
    "log_gdppc_2015": r"$\log\!GDPpc_{15}$",
    "dt_timing_year": "DT timing",
}
OUTCOME_ORDER = ["log_popd_1500", "log_popd_2025", "log_pop_growth_1950_2025",
                 "urban_change_1950_2025", "log_gdppc_2015", "dt_timing_year"]


def main() -> None:
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw
    print(f"Mode: {'B&W grayscale' if BW else 'colour'}")

    df = pd.read_parquet(DATA)
    # Preserve canonical ordering before mapping to display labels
    df = df.rename(columns={"substrate": "Substrate", "outcome": "Outcome"})
    pivot = df.pivot(index="Substrate", columns="Outcome", values="shapley_r2")
    pivot = pivot.reindex(index=SUBSTRATE_ORDER, columns=OUTCOME_ORDER)
    pivot.index = [SUBSTRATE_LABELS[s] for s in pivot.index]
    pivot.columns = [OUTCOME_LABELS[o] for o in pivot.columns]

    # Shapley R² is always >= 0, so sequential grayscale works perfectly in B&W.
    cmap = "Greys" if BW else "YlGnBu"
    out = FIG_BW if BW else FIG

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                cbar_kws={"label": "Shapley $R^2$"}, ax=ax,
                annot_kws={"size": 9})
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")

    # LaTeX table — uses raw outcome codes for groupby then maps to display labels
    full_minus_base = (
        df.groupby("Outcome")[["full_model_r2", "baseline_r2"]]
        .first()
        .apply(lambda r: r["full_model_r2"] - r["baseline_r2"], axis=1)
    )
    full_minus_base = full_minus_base.reindex(OUTCOME_ORDER)

    with open(TAB, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(pivot.columns) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(pivot.columns) + " \\\\\n\\midrule\n")
        for row in pivot.index:
            cells = [f"{v:.3f}" for v in pivot.loc[row]]
            f.write(row + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n")
        f.write("Total ($\\Sigma$) & " +
                " & ".join(f"{full_minus_base[o]:.3f}" for o in OUTCOME_ORDER) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {TAB}")


if __name__ == "__main__":
    main()
