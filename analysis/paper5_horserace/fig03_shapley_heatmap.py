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
    "sigma_v_T_pre1750": r"$\sigma_v^T$ 1421--1750",
    "H_pred_pwadj": r"Predicted Het",
    "ancestral_yield_log": r"Anc.\ crop yield",
    "pandemic_intensity_norm": r"Pre-1500 pandemic int.",
}
OUTCOME_LABELS = {
    "log_pop_growth_1950_2025": r"$\Delta\log\!P_{50\!-\!25}$",
    "urban_change_1950_2025": r"$\Delta$ Urban$_{50\!-\!25}$",
    "log_gdppc_2015": r"$\log\!GDPpc_{15}$",
    "dt_timing_year": "DT timing",
}


def main() -> None:
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw
    print(f"Mode: {'B&W grayscale' if BW else 'colour'}")

    df = pd.read_parquet(DATA)
    df = df.rename(columns={"substrate": "Substrate", "outcome": "Outcome"})
    df["Substrate"] = df["Substrate"].map(SUBSTRATE_LABELS)
    df["Outcome"] = df["Outcome"].map(OUTCOME_LABELS)
    pivot = df.pivot(index="Substrate", columns="Outcome", values="shapley_r2")

    # Shapley R² is always >= 0, so sequential grayscale works perfectly in B&W.
    # Use Greys (light=low, dark=high). Annotation colour adapts to background.
    cmap = "Greys" if BW else "YlGnBu"
    out = FIG_BW if BW else FIG

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                cbar_kws={"label": "Shapley $R^2$"}, ax=ax,
                annot_kws={"size": 9})
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")

    # LaTeX table
    with open(TAB, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(pivot.columns) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(pivot.columns) + " \\\\\n\\midrule\n")
        for row in pivot.index:
            cells = [f"{v:.3f}" for v in pivot.loc[row]]
            f.write(row + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n")
        # bottom row: total R² (full minus baseline)
        full_minus_base = (
            df.groupby("Outcome")[["full_model_r2", "baseline_r2"]]
            .first()
            .apply(lambda r: r["full_model_r2"] - r["baseline_r2"], axis=1)
        )
        f.write("Total ($\\Sigma$) & " +
                " & ".join(f"{full_minus_base[c]:.3f}" for c in pivot.columns) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {TAB}")


if __name__ == "__main__":
    main()
