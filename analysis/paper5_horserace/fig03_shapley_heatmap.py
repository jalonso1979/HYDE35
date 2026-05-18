# analysis/paper5_horserace/fig03_shapley_heatmap.py
"""Figure 3: Shapley R² heatmap (substrates × outcomes) + Table 4 (LaTeX)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/exercise1_shapley_results.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig03_shapley_heatmap.pdf"
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
    df = pd.read_parquet(DATA)
    df = df.rename(columns={"substrate": "Substrate", "outcome": "Outcome"})
    df["Substrate"] = df["Substrate"].map(SUBSTRATE_LABELS)
    df["Outcome"] = df["Outcome"].map(OUTCOME_LABELS)
    pivot = df.pivot(index="Substrate", columns="Outcome", values="shapley_r2")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={"label": "Shapley $R^2$"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG, bbox_inches="tight")
    print(f"Wrote {FIG}")

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
