# analysis/paper5_horserace/fig05_pathway_source_robustness.py
"""Figure 5: side-by-side mediation-share heatmaps comparing two pathway
clustering strategies.

Left panel:  Climate-only-clustered pathways (Exercise 2)
             pathway_0..4 from K=5 KMeans on climate primitives
             (productive_months, sigma_s, sigma_p, t_mean, p_mean, t_volatility)

Right panel: HYDE-features-clustered pathways (Exercise 3)
             hyde_pathway_0..4 from K=5 KMeans on HYDE-trajectory outcomes
             (peak_ag_expansion_year, peak_pop_growth_year, max_density,
              density/crop/urban shares in 1750, ag_intensity_change)

The suppressor-structure question: does the fraction of mediation shares
outside [0,1] persist under both clustering strategies, or does one cluster
source clean it up?
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("/Volumes/BIGDATA/HYDE35")
CLIMATE_RES = ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet"
HYDE_RES = ROOT / "analysis/data/deep_determinants/exercise3_climate_cluster_results.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig05_pathway_source_robustness.pdf"

# Short labels for readability
SUBSTRATE_LABELS = {
    "sigma_v_T_pre1750":       "Climate\nvolatility",
    "H_pred_pwadj":            "Predicted\nheterozygosity",
    "ancestral_yield_log":     "Ancestral\ncrop yield",
    "pandemic_intensity_norm": "Pandemic\nintensity",
}
OUTCOME_LABELS = {
    "log_pop_growth_1950_2025": "Pop growth\n1950-2025",
    "urban_change_1950_2025":   "Urban\nchange",
    "log_gdppc_2015":           "log GDPpc\n2015",
    "dt_timing_year":           "Dem. transition\nyear",
}

# Ordered for display
SUBSTRATE_ORDER = [
    "sigma_v_T_pre1750",
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]
OUTCOME_ORDER = [
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]


def _load_pivot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    piv = df.pivot(index="substrate", columns="outcome", values="mediation_share")
    piv = piv.loc[SUBSTRATE_ORDER, OUTCOME_ORDER]
    piv.index = [SUBSTRATE_LABELS[s] for s in SUBSTRATE_ORDER]
    piv.columns = [OUTCOME_LABELS[o] for o in OUTCOME_ORDER]
    return piv


def _stats(path: Path) -> dict:
    df = pd.read_parquet(path)
    outside = ((df["mediation_share"] < 0) | (df["mediation_share"] > 1))
    return {
        "n_outside": int(outside.sum()),
        "n_total": len(df),
        "pct_outside": float(outside.mean()) * 100,
        "median_abs": float(df["mediation_share"].abs().median()),
        "mean_abs": float(df["mediation_share"].abs().mean()),
    }


def _heatmap(ax: plt.Axes, piv: pd.DataFrame, title: str,
             stats: dict, show_cbar: bool = False) -> None:
    """Draw a clipped heatmap with annotations showing raw values."""
    vals = piv.values

    # Clip display to [-3, 3] for color scale; annotate with raw values
    vmin, vmax = -3.0, 3.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)

    im = ax.imshow(vals, cmap="RdBu_r", norm=norm, aspect="auto")
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Mediation share", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=8.5)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8.5)

    # Annotate each cell with the raw value
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            # White text on dark cells, dark on light
            bg_norm = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
            text_color = "white" if bg_norm < 0.15 or bg_norm > 0.85 else "#202020"
            is_suppressor = not (0 <= v <= 1)
            marker = " *" if is_suppressor else ""
            ax.text(c, r, f"{v:+.2f}{marker}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold" if is_suppressor else "normal")

    ax.set_title(title, fontsize=10, pad=8)
    foot = (f"Outside [0,1]: {stats['n_outside']}/{stats['n_total']} "
            f"({stats['pct_outside']:.0f}%)\n"
            f"Median |share|: {stats['median_abs']:.3f}")
    ax.set_xlabel(foot, fontsize=8, labelpad=10)


def main() -> None:
    climate_piv = _load_pivot(CLIMATE_RES)
    hyde_piv = _load_pivot(HYDE_RES)
    climate_stats = _stats(CLIMATE_RES)
    hyde_stats = _stats(HYDE_RES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(wspace=0.38)

    _heatmap(axes[0], climate_piv,
             "Climate-only-clustered pathways (Exercise 2)\n"
             r"$\it{Mediator:\ K=5\ on\ climate\ primitives}$",
             climate_stats, show_cbar=False)
    _heatmap(axes[1], hyde_piv,
             "HYDE-features-clustered pathways (Exercise 3)\n"
             r"$\it{Mediator:\ K=5\ on\ HYDE\ trajectory\ outcomes}$",
             hyde_stats, show_cbar=True)

    # Stars = outside [0,1] (suppressor)
    fig.text(0.5, 0.01,
             "* = mediation share outside [0,1] (suppressor / amplifier effect). "
             "Color scale clipped to [−3, 3]; raw values annotated.",
             ha="center", fontsize=8, color="#404040")

    fig.suptitle(
        "Figure 5 — Mediation-share robustness: pathway-source comparison\n"
        "4 substrates × 4 modern outcomes",
        fontsize=11, y=1.01, x=0.04, ha="left",
    )

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG}")

    # Summary
    print(f"\n=== Comparison summary ===")
    print(f"Exercise 2 (climate-only clusters):")
    print(f"  Outside [0,1]: {climate_stats['n_outside']}/{climate_stats['n_total']} "
          f"({climate_stats['pct_outside']:.0f}%)")
    print(f"  Median |share|: {climate_stats['median_abs']:.4f}")
    print(f"  Mean   |share|: {climate_stats['mean_abs']:.4f}")
    print(f"\nExercise 3 (HYDE-features clusters):")
    print(f"  Outside [0,1]: {hyde_stats['n_outside']}/{hyde_stats['n_total']} "
          f"({hyde_stats['pct_outside']:.0f}%)")
    print(f"  Median |share|: {hyde_stats['median_abs']:.4f}")
    print(f"  Mean   |share|: {hyde_stats['mean_abs']:.4f}")

    print(f"\nVerdict:")
    if climate_stats["pct_outside"] > 30 and hyde_stats["pct_outside"] > 30:
        print("  SUPPRESSOR STRUCTURE PERSISTS under both clustering strategies.")
        print("  Both Ex2 and Ex3 have >30% of shares outside [0,1]. The pathway")
        print("  clustering source does NOT explain the suppressor pattern.")
    elif climate_stats["pct_outside"] > 30 and hyde_stats["pct_outside"] <= 30:
        print("  CLEANS UP under HYDE clustering. Suppressor is specific to climate-only mediator.")
    elif climate_stats["pct_outside"] <= 30 and hyde_stats["pct_outside"] > 30:
        print("  CLEANS UP under climate-only clustering. Suppressor is specific to HYDE mediator.")
    else:
        print("  Both clusterings produce well-behaved shares. No suppressor structure.")


if __name__ == "__main__":
    main()
