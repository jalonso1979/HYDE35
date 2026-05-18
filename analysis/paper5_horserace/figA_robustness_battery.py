# analysis/paper5_horserace/figA_robustness_battery.py
"""Appendix figure: summary of all six robustness checks.

Layout: 3 rows x 2 cols (6 panels).
Each panel is a 4-substrate x 4-outcome heatmap showing the
key statistic for that check:
  (a) Continent FE  — mediation share
  (b) LOO median    — LOO median mediation share
  (c) WY-adjusted   — |t-obs|, with stars for p_adj_wy < 0.05 / 0.10
  (d) Climate 1421-1500 placebo — Shapley R²
  (e) Climate 1950-2008 placebo — Shapley R²
  (f) Pre-1950 outcome heterogeneity — Shapley R² (only dt_timing_year outcome)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/robustness_battery.parquet"
BASELINE = ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/figA_robustness_battery.pdf"

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

SUB_LABELS = {
    "sigma_v_T_pre1750": r"$\sigma_v^T$ (climate vol.)",
    "H_pred_pwadj": r"$\hat{H}$ (heterozygosity)",
    "ancestral_yield_log": "Crop yield (log)",
    "pandemic_intensity_norm": "Pandemic intensity",
}
OUT_LABELS = {
    "log_pop_growth_1950_2025": r"$\Delta\ln\text{Pop}$",
    "urban_change_1950_2025": r"$\Delta\text{Urban}$",
    "log_gdppc_2015": r"$\ln\text{GDPpc}$",
    "dt_timing_year": "DT timing",
}


def _heatmap(ax, pivot: pd.DataFrame, title: str, cmap: str = "RdBu_r",
             center: float = 0.0, fmt: str = ".2f",
             annot_override: pd.DataFrame | None = None,
             vmin=None, vmax=None) -> None:
    """Draw a heatmap in `ax` with the given pivot table."""
    data = pivot.reindex(index=SUBSTRATES, columns=OUTCOMES)
    # Clamp for display but keep original for annotation
    if vmin is None:
        vmax_abs = np.nanpercentile(np.abs(data.values), 95)
        vmin, vmax = -vmax_abs, vmax_abs

    im = ax.imshow(data.values, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([OUT_LABELS[o] for o in OUTCOMES], fontsize=7, rotation=20, ha="right")
    ax.set_yticks(range(len(SUBSTRATES)))
    ax.set_yticklabels([SUB_LABELS[s] for s in SUBSTRATES], fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    # Annotations
    annot_df = annot_override if annot_override is not None else data
    for i, s in enumerate(SUBSTRATES):
        for j, o in enumerate(OUTCOMES):
            try:
                val = annot_df.loc[s, o]
            except KeyError:
                val = np.nan
            if np.isfinite(val):
                text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6,
                        color="white" if abs(im.norm(data.values[i, j]
                                               if not np.isnan(data.values[i, j])
                                               else 0) - 0.5) > 0.3 else "black")


def main() -> None:
    df = pd.read_parquet(DATA)
    base = pd.read_parquet(BASELINE)

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle("Robustness battery: six checks", fontsize=12, fontweight="bold", y=0.98)

    # -----------------------------------------------------------------------
    # Panel (a): Baseline mediation (for reference)
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    piv_base = base.pivot(index="substrate", columns="outcome", values="mediation_share")
    _heatmap(ax, piv_base, "(a) Baseline mediation share",
             cmap="RdBu_r", center=0.0, fmt=".2f", vmin=-3, vmax=3)

    # -----------------------------------------------------------------------
    # Panel (b): Continent FE mediation
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    cfe = df[df["check"] == "continent_fe"]
    piv_cfe = cfe.pivot(index="substrate", columns="outcome", values="mediation_share")
    _heatmap(ax, piv_cfe, "(b) Continent FE mediation share",
             cmap="RdBu_r", center=0.0, fmt=".2f", vmin=-3, vmax=3)

    # -----------------------------------------------------------------------
    # Panel (c): LOO median
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    loo = df[df["check"] == "leave_one_out"]
    piv_loo = loo.pivot(index="substrate", columns="outcome", values="loo_median")
    _heatmap(ax, piv_loo, "(c) LOO median mediation share",
             cmap="RdBu_r", center=0.0, fmt=".2f", vmin=-3, vmax=3)

    # -----------------------------------------------------------------------
    # Panel (d): Westfall-Young |t| with significance stars
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    wy = df[df["check"] == "wy_correction"]
    piv_t = wy.pivot(index="substrate", columns="outcome", values="t_obs")
    # Build annotation pivot with stars
    piv_p = wy.pivot(index="substrate", columns="outcome", values="p_adj_wy")
    annot_wy = piv_t.copy().astype(str)
    for s in SUBSTRATES:
        for o in OUTCOMES:
            try:
                t = piv_t.loc[s, o]
                p = piv_p.loc[s, o]
                star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                annot_wy.loc[s, o] = f"{t:.2f}{star}"
            except KeyError:
                annot_wy.loc[s, o] = ""
    # Draw heatmap using |t| values (all positive)
    data_t = piv_t.reindex(index=SUBSTRATES, columns=OUTCOMES)
    im = ax.imshow(data_t.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|t-stat|")
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([OUT_LABELS[o] for o in OUTCOMES], fontsize=7, rotation=20, ha="right")
    ax.set_yticks(range(len(SUBSTRATES)))
    ax.set_yticklabels([SUB_LABELS[s] for s in SUBSTRATES], fontsize=7)
    ax.set_title("(d) WY-corrected |t-stat| (***p<0.01, **p<0.05, *p<0.10)", fontsize=9,
                 fontweight="bold", pad=4)
    for i, s in enumerate(SUBSTRATES):
        for j, o in enumerate(OUTCOMES):
            try:
                text = annot_wy.loc[s, o]
            except KeyError:
                text = ""
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    # -----------------------------------------------------------------------
    # Panel (e): Climate placebos — Shapley R² for sigma_v_T only
    # -----------------------------------------------------------------------
    ax = axes[2, 0]
    # Compare baseline vs pre-1500 vs modern Shapley R² for sigma_v_T
    from analysis.paper5_horserace.exercise1_shapley import SUBSTRATES as S_LIST
    base_shapley = pd.read_parquet(
        ROOT / "analysis/data/deep_determinants/exercise1_shapley_results.parquet"
    )
    base_s = base_shapley[base_shapley.substrate == "sigma_v_T_pre1750"].set_index("outcome")["shapley_r2"]

    p1500 = df[df["check"] == "placebo_pre1500_window"]
    p1500_s = p1500[p1500.substrate == "sigma_v_T_pre1750"].set_index("outcome")["shapley_r2"]
    pmod = df[df["check"] == "placebo_modern_window"]
    pmod_s = pmod[pmod.substrate == "sigma_v_T_pre1750"].set_index("outcome")["shapley_r2"]

    x = np.arange(len(OUTCOMES))
    w = 0.25
    bars1 = ax.bar(x - w, [base_s.get(o, 0) for o in OUTCOMES], w, label="Baseline (1421-1750)", color="#2166ac")
    bars2 = ax.bar(x, [p1500_s.get(o, 0) for o in OUTCOMES], w, label="Placebo: 1421-1500", color="#f4a582")
    bars3 = ax.bar(x + w, [pmod_s.get(o, 0) for o in OUTCOMES], w, label="Placebo: 1950-2008", color="#d6604d")
    ax.set_xticks(x)
    ax.set_xticklabels([OUT_LABELS[o] for o in OUTCOMES], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Shapley R²", fontsize=8)
    ax.set_title(r"(e) Climate-window placebos: $\sigma_v^T$ Shapley R²", fontsize=9, fontweight="bold", pad=4)
    ax.legend(fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # Panel (f): Pre-1950 early-transition Shapley R²
    # -----------------------------------------------------------------------
    ax = axes[2, 1]
    p1950_shapley = df[df["check"] == "pre1900_outcome_shapley"]

    # Full-sample baseline for dt_timing_year
    full_dt = base_shapley[base_shapley.outcome == "dt_timing_year"].set_index("substrate")["shapley_r2"]
    early_dt = p1950_shapley.set_index("substrate")["shapley_r2"]

    x = np.arange(len(SUBSTRATES))
    w = 0.35
    ax.bar(x - w / 2, [full_dt.get(s, 0) for s in SUBSTRATES], w,
           label="Full sample (n=144)", color="#2166ac")
    ax.bar(x + w / 2, [early_dt.get(s, 0) for s in SUBSTRATES], w,
           label="Early transition (<1950, n=44)", color="#d6604d")
    ax.set_xticks(x)
    ax.set_xticklabels([SUB_LABELS[s] for s in SUBSTRATES], fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Shapley R²", fontsize=8)
    ax.set_title("(f) Outcome heterogeneity: DT timing, full vs early-transition",
                 fontsize=9, fontweight="bold", pad=4)
    ax.legend(fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, bbox_inches="tight", dpi=150)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
