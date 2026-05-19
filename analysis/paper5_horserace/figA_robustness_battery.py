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

Run:
    python -m analysis.paper5_horserace.figA_robustness_battery          # colour
    python -m analysis.paper5_horserace.figA_robustness_battery --bw     # grayscale B&W
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# B&W toggle — set to True to produce grayscale output
# ---------------------------------------------------------------------------
BW = False  # default; overridden by --bw CLI flag

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/robustness_battery.parquet"
BASELINE = ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/figA_robustness_battery.pdf"
FIG_BW = ROOT / "analysis/figures/paper5_horserace/figA_robustness_battery_bw.pdf"

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


def _bw_div_cmap(vmin: float, vmax: float, vcenter: float = 0.0):
    """Return a diverging white-centred grayscale colormap normalised to [vmin,vmax]."""
    center_frac = (vcenter - vmin) / (vmax - vmin) if (vmax - vmin) != 0 else 0.5
    center_frac = np.clip(center_frac, 0.05, 0.95)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bw_div",
        [(0.0, "0.12"), (center_frac, "white"), (1.0, "0.12")],
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def _heatmap(ax, pivot: pd.DataFrame, title: str, cmap: str = "RdBu_r",
             center: float = 0.0, fmt: str = ".2f",
             annot_override: pd.DataFrame | None = None,
             vmin=None, vmax=None, bw: bool = False) -> None:
    """Draw a heatmap in `ax` with the given pivot table."""
    data = pivot.reindex(index=SUBSTRATES, columns=OUTCOMES)
    # Clamp for display but keep original for annotation
    if vmin is None:
        vmax_abs = np.nanpercentile(np.abs(data.values), 95)
        vmin, vmax = -vmax_abs, vmax_abs

    if bw:
        cmap_use, norm = _bw_div_cmap(vmin, vmax, vcenter=center)
        im = ax.imshow(data.values, cmap=cmap_use, norm=norm, aspect="auto")
    else:
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
                raw_v = data.values[i, j] if not np.isnan(data.values[i, j]) else 0
                norm_v = im.norm(raw_v)
                if bw:
                    # For symmetric BW map: text white on dark, black on white
                    center_frac = (center - vmin) / (vmax - vmin) if (vmax - vmin) else 0.5
                    dist = abs(norm_v - center_frac)
                    text_color = "white" if dist > 0.35 else "black"
                else:
                    text_color = "white" if abs(norm_v - 0.5) > 0.3 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=6,
                        color=text_color)


def main() -> None:
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw
    print(f"Mode: {'B&W grayscale' if BW else 'colour'}")

    df = pd.read_parquet(DATA)
    base = pd.read_parquet(BASELINE)

    out = FIG_BW if BW else FIG

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle("Robustness battery: six checks", fontsize=12, fontweight="bold", y=0.98)

    # -----------------------------------------------------------------------
    # Panel (a): Baseline mediation (for reference)
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    piv_base = base.pivot(index="substrate", columns="outcome", values="mediation_share")
    _heatmap(ax, piv_base, "(a) Baseline mediation share",
             cmap="RdBu_r", center=0.0, fmt=".2f", vmin=-3, vmax=3, bw=BW)

    # -----------------------------------------------------------------------
    # Panel (b): Continent FE mediation
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    cfe = df[df["check"] == "continent_fe"]
    piv_cfe = cfe.pivot(index="substrate", columns="outcome", values="mediation_share")
    _heatmap(ax, piv_cfe, "(b) Continent FE mediation share",
             cmap="RdBu_r", center=0.0, fmt=".2f", vmin=-3, vmax=3, bw=BW)

    # -----------------------------------------------------------------------
    # Panel (c): LOO median
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    loo = df[df["check"] == "leave_one_out"]
    piv_loo = loo.pivot(index="substrate", columns="outcome", values="loo_median")
    _heatmap(ax, piv_loo, "(c) LOO median mediation share",
             cmap="RdBu_r", center=0.0, fmt=".2f", vmin=-3, vmax=3, bw=BW)

    # -----------------------------------------------------------------------
    # Panel (d): Westfall-Young |t| with significance stars
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    wy = df[df["check"] == "wy_correction"]
    piv_t = wy.pivot(index="substrate", columns="outcome", values="t_obs")
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
    wy_cmap = "Greys" if BW else "YlOrRd"
    im = ax.imshow(data_t.values, cmap=wy_cmap, aspect="auto", vmin=0, vmax=6)
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
            raw_v = data_t.values[i, j] if not np.isnan(data_t.values[i, j]) else 0
            norm_v = im.norm(raw_v)
            txt_color = "white" if norm_v > 0.65 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=txt_color)

    # -----------------------------------------------------------------------
    # Panel (e): Climate placebos — Shapley R² for sigma_v_T only
    # -----------------------------------------------------------------------
    ax = axes[2, 0]
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
    if BW:
        # Use grayscale fills + hatch patterns for distinction
        c1, c2, c3 = "white", "0.6", "0.25"
        h1, h2, h3 = "///", "...", "xxx"
        ec = "black"
    else:
        c1, c2, c3 = "#2166ac", "#f4a582", "#d6604d"
        h1, h2, h3 = None, None, None
        ec = None
    bar_kw = lambda c, h: dict(color=c, hatch=h, edgecolor=ec if h else c)
    ax.bar(x - w, [base_s.get(o, 0) for o in OUTCOMES], w,
           label="Baseline (1421-1750)", **bar_kw(c1, h1))
    ax.bar(x, [p1500_s.get(o, 0) for o in OUTCOMES], w,
           label="Placebo: 1421-1500", **bar_kw(c2, h2))
    ax.bar(x + w, [pmod_s.get(o, 0) for o in OUTCOMES], w,
           label="Placebo: 1950-2008", **bar_kw(c3, h3))
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

    full_dt = base_shapley[base_shapley.outcome == "dt_timing_year"].set_index("substrate")["shapley_r2"]
    early_dt = p1950_shapley.set_index("substrate")["shapley_r2"]

    x = np.arange(len(SUBSTRATES))
    w = 0.35
    if BW:
        cf1, cf2 = "white", "0.45"
        hf1, hf2 = "///", "..."
    else:
        cf1, cf2 = "#2166ac", "#d6604d"
        hf1, hf2 = None, None
    ax.bar(x - w / 2, [full_dt.get(s, 0) for s in SUBSTRATES], w,
           label="Full sample (n=144)", **bar_kw(cf1, hf1))
    ax.bar(x + w / 2, [early_dt.get(s, 0) for s in SUBSTRATES], w,
           label="Early transition (<1950, n=44)", **bar_kw(cf2, hf2))
    ax.set_xticks(x)
    ax.set_xticklabels([SUB_LABELS[s] for s in SUBSTRATES], fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Shapley R²", fontsize=8)
    ax.set_title("(f) Outcome heterogeneity: DT timing, full vs early-transition",
                 fontsize=9, fontweight="bold", pad=4)
    ax.legend(fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
