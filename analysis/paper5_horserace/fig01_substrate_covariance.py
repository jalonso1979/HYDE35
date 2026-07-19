"""Figure 1: pairwise scatter of four substrates, with marginal histograms
and pairwise correlations.  Also writes Table 1 (descriptives) and
Table 2 (4x4 substrate correlation matrix) as LaTeX.

Run:
    python -m analysis.paper5_horserace.fig01_substrate_covariance          # colour
    python -m analysis.paper5_horserace.fig01_substrate_covariance --bw     # grayscale B&W
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
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig01_substrate_covariance.pdf"
FIG_BW = ROOT / "analysis/figures/paper5_horserace/fig01_substrate_covariance_bw.pdf"
TAB1 = ROOT / "analysis/figures/paper5_horserace/tab01_descriptives.tex"
TAB2 = ROOT / "analysis/figures/paper5_horserace/tab02_substrate_correlations.tex"
TAB2B = ROOT / "analysis/figures/paper5_horserace/tab02b_climate_subcorrelations.tex"

# Climate bundle members (for descriptives and within-climate sub-correlation)
CLIMATE_BUNDLE = [
    "t_mean_pre1750",
    "p_mean_pre1750",
    "sigma_v_T_pre1750",
    "sigma_v_P_pre1750",
]

# Headline 5 substrates for the cross-substrate scatter and 5×5 correlation table.
# σᵥᵀ represents the climate bundle; AMY1 represents the functional-allele bundle.
# Each is the most-interpretable single member of its coalition for display.
SUBSTRATES = [
    "sigma_v_T_pre1750",
    "fa_amy1",
    "neolithic_frac",
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
CONTROLS = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]

# Long labels for the pairplot axes
AXIS_LABELS = {
    "sigma_v_T_pre1750": r"$\sigma_v^T$ (climate rep.)",
    "fa_amy1": r"AMY1 (func. rep.)",
    "neolithic_frac": r"Neolithic ancestry frac.",
    "ancestral_yield_log": r"Anc. crop yield (log)",
    "pandemic_intensity_norm": r"Pre-1500 pandemic",
}

# Short labels for LaTeX table column headers
SHORT_LABELS = {
    "sigma_v_T_pre1750": r"$\sigma_v^T$",
    "fa_amy1": r"AMY1",
    "neolithic_frac": r"$\nu$",
    "ancestral_yield_log": r"Crop yield",
    "pandemic_intensity_norm": r"Pandemic",
    "t_mean_pre1750": r"$\bar T$",
    "p_mean_pre1750": r"$\bar P$",
    "sigma_v_P_pre1750": r"$\sigma_v^P$",
    "H_pred_pwadj": r"$H_i$",
}

FULL_LABELS = {
    "t_mean_pre1750": r"Climate: mean $T$ 1421--1750",
    "p_mean_pre1750": r"Climate: mean $P$ 1421--1750",
    "sigma_v_T_pre1750": r"Climate: $\sigma_v^T$ 1421--1750",
    "sigma_v_P_pre1750": r"Climate: $\sigma_v^P$ 1421--1750",
    "fa_lct": r"Functional: LCT (lactase)",
    "fa_adh1b": r"Functional: ADH1B (alcohol)",
    "fa_amy1": r"Functional: AMY1 (starch)",
    "fa_edar": r"Functional: EDAR (E.\ Asian morphology)",
    "fa_darc": r"Functional: DARC (malaria/Duffy)",
    "fa_slc24a5": r"Functional: SLC24A5 (pigmentation)",
    "fa_hbb": r"Functional: HBB (malaria/sickle)",
    "fa_fads": r"Functional: FADS1/2 (PUFA)",
    "neolithic_frac": r"Neolithic ancestry fraction",
    "H_pred_pwadj": r"Predicted heterozygosity $H_i$ (demoted)",
    "ancestral_yield_log": r"Ancestral crop yield (log)",
    "pandemic_intensity_norm": r"Pre-1500 pandemic intensity",
    "log_popd_1500": r"$\log$ pop density 1500",
    "log_popd_2025": r"$\log$ pop density 2025",
    "log_pop_growth_1950_2025": r"$\Delta\log P_{1950\to2025}$",
    "urban_change_1950_2025": r"$\Delta$ urban share 1950--2025",
    "log_gdppc_2015": r"$\log$ GDPpc 2015",
    "dt_timing_year": r"DT timing year",
    "abs_lat": r"Absolute latitude",
    "log_area": r"$\log$ area",
    "landlocked": r"Landlocked",
    "ruggedness_proxy": r"Ruggedness (log)",
    "log_dist_neolithic": r"$\log$ distance from Neolithic",
}


# ---------------------------------------------------------------------------
# Table 1 — Descriptive statistics
# ---------------------------------------------------------------------------

def _descriptives_table(df: pd.DataFrame, out: Path) -> None:
    """Write mean / SD / N for substrates, outcomes, controls as a LaTeX tabular.

    Substrate block: 4 climate-bundle members + 8 functional alleles +
        Neolithic frac + A + Π (15 rows).
    Outcome block: 2 density + 4 modern = 6 rows.
    Controls: 5 geography vars.
    """

    def _row(c: str):
        s = df[c].dropna()
        return (FULL_LABELS.get(c, c), s.mean(), s.std(), len(s))

    functional_bundle = ["fa_lct", "fa_adh1b", "fa_amy1", "fa_edar",
                         "fa_darc", "fa_slc24a5", "fa_hbb", "fa_fads"]
    substrates_full = (CLIMATE_BUNDLE
                       + functional_bundle
                       + ["neolithic_frac", "ancestral_yield_log",
                          "pandemic_intensity_norm"])

    with open(out, "w") as f:
        f.write("% Table 1 — Descriptive statistics\n")
        f.write("% Generated by fig01_substrate_covariance.py\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Variable & Mean & SD & $N$ \\\\\n\\midrule\n")

        f.write("\\multicolumn{4}{l}{\\textit{Substrates}} \\\\\n")
        for c in substrates_full:
            lbl, mn, sd, n = _row(c)
            f.write(f"\\quad {lbl} & {mn:.3f} & {sd:.3f} & {n} \\\\\n")

        f.write("\\midrule\n\\multicolumn{4}{l}{\\textit{Outcomes}} \\\\\n")
        for c in OUTCOMES:
            lbl, mn, sd, n = _row(c)
            f.write(f"\\quad {lbl} & {mn:.3f} & {sd:.3f} & {n} \\\\\n")

        f.write("\\midrule\n\\multicolumn{4}{l}{\\textit{Geography controls}} \\\\\n")
        for c in CONTROLS:
            lbl, mn, sd, n = _row(c)
            f.write(f"\\quad {lbl} & {mn:.3f} & {sd:.3f} & {n} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Table 2 — Substrate correlation matrix
# ---------------------------------------------------------------------------

def _correlation_table(corr: pd.DataFrame, out: Path) -> None:
    """Write a 4x4 LaTeX correlation matrix; |r| > 0.4 in bold."""
    short_cols = [SHORT_LABELS.get(c, c) for c in corr.columns]
    short_rows = [SHORT_LABELS.get(c, c) for c in corr.index]

    with open(out, "w") as f:
        f.write("% Table 2 — Substrate pairwise correlations\n")
        f.write("% Generated by fig01_substrate_covariance.py\n")
        f.write("% Correlations |r| > 0.4 are in bold.\n")
        ncols = len(SUBSTRATES)
        f.write("\\begin{tabular}{l" + "r" * ncols + "}\n\\toprule\n")
        f.write(" & " + " & ".join(short_cols) + " \\\\\n\\midrule\n")
        for row_name, row_label in zip(corr.index, short_rows):
            cells = []
            for v in corr.loc[row_name]:
                if abs(v) > 0.4:
                    cells.append(f"\\textbf{{{v:.2f}}}")
                else:
                    cells.append(f"{v:.2f}")
            f.write(row_label + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 1 — Pairplot
# ---------------------------------------------------------------------------

def _pairplot(sub: pd.DataFrame, out: Path, bw: bool = False) -> None:
    """4x4 seaborn pairplot with regression lines and histograms on diagonal."""
    sns.set_theme(style="ticks", font_scale=0.9)

    renamed = sub.rename(columns=AXIS_LABELS)

    if bw:
        scatter_color = "0.3"
        line_color = "black"
        hist_color = "0.5"
        r_strong_color = "black"
        r_weak_color = "0.55"
    else:
        scatter_color = "steelblue"
        line_color = "C3"
        hist_color = "steelblue"
        r_strong_color = "C3"
        r_weak_color = "0.3"

    g = sns.pairplot(
        renamed,
        kind="reg",
        diag_kind="hist",
        height=2.2,
        plot_kws={
            "scatter_kws": {"alpha": 0.45, "s": 8, "color": scatter_color},
            "line_kws": {"color": line_color, "lw": 1.5},
            "ci": 95,
        },
        diag_kws={"bins": 20, "color": hist_color, "edgecolor": "white"},
    )

    # Tighten tick labels
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis="both", labelsize=7)

    # Add Pearson r annotations on upper triangle
    col_names = renamed.columns.tolist()
    for i in range(len(col_names)):
        for j in range(len(col_names)):
            if j > i:  # upper triangle
                ax = g.axes[i, j]
                x = sub.iloc[:, j]
                y = sub.iloc[:, i]
                mask = x.notna() & y.notna()
                if mask.sum() > 2:
                    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                    ax.set_visible(True)
                    strong = abs(r) > 0.4
                    ax.text(
                        0.5, 0.5,
                        f"r = {r:.2f}",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=9,
                        fontweight="bold" if strong else "normal",
                        color=r_strong_color if strong else r_weak_color,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # Remove spines for cleaner look
                    for spine in ax.spines.values():
                        spine.set_visible(False)

    g.figure.suptitle(
        "Figure 1: Substrate covariance — four pre-industrial deep determinants",
        y=1.01, fontsize=10,
    )
    g.figure.tight_layout()
    g.figure.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw

    df = pd.read_parquet(PANEL)
    print(f"Panel loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Mode: {'B&W grayscale' if BW else 'colour'}")

    # Table 1: descriptives on full sample (dropna per variable) — only in colour mode
    if not BW:
        _descriptives_table(df, TAB1)

    # Subset to rows with all four substrates present (σᵥᵀ represents climate bundle)
    sub = df[SUBSTRATES].dropna()
    print(f"Substrate subset (all 4 non-null): {len(sub)} rows")

    if not BW:
        # Table 2: 4×4 cross-substrate correlation matrix (σᵥᵀ as bundle representative)
        corr = sub.corr().round(3)
        print("\nSubstrate correlations:")
        print(corr.to_string())
        print(f"\nMax |r| off-diagonal: {corr.where(~np.eye(4, dtype=bool)).abs().max().max():.3f}")
        _correlation_table(corr, TAB2)

        # Table 2b: within-climate 4×4 correlation matrix
        clim_sub = df[CLIMATE_BUNDLE].dropna()
        clim_corr = clim_sub.corr().round(3)
        print("\nWithin-climate correlations:")
        print(clim_corr.to_string())
        _correlation_table(clim_corr, TAB2B)

    # Figure 1
    out = FIG_BW if BW else FIG
    out.parent.mkdir(parents=True, exist_ok=True)
    _pairplot(sub, out, bw=BW)

    print("\nAll outputs written.")


if __name__ == "__main__":
    main()
