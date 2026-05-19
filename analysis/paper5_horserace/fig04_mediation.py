# analysis/paper5_horserace/fig04_mediation.py
"""Figure 4: mediation-share diagram + Table 5 (LaTeX) showing how each
substrate's effect on each outcome is mediated by the agricultural pathway.

NOTE on interpretation: the bootstrap mediation shares span a very wide range
(-2.16 to +5.59), indicating suppressor effects in most cells. The one clean,
interpretable mediation is pandemic_intensity_norm -> log_gdppc_2015 (~0.47).
All other cells have wide CIs and/or point estimates outside [0, 1]. The figure
uses y-axis limits of (-3, 6) plus symlog stretch and annotates suppressor /
over-mediation cells to make this visually unambiguous.

Run:
    python -m analysis.paper5_horserace.fig04_mediation          # colour
    python -m analysis.paper5_horserace.fig04_mediation --bw     # grayscale B&W
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# B&W toggle — set to True to produce grayscale output
# ---------------------------------------------------------------------------
BW = False  # default; overridden by --bw CLI flag

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig04_mediation.pdf"
FIG_BW = ROOT / "analysis/figures/paper5_horserace/fig04_mediation_bw.pdf"
TAB = ROOT / "analysis/figures/paper5_horserace/tab05_mediation_table.tex"

SUBSTRATE_LABELS = {
    "climate_bundle": r"Climate bundle",
    "H_pred_pwadj": r"Pred. Het.",
    "ancestral_yield_log": r"Anc. crop",
    "pandemic_intensity_norm": "Pre-1500\npandemic",
}

OUTCOME_LABELS = {
    "log_popd_1500": r"$\log D_{1500}$",
    "log_popd_2025": r"$\log D_{2025}$",
    "log_pop_growth_1950_2025": r"$\Delta\log\!P_{50\text{-}25}$",
    "urban_change_1950_2025": r"$\Delta$ Urban$_{50\text{-}25}$",
    "log_gdppc_2015": r"$\log\!GDPpc_{15}$",
    "dt_timing_year": "DT timing",
}

SUBSTRATE_ORDER = list(SUBSTRATE_LABELS.keys())
OUTCOME_ORDER = list(OUTCOME_LABELS.keys())

# Y-axis clip: wide but not distorted by the extreme pandemic->pop outlier (~5.58)
YMIN, YMAX = -3.0, 6.0


def classify_cell(y: float, ci_lo: float, ci_hi: float) -> str:
    """Return annotation label for suppressor / over-mediation cells."""
    clean = ci_lo >= -0.1 and ci_hi <= 1.1  # CI entirely in [0,1] ± small buffer
    if clean and 0 <= y <= 1:
        return ""  # interpretable mediation — no annotation
    if y < 0:
        return "supp−"     # suppressor (negative mediation)
    if y > 1:
        return "over+"  # over-mediation
    # point in [0,1] but CI extremely wide
    if (ci_hi - ci_lo) > 10:
        return "wide CI"
    return ""


def main() -> None:
    global BW
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw", action="store_true", help="Produce grayscale B&W version")
    args = parser.parse_args()
    BW = args.bw
    print(f"Mode: {'B&W grayscale' if BW else 'colour'}")

    df = pd.read_parquet(DATA)

    # ------------------------------------------------------------------ #
    # Figure 4                                                             #
    # ------------------------------------------------------------------ #
    # Marker / linestyle differentiation for B&W
    MARKERS = ["o", "s", "^", "D"]   # per substrate position
    if BW:
        color_interp = "black"
        color_suppressor = "0.55"
        refline_color = "black"
        ecolor = "0.4"
    else:
        color_interp = "C0"
        color_suppressor = "C1"
        refline_color = "C3"
        ecolor = "gray"

    out = FIG_BW if BW else FIG

    fig, axes = plt.subplots(1, 6, figsize=(18, 4.5), sharey=True)

    for ax, outcome in zip(axes, OUTCOME_ORDER):
        sub = df[df["outcome"] == outcome].copy()
        sub = sub.set_index("substrate").loc[SUBSTRATE_ORDER].reset_index()
        x = np.arange(len(sub))
        y = sub["mediation_share"].values
        ci_lo = sub["ci_lower"].values
        ci_hi = sub["ci_upper"].values
        yerr_low = np.clip(y - ci_lo, 0, None)  # always non-negative
        yerr_hi = np.clip(ci_hi - y, 0, None)

        for i in range(len(sub)):
            label = classify_cell(y[i], ci_lo[i], ci_hi[i])
            is_suppressor = label != ""
            pt_color = color_suppressor if is_suppressor else color_interp
            marker = MARKERS[i] if BW else "o"
            # In B&W use open markers for suppressors for extra distinction
            fillstyle = "none" if (BW and is_suppressor) else "full"
            ax.errorbar(
                x[i], y[i],
                yerr=[[yerr_low[i]], [yerr_hi[i]]],
                fmt=marker, capsize=4,
                color=pt_color, ecolor=ecolor, elinewidth=1, zorder=3,
                fillstyle=fillstyle, markersize=7,
            )
            if label:
                # Annotate clipped points — place text at clip boundary
                ytext = min(max(y[i], YMIN + 0.1), YMAX - 0.3)
                ax.annotate(
                    label,
                    xy=(x[i], ytext),
                    xytext=(x[i] + 0.08, ytext + 0.25),
                    fontsize=6.5, color=pt_color,
                    ha="left",
                )
                # Draw arrow when point is above clip
                if y[i] > YMAX:
                    ax.annotate(
                        "",
                        xy=(x[i], YMAX - 0.05),
                        xytext=(x[i], YMAX - 0.5),
                        arrowprops=dict(arrowstyle="->", color=pt_color, lw=1.2),
                    )

        # Reference lines
        ax.axhline(0, color="black", linewidth=0.6, zorder=2)
        ax.axhline(1, color=refline_color, linewidth=0.8, linestyle="--",
                   zorder=2, label="Full mediation")

        ax.set_ylim(YMIN, YMAX)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SUBSTRATE_LABELS[s] for s in sub["substrate"]],
            rotation=30, ha="right", fontsize=9,
        )
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=11)
        ax.tick_params(axis="y", labelsize=8)

    axes[0].set_ylabel("Mediation share", fontsize=10)

    # Shared legend
    if BW:
        interp_handle = plt.Line2D([0], [0], marker="o", color="black",
                                   linestyle="None", label="Interpretable [0, 1]",
                                   markersize=7)
        supp_handle = plt.Line2D([0], [0], marker="o", color="0.55",
                                 linestyle="None", fillstyle="none",
                                 label="Suppressor / over-mediation", markersize=7)
        ref_line = plt.Line2D([0], [0], color="black", linestyle="--",
                              linewidth=0.9, label="Full mediation (=1)")
        # Marker legend (per substrate, B&W only)
        marker_handles = [
            plt.Line2D([0], [0], marker=MARKERS[k], color="black",
                       linestyle="None", markersize=6,
                       label=list(SUBSTRATE_LABELS.values())[k])
            for k in range(4)
        ]
        fig.legend(
            handles=[interp_handle, supp_handle, ref_line] + marker_handles,
            loc="upper center", ncol=4, fontsize=8,
            bbox_to_anchor=(0.5, 1.04), frameon=False,
        )
    else:
        blue_patch = mpatches.Patch(color="C0", label="Interpretable [0, 1]")
        orange_patch = mpatches.Patch(color="C1", label="Suppressor / over-mediation")
        red_line = plt.Line2D([0], [0], color="C3", linestyle="--", linewidth=0.9,
                              label="Full mediation (=1)")
        fig.legend(
            handles=[blue_patch, orange_patch, red_line],
            loc="upper center", ncol=3, fontsize=8.5,
            bbox_to_anchor=(0.5, 1.02), frameon=False,
        )

    # Footnote
    fig.text(
        0.5, -0.05,
        (
            "Note: Mediation share = indirect effect / total effect (bootstrap 1000 reps, HC3)."
            " Climate-bundle uses partial-R² mediation; H, A, $\\Pi$ use $\\beta$-attenuation."
            " Values outside [0,1] indicate suppressor structure or over-mediation."
        ),
        ha="center", fontsize=7.5, style="italic", wrap=True,
    )

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")

    # ------------------------------------------------------------------ #
    # Table 5 — LaTeX                                                      #
    # ------------------------------------------------------------------ #
    # Build pivots indexed by substrate, columns = outcomes
    piv_share = df.pivot(index="substrate", columns="outcome", values="mediation_share")
    piv_lo = df.pivot(index="substrate", columns="outcome", values="ci_lower")
    piv_hi = df.pivot(index="substrate", columns="outcome", values="ci_upper")

    # Reorder to canonical substrate / outcome ordering
    piv_share = piv_share.loc[SUBSTRATE_ORDER, OUTCOME_ORDER]
    piv_lo = piv_lo.loc[SUBSTRATE_ORDER, OUTCOME_ORDER]
    piv_hi = piv_hi.loc[SUBSTRATE_ORDER, OUTCOME_ORDER]

    with open(TAB, "w") as f:
        n_outcomes = len(OUTCOME_ORDER)
        col_spec = "l" + "r" * n_outcomes
        col_headers = " & ".join(OUTCOME_LABELS[o] for o in OUTCOME_ORDER)

        f.write("% Table 5: Agricultural-pathway mediation shares (bootstrap 95 pct CI)\n")
        f.write("% Cells: point estimate [ci_lower, ci_upper]\n")
        f.write("% Values outside [0,1] indicate suppressor / over-mediation effects.\n")
        f.write(r"\begin{tabular}{" + col_spec + "}\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Substrate} & " + col_headers + r" \\" + "\n")
        f.write(r"\midrule" + "\n")

        for sub in SUBSTRATE_ORDER:
            row_label = SUBSTRATE_LABELS[sub].replace("\n", " ")
            cells = []
            for out in OUTCOME_ORDER:
                share = piv_share.loc[sub, out]
                lo = piv_lo.loc[sub, out]
                hi = piv_hi.loc[sub, out]
                cell_str = f"{share:.2f} [{lo:.2f},\\;{hi:.2f}]"
                # Bold cleanly-interpretable cells (point in [0,1] and CI almost there)
                if 0 <= share <= 1 and lo >= -0.1 and hi <= 1.2:
                    cell_str = r"\textbf{" + cell_str + r"}"
                cells.append(cell_str)
            f.write(row_label + " & " + " & ".join(cells) + r" \\" + "\n")

        f.write(r"\midrule" + "\n")
        f.write(
            r"\multicolumn{"
            + str(n_outcomes + 1)
            + r"}{p{0.95\textwidth}}{\footnotesize "
            r"\textit{Mediation share = indirect / total effect, bootstrapped (1000 reps) "
            r"with HC3 SEs. Climate-bundle uses partial-R² mediation; other substrates use "
            r"$\beta$-attenuation. Bolded cells have point estimate in [0,\,1] with CI "
            r"nearly contained; the rest indicate suppressor or over-mediation structure.}}"
            r" \\" + "\n"
        )
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    print(f"Wrote {TAB}")


if __name__ == "__main__":
    main()
