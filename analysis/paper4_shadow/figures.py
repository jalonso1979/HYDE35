"""Figure generation for Paper 4 — The Long Shadow of Seasonality (Figs 3-9).

All figures save to ``analysis/figures/paper4/`` at 150 dpi.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "paper4"

PATHWAY_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
PATHWAY_LABELS = {
    0: "Intensive",
    1: "Early ext.",
    2: "Crop-dom.",
    3: "Pastoral",
    4: "Irrigation",
}


def ensure_out_dir() -> Path:
    """Create the output directory if it does not exist and return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# --------------------------------------------------------------------------- #
# Fig 3 — Seasonality predicts pathway
# --------------------------------------------------------------------------- #

def fig3_seasonality_predicts_pathway(
    xs: pd.DataFrame,
    seasonality_col: str = "seasonality_historical",
    pathway_col: str = "cluster",
) -> Path:
    """Horizontal box plots of seasonality distribution per pathway.

    Scatter points overlaid with alpha=0.5.  Returns path to saved figure.
    """
    ensure_out_dir()

    df = xs[[seasonality_col, pathway_col]].dropna()
    pathways = sorted(df[pathway_col].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_pathway = [
        df.loc[df[pathway_col] == pw, seasonality_col].values for pw in pathways
    ]
    labels = [PATHWAY_LABELS.get(pw, str(pw)) for pw in pathways]

    bp = ax.boxplot(
        data_by_pathway,
        vert=False,
        patch_artist=True,
        labels=labels,
        widths=0.6,
    )

    for i, (patch, pw) in enumerate(zip(bp["boxes"], pathways)):
        color = PATHWAY_COLORS[i % len(PATHWAY_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

        y_pos = i + 1
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data_by_pathway[i]))
        ax.scatter(
            data_by_pathway[i],
            y_pos + jitter,
            color=color,
            alpha=0.5,
            s=18,
            zorder=3,
        )

    ax.set_xlabel("Intra-annual Seasonality (σ_s)")
    ax.set_title("Does Climate Seasonality Predict Agricultural Pathway?")
    fig.tight_layout()

    out = OUT_DIR / "fig3_seasonality_pathway.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Fig 4 — Dual-channel DAG (conceptual, no data)
# --------------------------------------------------------------------------- #

def fig4_dual_channel_dag() -> Path:
    """Pure conceptual diagram of the dual-channel causal model."""
    ensure_out_dir()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Node positions: (x, y)
    nodes = {
        "seasonality": (1.0, 3.8),
        "volatility":  (1.0, 1.2),
        "ag_system":   (3.8, 3.8),
        "malthus":     (3.8, 1.2),
        "escape":      (6.6, 2.5),
        "modern":      (9.0, 2.5),
    }

    node_text = {
        "seasonality": "Intra-annual\nSeasonality\n(σs)",
        "volatility":  "Inter-annual\nVolatility\n(σv)",
        "ag_system":   "Agricultural\nSystem\nSelection",
        "malthus":     "Malthusian\nFeedback\nStrength",
        "escape":      "Speed of\nEscape",
        "modern":      "Modern\nOutcomes",
    }

    bbox_props = dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        linewidth=1.5,
    )

    for key, (x, y) in nodes.items():
        ax.text(
            x, y, node_text[key],
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=bbox_props,
        )

    # Arrows — Stage 1 (seasonality chain, red)
    red_arrows = [
        ("seasonality", "ag_system"),
        ("ag_system", "escape"),
        ("escape", "modern"),
    ]
    # Arrows — Stage 2 (volatility chain, blue)
    blue_arrows = [
        ("volatility", "malthus"),
        ("malthus", "escape"),
    ]

    def _draw_arrow(src_key, dst_key, color):
        sx, sy = nodes[src_key]
        dx, dy = nodes[dst_key]
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", lw=2.5, color=color),
        )

    for src, dst in red_arrows:
        _draw_arrow(src, dst, "#d62728")
    for src, dst in blue_arrows:
        _draw_arrow(src, dst, "#1f77b4")

    # Legend
    ax.plot([], [], color="#d62728", lw=2.5, label="Stage 1: Seasonality → System → Escape")
    ax.plot([], [], color="#1f77b4", lw=2.5, label="Stage 2: Volatility → Malthus → Escape")
    ax.legend(loc="lower center", fontsize=9, ncol=2, frameon=False)

    ax.set_title("Dual-Channel Causal Model", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()

    out = OUT_DIR / "fig4_dual_channel_dag.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Fig 5 — Rolling Malthusian coefficient by pathway
# --------------------------------------------------------------------------- #

def fig5_rolling_malthusian_by_pathway(roll_df: pd.DataFrame) -> Path:
    """Line plot of time-varying Malthusian coefficient per pathway.

    Parameters
    ----------
    roll_df : DataFrame
        Must have columns: center_year, coefficient, pathway.
    """
    ensure_out_dir()

    fig, ax = plt.subplots(figsize=(9, 5))

    pathways = sorted(roll_df["pathway"].unique())
    for i, pw in enumerate(pathways):
        sub = roll_df[roll_df["pathway"] == pw].sort_values("center_year")
        color = PATHWAY_COLORS[i % len(PATHWAY_COLORS)]
        label = PATHWAY_LABELS.get(pw, str(pw))
        ax.plot(
            sub["center_year"], sub["coefficient"],
            marker="o", markersize=4,
            color=color, label=label, linewidth=1.8,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Center Year")
    ax.set_ylabel("Malthusian Coefficient")
    ax.set_title("Time-Varying Malthusian Coefficient by Agricultural Pathway")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out = OUT_DIR / "fig5_rolling_malthus_pathway.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Fig 6 — Pathway-stratified IRFs
# --------------------------------------------------------------------------- #

def fig6_pathway_stratified_irfs(
    irf_dict: dict[str, pd.DataFrame],
    shock_label: str = "Temp",
) -> Path:
    """Subplot per pathway showing impulse-response with confidence band.

    Parameters
    ----------
    irf_dict : dict
        Maps pathway_label -> DataFrame with columns
        ``horizon``, ``coefficient``, ``ci_lower``, ``ci_upper``.
    shock_label : str
        Label used in title and filename.
    """
    ensure_out_dir()

    n = len(irf_dict)
    ncols = min(n, 4)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4), squeeze=False)

    for idx, (pw_label, df) in enumerate(irf_dict.items()):
        ax = axes[0, idx]
        df = df.sort_values("horizon")
        ax.plot(df["horizon"], df["coefficient"], "o-", color="black", markersize=4)
        ax.fill_between(
            df["horizon"], df["ci_lower"], df["ci_upper"],
            alpha=0.25, color="steelblue",
        )
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(pw_label, fontsize=10)
        ax.set_xlabel("Horizon")
        if idx == 0:
            ax.set_ylabel("Response")

    fig.suptitle(
        f"Impulse–Response to {shock_label} Shock by Pathway",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fname = f"fig6_irf_by_pathway_{shock_label.lower()}.png"
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Fig 7 — Escape rolling
# --------------------------------------------------------------------------- #

def fig7_escape_rolling(roll_df: pd.DataFrame) -> Path:
    """Rolling escape coefficient — stratified by pathway or aggregate with CI.

    Parameters
    ----------
    roll_df : DataFrame
        Must have ``center_year`` and ``coefficient``.
        If ``pathway`` column present → stratified lines.
        Otherwise uses ``se`` column for CI fill.
    """
    ensure_out_dir()

    fig, ax = plt.subplots(figsize=(9, 5))

    stratified = "pathway" in roll_df.columns

    if stratified:
        pathways = sorted(roll_df["pathway"].unique())
        for i, pw in enumerate(pathways):
            sub = roll_df[roll_df["pathway"] == pw].sort_values("center_year")
            color = PATHWAY_COLORS[i % len(PATHWAY_COLORS)]
            label = PATHWAY_LABELS.get(pw, str(pw))
            ax.plot(
                sub["center_year"], sub["coefficient"],
                marker="o", markersize=3,
                color=color, label=label, linewidth=1.6,
            )
        ax.legend(fontsize=8, loc="best")
    else:
        roll_df = roll_df.sort_values("center_year")
        ax.plot(
            roll_df["center_year"], roll_df["coefficient"],
            "o-", color="black", markersize=4, linewidth=1.6,
        )
        if "se" in roll_df.columns:
            ci_lo = roll_df["coefficient"] - 1.96 * roll_df["se"]
            ci_hi = roll_df["coefficient"] + 1.96 * roll_df["se"]
            ax.fill_between(
                roll_df["center_year"], ci_lo, ci_hi,
                alpha=0.2, color="steelblue",
            )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(1970, color="gray", linewidth=1.2, linestyle="--")
    ax.text(
        1970, ax.get_ylim()[1] * 0.95, " Green Revolution",
        fontsize=8, va="top", ha="left", color="gray",
    )

    ax.set_xlabel("Center Year")
    ax.set_ylabel("Escape Coefficient")
    ax.set_title("Rolling Escape from the Malthusian Trap")
    fig.tight_layout()

    out = OUT_DIR / "fig7_escape_rolling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Fig 8 — Escape mechanism (interaction coefficients)
# --------------------------------------------------------------------------- #

def fig8_escape_mechanism(interaction_results: dict) -> Path:
    """Bar chart of interaction coefficients from escape-mechanism regressions.

    Parameters
    ----------
    interaction_results : dict
        Output of ``run_escape_interactions``.  Keys are mediator names plus
        ``"joint"``.  Each value has ``"model"`` key with a fitted OLS result.
    """
    import statsmodels.api as sm  # noqa: F811

    ensure_out_dir()

    # Collect individual mediator results (skip "joint")
    mediators = [k for k in interaction_results if k != "joint"]

    labels = []
    coefs = []
    errors = []

    for med in mediators:
        res = interaction_results[med]
        if res is None:
            continue
        model = res["model"]
        # Find interaction term (column with "_x_" in name)
        interaction_terms = [p for p in model.params.index if "_x_" in p]
        if not interaction_terms:
            continue
        iterm = interaction_terms[0]
        coefs.append(model.params[iterm])
        errors.append(1.96 * model.bse[iterm])
        labels.append(med.replace("_", " ").title())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: individual mediator bars
    ax = axes[0]
    x_pos = np.arange(len(labels))
    bar_colors = ["#4c72b0" if c < 0 else "#dd8452" for c in coefs]
    ax.bar(x_pos, coefs, yerr=errors, capsize=5, color=bar_colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Interaction Coefficient")
    ax.set_title("Individual Mediators")

    # Right panel: joint model interaction terms
    ax2 = axes[1]
    joint_res = interaction_results.get("joint")
    if joint_res is not None and joint_res is not None:
        model_j = joint_res["model"]
        j_terms = [p for p in model_j.params.index if "_x_" in p]
        j_labels = [t.split("_x_")[-1].replace("_", " ").title() for t in j_terms]
        j_coefs = [model_j.params[t] for t in j_terms]
        j_errors = [1.96 * model_j.bse[t] for t in j_terms]
        j_colors = ["#4c72b0" if c < 0 else "#dd8452" for c in j_coefs]
        j_pos = np.arange(len(j_labels))
        ax2.bar(j_pos, j_coefs, yerr=j_errors, capsize=5, color=j_colors, alpha=0.8, edgecolor="black")
        ax2.set_xticks(j_pos)
        ax2.set_xticklabels(j_labels, fontsize=9, rotation=15, ha="right")
        ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_ylabel("Interaction Coefficient")
    ax2.set_title("Joint Model")

    fig.suptitle("Escape Mechanism: Interaction Coefficients", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = OUT_DIR / "fig8_escape_mechanism.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Fig 9 — Long shadow scatter grid
# --------------------------------------------------------------------------- #

def fig9_long_shadow(
    xs: pd.DataFrame,
    seasonality_col: str = "seasonality_historical",
) -> Path:
    """2x2 scatter grid: seasonality vs d_crop, pop_gr, d_urban, d_irrig.

    OLS fit line in red with beta and p-value annotated in each subplot title.
    """
    import statsmodels.api as sm  # noqa: F811

    ensure_out_dir()

    outcomes = ["d_crop", "pop_gr", "d_urban", "d_irrig"]
    outcome_labels = {
        "d_crop": "Δ Crop Share",
        "pop_gr": "Log Pop. Growth",
        "d_urban": "Δ Urban Share",
        "d_irrig": "Δ Irrigation Share",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, outcome in zip(axes.flat, outcomes):
        sub = xs[[seasonality_col, outcome]].dropna()
        x = sub[seasonality_col].values
        y = sub[outcome].values

        ax.scatter(x, y, alpha=0.5, s=20, color="steelblue", edgecolors="none")

        # OLS fit
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit(cov_type="HC1")
        beta = model.params[1]
        pval = model.pvalues[1]

        x_grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_grid, model.predict(sm.add_constant(x_grid)), color="red", linewidth=1.8)

        label = outcome_labels.get(outcome, outcome)
        ax.set_title(f"{label}  (β={beta:.3f}, p={pval:.3f})", fontsize=10)
        ax.set_xlabel("Seasonality (σ_s)")
        ax.set_ylabel(label)

    fig.suptitle("The Long Shadow of Seasonality", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = OUT_DIR / "fig9_long_shadow.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
