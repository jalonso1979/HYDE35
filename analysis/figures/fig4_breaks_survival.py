"""
Figure 4: Timing of Malthusian Structural Breaks and Survival by Pathway
Two-panel figure for economic history paper using HYDE 3.5 data.

Panel A (top): Stacked histogram of earliest structural break year, by pathway (200-yr bins)
Panel B (bottom): Kaplan-Meier survival curves — probability of remaining in Malthusian regime
"""

import sys
sys.path.insert(0, "/Volumes/BIGDATA/HYDE35")

import numpy as np
import pandas as pd
import pycountry
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter

from analysis.shared.config import ANALYSIS_DATA
from analysis.paper2_malthus.panels import build_malthusian_panel
from analysis.paper2_malthus.breaks import detect_breaks_all_entities

# ── output paths ─────────────────────────────────────────────────────────────
OUT_PNG = "/Volumes/BIGDATA/HYDE35/analysis/figures/fig4_breaks_survival.png"
OUT_PDF = "/Volumes/BIGDATA/HYDE35/analysis/figures/fig4_breaks_survival.pdf"

# ── cluster metadata — match Figure 1 exactly ────────────────────────────────
LABELS = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}

# Wong 2011 colorblind-friendly palette (same as Fig 1)
COLORS = {
    0: "#E69F00",   # orange
    1: "#56B4E9",   # sky blue
    2: "#CC79A7",   # pink/magenta (singleton, excluded here)
    3: "#009E73",   # green
    4: "#0072B2",   # blue
}

# Clusters to plot (exclude 2 = Egypt singleton)
PLOT_CLUSTERS = [0, 1, 3, 4]

# ── data preparation ──────────────────────────────────────────────────────────
print("Loading data …")
cp = pd.read_parquet(ANALYSIS_DATA / "country_analysis_panel.parquet")
clustered = pd.read_parquet(ANALYSIS_DATA / "paper1_clustered_features.parquet")


def iso_num_to_alpha3(num):
    try:
        c = pycountry.countries.get(numeric=str(int(num)).zfill(3))
        return c.alpha_3 if c else str(int(num))
    except Exception:
        return str(int(num))


code_map = {c: iso_num_to_alpha3(c) for c in cp.country.unique()}
cp["iso3"] = cp["country"].map(code_map)
cp_post0 = cp[cp.year >= 0]
high_pop = cp_post0.groupby("iso3")["popdens_p_km2_mean"].max()
valid = high_pop[high_pop > 1.0].index
cp_f = cp_post0[cp_post0.iso3.isin(valid)]

print("Building Malthusian panel …")
malthus = build_malthusian_panel(cp_f, entity_col="iso3")

print("Detecting structural breaks (this may take a moment) …")
breaks = detect_breaks_all_entities(
    malthus, entity_col="iso3", min_segment=3, significance=0.05
)
earliest_break = (
    breaks.groupby("iso3")["break_year"]
    .min()
    .reset_index()
    .rename(columns={"break_year": "break_year"})
)

# Histogram dataset (only countries with detected breaks, exclude cluster 2)
merged = earliest_break.merge(clustered[["iso3", "cluster"]], on="iso3")
merged["pathway"] = merged["cluster"].map(LABELS)
merged = merged[merged.cluster != 2]

# Survival dataset (all countries in clusters, excluding cluster 2)
surv = clustered[["iso3", "cluster"]].copy()
surv = surv[surv.cluster != 2]
surv = surv.merge(earliest_break, on="iso3", how="left")
surv["duration"] = surv["break_year"].fillna(2000)
surv["event"] = surv["break_year"].notna().astype(int)
surv["pathway"] = surv["cluster"].map(LABELS)

print(f"Countries with breaks: {len(merged)}")
print(f"Survival dataset: {len(surv)} rows")
for c in PLOT_CLUSTERS:
    sub = surv[surv.cluster == c]
    print(f"  Cluster {c} ({LABELS[c]}): n={len(sub)}, events={sub['event'].sum()}")

# ── bin edges for histogram ───────────────────────────────────────────────────
BIN_EDGES = np.arange(200, 2001, 200)   # 200, 400, …, 2000
BIN_LABELS = [str(b) for b in BIN_EDGES]   # x tick labels at right edge of each bin

# Pre-bin each cluster
def assign_bin(year, edges):
    for e in edges:
        if year <= e:
            return e
    return edges[-1]

merged["bin"] = merged["break_year"].apply(assign_bin, edges=BIN_EDGES)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

fig, (ax_A, ax_B) = plt.subplots(
    2, 1, figsize=(10, 10),
    gridspec_kw={"hspace": 0.42}
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Stacked histogram of break years by pathway
# ─────────────────────────────────────────────────────────────────────────────
x_positions = np.arange(len(BIN_EDGES))
bar_width = 0.75

# Build count matrix: rows = bins, cols = clusters (in PLOT_CLUSTERS order)
all_bins = pd.DataFrame({"bin": BIN_EDGES})
bottom = np.zeros(len(BIN_EDGES))

legend_patches = []
for c_id in PLOT_CLUSTERS:
    sub = merged[merged.cluster == c_id]
    counts_s = sub.groupby("bin").size().reindex(BIN_EDGES, fill_value=0)
    counts = counts_s.values.astype(float)

    ax_A.bar(
        x_positions,
        counts,
        bottom=bottom,
        color=COLORS[c_id],
        width=bar_width,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=LABELS[c_id],
    )
    bottom += counts
    legend_patches.append(
        mpatches.Patch(color=COLORS[c_id], label=LABELS[c_id])
    )

ax_A.set_xticks(x_positions)
ax_A.set_xticklabels(BIN_LABELS, fontsize=9)
ax_A.set_xlabel("Break year (200-year intervals)", labelpad=6)
ax_A.set_ylabel("Number of countries", labelpad=6)
ax_A.set_title("(A)  Timing of Malthusian Structural Breaks by Pathway", loc="left",
               fontsize=11, fontweight="bold", pad=8)
ax_A.set_xlim(-0.5, len(BIN_EDGES) - 0.5)
ax_A.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax_A.legend(
    handles=legend_patches,
    fontsize=8.5,
    frameon=False,
    loc="upper left",
    ncol=2,
    handlelength=1.2,
    columnspacing=1.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B — Kaplan-Meier survival curves by pathway
# ─────────────────────────────────────────────────────────────────────────────
kmf = KaplanMeierFitter()
km_lines = []
km_labels = []

# Ordered for visual clarity: longest-surviving first
PLOT_ORDER = [4, 0, 1, 3]   # Early ext., Crop-dom., Pastoral, High-dens.

for c_id in PLOT_ORDER:
    sub = surv[surv.cluster == c_id]
    label = LABELS[c_id]
    color = COLORS[c_id]
    n = len(sub)
    n_events = sub["event"].sum()

    kmf.fit(sub["duration"], sub["event"], label=label)
    sf = kmf.survival_function_
    timeline = sf.index.values
    surv_prob = sf[label].values

    # Plot the step curve
    line, = ax_B.step(
        timeline, surv_prob,
        where="post",
        color=color,
        linewidth=2.0,
        label=f"{label} (n={n}, events={n_events})",
        zorder=4,
    )
    km_lines.append(line)
    km_labels.append(f"{label} (n={n})")

    # Add median survival line if median is defined (survival crosses 0.5)
    if surv_prob.min() <= 0.5:
        # Find x where survival first drops to/below 0.5
        idx_med = np.searchsorted(-surv_prob, -0.5)  # first index where surv <= 0.5
        if idx_med >= len(timeline):
            idx_med = len(timeline) - 1
        # More precisely: find last time surv > 0.5, then interpolate step
        above_half = timeline[surv_prob > 0.5]
        if len(above_half) > 0:
            t_med = above_half[-1]
        else:
            t_med = timeline[0]
        # The median in KM is the first time survival <= 0.5
        median_val = kmf.median_survival_time_

        # Draw dashed median marker: horizontal from y=0.5 to curve, then vertical down
        ax_B.hlines(
            y=0.5, xmin=0, xmax=median_val,
            colors=color, linestyles="dashed", linewidth=1.0, alpha=0.7, zorder=3
        )
        ax_B.vlines(
            x=median_val, ymin=0, ymax=0.5,
            colors=color, linestyles="dashed", linewidth=1.0, alpha=0.7, zorder=3
        )
        # Small dot at (median, 0.5)
        ax_B.scatter(
            [median_val], [0.5],
            color=color, s=40, zorder=5, clip_on=False
        )

ax_B.axhline(0.5, color="#999999", linewidth=0.7, linestyle=":", zorder=1)
ax_B.set_xlabel("Years from 0 CE", labelpad=6)
ax_B.set_ylabel("Probability of remaining\nin Malthusian regime", labelpad=6)
ax_B.set_title("(B)  Kaplan-Meier Survival by Pathway", loc="left",
               fontsize=11, fontweight="bold", pad=8)
ax_B.set_xlim(0, 2050)
ax_B.set_ylim(-0.03, 1.05)

# X axis ticks at century marks
ax_B.set_xticks(range(0, 2001, 200))
ax_B.set_xticklabels([str(t) for t in range(0, 2001, 200)], fontsize=9)

# 0.5 label on y axis
yticks = [0.0, 0.25, 0.5, 0.75, 1.0]
ax_B.set_yticks(yticks)
ax_B.set_yticklabels([f"{t:.2f}" for t in yticks], fontsize=9)

# Add a note for censored observations
ax_B.text(
    2020, 0.02,
    "Countries without detected break\ncensored at 2000 CE",
    fontsize=7.5, color="#666666", ha="right", va="bottom", style="italic"
)

# Legend
surv_legend = [
    Line2D([0], [0], color=COLORS[c], linewidth=2,
           label=f"{LABELS[c]}  (n={len(surv[surv.cluster==c])}, "
                 f"events={surv[surv.cluster==c]['event'].sum()})")
    for c in PLOT_ORDER
]
# Add median marker explanation
surv_legend.append(
    Line2D([0], [0], color="#888888", linewidth=1.0, linestyle="dashed",
           label="Median survival")
)
ax_B.legend(
    handles=surv_legend,
    fontsize=8.5,
    frameon=False,
    loc="upper right",
    handlelength=1.8,
)

# ── figure title & save ───────────────────────────────────────────────────────
fig.suptitle(
    "Figure 4  |  Timing of Malthusian Structural Breaks and Survival by Pathway",
    fontsize=11,
    fontweight="bold",
    y=1.01,
)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
