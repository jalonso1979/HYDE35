"""
Figure 1: Agricultural Transition Pathways — Five-Cluster Typology
2x2 panel figure for economic history paper using HYDE 3.5 data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, LogFormatter
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "/Volumes/BIGDATA/HYDE35/analysis/data/paper1_clustered_features.parquet"
OUT_PNG   = "/Volumes/BIGDATA/HYDE35/analysis/figures/fig1_pathway_typology.png"
OUT_PDF   = "/Volumes/BIGDATA/HYDE35/analysis/figures/fig1_pathway_typology.pdf"

# ── cluster metadata ─────────────────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Crop-dominant\nlate developers",
    1: "Pastoral/mixed\nlate developers",
    2: "Irrigation pioneer\n(Egypt)",
    3: "High-density\nintensive",
    4: "Early extensifiers",
}
CLUSTER_N = {0: 46, 1: 65, 2: 1, 3: 16, 4: 29}

# Colorblind-friendly palette (Wong 2011 / tab10 subset)
# Using a carefully chosen 5-color set distinguishable under common CVD
COLORS = {
    0: "#E69F00",   # orange
    1: "#56B4E9",   # sky blue
    2: "#CC79A7",   # pink/magenta (singleton — stands out)
    3: "#009E73",   # green
    4: "#0072B2",   # blue
}

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df["cluster"] = df["cluster"].astype(int)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        False,
    "figure.dpi":       150,   # preview; saved at 300
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.38, wspace=0.35)

ax_A, ax_B = axes[0, 0], axes[0, 1]
ax_C, ax_D = axes[1, 0], axes[1, 1]

# helper: bold panel label at top-left
def panel_label(ax, letter):
    ax.text(-0.08, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# (A) Scatter: peak_ag_expansion_year vs density_1750 (log y)
# ─────────────────────────────────────────────────────────────────────────────
for c_id in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c_id]
    ax_A.scatter(
        sub["peak_ag_expansion_year"],
        sub["density_1750"],
        c=COLORS[c_id],
        s=28 if c_id != 2 else 80,
        alpha=0.80,
        linewidths=0.4,
        edgecolors="white",
        zorder=3,
        label=CLUSTER_LABELS[c_id].replace("\n", " "),
    )

# cluster centroid labels
for c_id in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c_id]
    cx = sub["peak_ag_expansion_year"].median()
    cy = np.exp(np.log(sub["density_1750"]).median())
    short = {
        0: "C0", 1: "C1", 2: "C2\n(EGY)", 3: "C3", 4: "C4"
    }[c_id]
    ax_A.annotate(
        short,
        xy=(cx, cy),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        color=COLORS[c_id],
        fontweight="bold",
        clip_on=True,
    )

ax_A.set_yscale("log")
ax_A.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{x:g}"))
ax_A.set_xlabel("Peak agricultural expansion year")
ax_A.set_ylabel("Agricultural density, 1750\n(log scale, persons/km²)")
ax_A.set_title("Peak expansion year vs. 1750 density", pad=6)
panel_label(ax_A, "A")

# ─────────────────────────────────────────────────────────────────────────────
# (B) Horizontal bar: mean density_1750 by pathway, ordered by density
# ─────────────────────────────────────────────────────────────────────────────
summary = (
    df.groupby("cluster")["density_1750"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values("mean")
)

yticks = range(len(summary))
bar_colors = [COLORS[c] for c in summary["cluster"]]

bars = ax_B.barh(
    yticks,
    summary["mean"],
    xerr=summary["std"],
    color=bar_colors,
    edgecolor="white",
    linewidth=0.6,
    error_kw=dict(ecolor="#444444", capsize=3, linewidth=0.8),
    height=0.6,
    zorder=3,
)

yticklabels = [
    f"C{int(c)}  {CLUSTER_LABELS[int(c)].replace(chr(10), ' ')}"
    for c in summary["cluster"]
]
ax_B.set_yticks(yticks)
ax_B.set_yticklabels(yticklabels, fontsize=7.5)
ax_B.set_xlabel("Mean density, 1750 (persons/km²)")
ax_B.set_title("Agricultural density by pathway", pad=6)
ax_B.axvline(0, color="#888888", linewidth=0.5)
panel_label(ax_B, "B")

# ─────────────────────────────────────────────────────────────────────────────
# (C) Stacked bar: crop share vs grazing share by pathway
# ─────────────────────────────────────────────────────────────────────────────
land_summary = (
    df.groupby("cluster")["crop_share_1750"]
    .mean()
    .reset_index()
    .sort_values("crop_share_1750", ascending=False)
)
land_summary["grazing_share"] = 1.0 - land_summary["crop_share_1750"]

x_pos = range(len(land_summary))
crop_color   = "#4E9A5E"   # muted green — cropland
graze_color  = "#C5A244"   # muted amber — grazing / pasture

ax_C.bar(
    x_pos,
    land_summary["crop_share_1750"],
    color=crop_color,
    edgecolor="white",
    linewidth=0.5,
    label="Cropland",
    zorder=3,
)
ax_C.bar(
    x_pos,
    land_summary["grazing_share"],
    bottom=land_summary["crop_share_1750"],
    color=graze_color,
    edgecolor="white",
    linewidth=0.5,
    label="Grazing / pasture",
    zorder=3,
)

xlabels = [
    f"C{int(c)}" for c in land_summary["cluster"]
]
# add n= as second line
xlabels_full = [
    f"C{int(c)}\n(n={CLUSTER_N[int(c)]})"
    for c in land_summary["cluster"]
]
ax_C.set_xticks(x_pos)
ax_C.set_xticklabels(xlabels_full, fontsize=8)
ax_C.set_ylabel("Share of agricultural land, 1750")
ax_C.set_ylim(0, 1.08)
ax_C.set_title("Land-use composition by pathway", pad=6)
ax_C.legend(fontsize=7.5, frameon=False, loc="upper right")
panel_label(ax_C, "C")

# ─────────────────────────────────────────────────────────────────────────────
# (D) Scatter: pop_growth_0_1000 vs pop_growth_1000_1750, colored by cluster
# ─────────────────────────────────────────────────────────────────────────────
for c_id in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c_id]
    ax_D.scatter(
        sub["pop_growth_0_1000"],
        sub["pop_growth_1000_1750"],
        c=COLORS[c_id],
        s=28 if c_id != 2 else 80,
        alpha=0.80,
        linewidths=0.4,
        edgecolors="white",
        zorder=3,
        label=CLUSTER_LABELS[c_id].replace("\n", " "),
    )

# reference lines at (0, 0)
ax_D.axhline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
ax_D.axvline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=1)

ax_D.set_xlabel("Pop. growth rate, 0–1000 CE (ann.)")
ax_D.set_ylabel("Pop. growth rate, 1000–1750 CE (ann.)")
ax_D.set_title("Population growth across two periods", pad=6)
panel_label(ax_D, "D")

# ── shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(
        color=COLORS[c],
        label=f"C{c}: {CLUSTER_LABELS[c].replace(chr(10), ' ')} (n={CLUSTER_N[c]})",
    )
    for c in sorted(COLORS)
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
    columnspacing=1.2,
    handlelength=1.2,
)

# ── title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 1  |  Agricultural Transition Pathways — Five-Cluster Typology",
    fontsize=11,
    fontweight="bold",
    y=1.01,
)

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
