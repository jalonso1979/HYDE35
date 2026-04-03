"""
Figure 5: Agricultural Structure Evolution by Pathway (0–1750 CE)
Two-panel time-series figure for economic history paper using HYDE 3.5 data.

Panel (A): Crop share of agricultural land by pathway, 0–1750 CE
Panel (B): Urban population share by pathway, 0–1750 CE
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pycountry
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
from pathlib import Path
ANALYSIS_DATA = Path("/Volumes/BIGDATA/HYDE35/analysis/data")
OUT_PNG = "/Volumes/BIGDATA/HYDE35/analysis/figures/fig5_ag_structure_evolution.png"
OUT_PDF = "/Volumes/BIGDATA/HYDE35/analysis/figures/fig5_ag_structure_evolution.pdf"

# ── cluster metadata ─────────────────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}

# Same colorblind-friendly palette as other figures (Wong 2011)
COLORS = {
    0: "#E69F00",   # orange
    1: "#56B4E9",   # sky blue
    2: "#CC79A7",   # pink/magenta (singleton — excluded from this figure)
    3: "#009E73",   # green
    4: "#0072B2",   # blue
}

# ── load and merge data ───────────────────────────────────────────────────────
cp = pd.read_parquet(ANALYSIS_DATA / "country_analysis_panel.parquet")
clustered = pd.read_parquet(ANALYSIS_DATA / "paper1_clustered_features.parquet")

def iso_num_to_alpha3(num):
    try:
        c = pycountry.countries.get(numeric=str(int(num)).zfill(3))
        return c.alpha_3 if c else str(int(num))
    except:
        return str(int(num))

code_map = {c: iso_num_to_alpha3(c) for c in cp.country.unique()}
cp["iso3"] = cp["country"].map(code_map)
cp_merged = cp.merge(clustered[["iso3", "cluster"]], on="iso3", how="inner")

labels = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}
cp_merged["pathway"] = cp_merged["cluster"].map(labels)

# Filter: 0–1750 CE, exclude Egypt cluster (cluster 2)
cp_post0 = cp_merged[(cp_merged.year >= 0) & (cp_merged.cluster != 2)].copy()

# Compute crop share of total agricultural land
crop = (
    cp_post0.get("nonrice_mha_mean", pd.Series(0, index=cp_post0.index))
    + cp_post0.get("rice_mha_mean", pd.Series(0, index=cp_post0.index))
)
graz = cp_post0.get("grazing_mha_mean", pd.Series(0, index=cp_post0.index))
total = crop + graz
cp_post0["crop_share"] = np.where(total > 0, crop / total, np.nan)

# ── aggregate: mean and SEM by year and pathway ───────────────────────────────
def agg_group(df, var):
    grp = df.groupby(["year", "pathway"])[var]
    mean = grp.mean().reset_index().rename(columns={var: "mean"})
    sem  = grp.sem().reset_index().rename(columns={var: "sem"})
    out  = mean.merge(sem, on=["year", "pathway"])
    out["cluster"] = out["pathway"].map({v: k for k, v in labels.items()})
    return out

crop_agg  = agg_group(cp_post0, "crop_share")
urban_agg = agg_group(cp_post0, "urban_share_mean")

# Pathway ordering for consistent legend (by cluster id, excluding 2)
pathway_order = [labels[c] for c in sorted(COLORS) if c != 2]

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

fig, (ax_A, ax_B) = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.38)

# helper: bold panel label at top-left
def panel_label(ax, letter):
    ax.text(-0.07, 1.05, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# Panel (A): Crop share of agricultural land
# ─────────────────────────────────────────────────────────────────────────────
for pathway in pathway_order:
    sub = crop_agg[crop_agg["pathway"] == pathway].sort_values("year")
    if sub.empty:
        continue
    c_id = sub["cluster"].iloc[0]
    color = COLORS[c_id]
    ax_A.plot(sub["year"], sub["mean"], color=color, linewidth=1.8,
              label=pathway, zorder=3)
    ax_A.fill_between(
        sub["year"],
        sub["mean"] - sub["sem"],
        sub["mean"] + sub["sem"],
        color=color, alpha=0.15, linewidth=0, zorder=2,
    )

# Reference line at 0.5 (equal crop/grazing split)
ax_A.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--",
             zorder=1, label="_nolegend_")
ax_A.text(1755, 0.5, "equal split", va="center", ha="left",
          fontsize=7.5, color="#888888")

ax_A.set_xlim(0, 1750)
ax_A.set_ylim(0, 1.02)
ax_A.set_xlabel("Year (CE)")
ax_A.set_ylabel("Cropland share of total agricultural land")
ax_A.set_title("Agricultural land composition by pathway, 0–1750 CE", pad=6)
panel_label(ax_A, "A")

# Legend in upper-left whitespace
legend_A = ax_A.legend(
    title="Pathway",
    title_fontsize=8,
    fontsize=8,
    frameon=True,
    framealpha=0.85,
    edgecolor="#cccccc",
    loc="upper left",
    bbox_to_anchor=(0.01, 0.98),
    handlelength=1.8,
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (B): Urban population share
# ─────────────────────────────────────────────────────────────────────────────
for pathway in pathway_order:
    sub = urban_agg[urban_agg["pathway"] == pathway].sort_values("year")
    if sub.empty:
        continue
    c_id = sub["cluster"].iloc[0]
    color = COLORS[c_id]
    ax_B.plot(sub["year"], sub["mean"], color=color, linewidth=1.8,
              label=pathway, zorder=3)
    ax_B.fill_between(
        sub["year"],
        np.maximum(sub["mean"] - sub["sem"], 0),
        sub["mean"] + sub["sem"],
        color=color, alpha=0.15, linewidth=0, zorder=2,
    )

ax_B.set_xlim(0, 1750)
ax_B.set_ylim(bottom=0)
ax_B.set_xlabel("Year (CE)")
ax_B.set_ylabel("Urban population share")
ax_B.set_title("Urban population share by pathway, 0–1750 CE", pad=6)
panel_label(ax_B, "B")

# Annotation noting the divergence
ymax = urban_agg["mean"].max()
ax_B.annotate(
    "Dramatic divergence\nin urbanization post-1000 CE",
    xy=(1200, ymax * 0.85),
    xytext=(700, ymax * 0.70),
    fontsize=7.5,
    color="#555555",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6),
)

# Legend in upper-left whitespace
ax_B.legend(
    title="Pathway",
    title_fontsize=8,
    fontsize=8,
    frameon=True,
    framealpha=0.85,
    edgecolor="#cccccc",
    loc="upper left",
    bbox_to_anchor=(0.01, 0.98),
    handlelength=1.8,
)

# ── figure title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 5  |  Agricultural Structure Evolution by Pathway (0–1750 CE)",
    fontsize=11,
    fontweight="bold",
    y=1.01,
)

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
