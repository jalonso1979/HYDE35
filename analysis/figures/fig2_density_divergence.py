"""
Figure 2: The Great Divergence in Population Density by Pathway (0-1750 CE)
Publication-quality figure for economic history paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import pycountry

# ---------------------------------------------------------------------------
# 1. Helper: ISO numeric -> ISO alpha-3
# ---------------------------------------------------------------------------
def iso_num_to_alpha3(num):
    try:
        c = pycountry.countries.get(numeric=str(int(num)).zfill(3))
        return c.alpha_3 if c else str(int(num))
    except Exception:
        return str(int(num))

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
panel = pd.read_parquet("analysis/data/country_analysis_panel.parquet")
clust = pd.read_parquet("analysis/data/paper1_clustered_features.parquet")

# Convert panel country (ISO numeric) -> iso3
panel = panel.copy()
panel["iso3"] = panel["country"].apply(iso_num_to_alpha3)

# Handle numeric-string iso3 codes in cluster file (530, 736, 891)
# These are genuine ISO numeric codes stored as strings; convert them to alpha3
def fix_cluster_iso3(val):
    if val.isnumeric():
        try:
            c = pycountry.countries.get(numeric=val.zfill(3))
            return c.alpha_3 if c else val
        except Exception:
            return val
    return val

clust = clust.copy()
clust["iso3"] = clust["iso3"].apply(fix_cluster_iso3)

# Merge
merged = panel.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")

# Filter: year >= 0, exclude cluster 2
merged = merged[(merged["year"] >= 0) & (merged["cluster"] != 2)]

print(f"Merged rows: {len(merged)}")
print(f"Years: {sorted(merged['year'].unique())}")
print(f"Clusters represented: {sorted(merged['cluster'].unique())}")
print(f"Countries per cluster:\n{merged.groupby('cluster')['iso3'].nunique()}")

# ---------------------------------------------------------------------------
# 3. Pathway metadata
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    3: "High-density intensive",
    4: "Early extensifiers",
}

# Colorblind-friendly palette (Wong 2011 / IBM accessible)
# Avoid red-green conflicts; use blue/orange/purple/teal
CLUSTER_COLORS = {
    0: "#E69F00",   # orange
    1: "#56B4E9",   # sky blue
    3: "#CC79A7",   # pink/mauve
    4: "#009E73",   # bluish green
}

CLUSTER_LINESTYLES = {
    0: "solid",
    1: "dashed",
    3: "dotted",
    4: (0, (3, 1, 1, 1)),   # dash-dot
}

CLUSTER_LINEWIDTHS = {
    0: 1.8,
    1: 1.8,
    3: 2.0,
    4: 1.8,
}

# ---------------------------------------------------------------------------
# 4. Compute pathway-year aggregates
# ---------------------------------------------------------------------------
def pathway_stats(df, value_col):
    """Mean and SE per (cluster, year)."""
    grp = df.groupby(["cluster", "year"])[value_col]
    stats = grp.agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["lo"] = stats["mean"] - stats["se"]
    stats["hi"] = stats["mean"] + stats["se"]
    return stats

density_stats = pathway_stats(merged, "popdens_p_km2_mean")
years = sorted(density_stats["year"].unique())

# ---------------------------------------------------------------------------
# 5. Compute density ratio: cluster-3 / cluster-1
# ---------------------------------------------------------------------------
def ratio_with_ci(stats, num_cluster, den_cluster):
    num = stats[stats["cluster"] == num_cluster].set_index("year")
    den = stats[stats["cluster"] == den_cluster].set_index("year")
    common = num.index.intersection(den.index)
    num = num.loc[common]
    den = den.loc[common]

    ratio = num["mean"] / den["mean"]

    # Delta-method SE for ratio: SE(r) ≈ r * sqrt((SE_n/mu_n)^2 + (SE_d/mu_d)^2)
    rel_se = np.sqrt((num["se"] / num["mean"])**2 + (den["se"] / den["mean"])**2)
    se_ratio = ratio * rel_se

    return (
        common.values,
        ratio.values,
        (ratio - se_ratio).values,
        (ratio + se_ratio).values,
    )

ratio_years, ratio_vals, ratio_lo, ratio_hi = ratio_with_ci(density_stats, 3, 1)

# ---------------------------------------------------------------------------
# 6. Figure layout
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        False,
    "figure.dpi":       300,
    "savefig.dpi":      300,
})

fig, (ax_a, ax_b) = plt.subplots(
    2, 1,
    figsize=(10, 10),
    gridspec_kw={"hspace": 0.42},
)

# ---------------------------------------------------------------------------
# 7. Panel A — Population density over time (log scale)
# ---------------------------------------------------------------------------
ax = ax_a

for cl in [0, 1, 3, 4]:
    sub = density_stats[density_stats["cluster"] == cl].sort_values("year")
    yr  = sub["year"].values
    mn  = sub["mean"].values
    lo  = sub["lo"].values.clip(min=1e-3)   # avoid log(<=0)
    hi  = sub["hi"].values

    ax.plot(
        yr, mn,
        label=CLUSTER_LABELS[cl],
        color=CLUSTER_COLORS[cl],
        linestyle=CLUSTER_LINESTYLES[cl],
        linewidth=CLUSTER_LINEWIDTHS[cl],
        zorder=3,
    )
    ax.fill_between(
        yr, lo, hi,
        color=CLUSTER_COLORS[cl],
        alpha=0.15,
        zorder=2,
    )

ax.set_yscale("log")
ax.set_xlim(0, 1750)
ax.set_ylim(bottom=0.05)

# Y-axis: nice log ticks
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"{v:g}")
)

ax.set_xlabel("Year (CE)", fontsize=10)
ax.set_ylabel("Population density (persons / km²)", fontsize=10)
ax.set_title(
    "(A)  Population Density by Agricultural Pathway, 0–1750 CE",
    fontweight="bold", loc="left", pad=8,
)

# --- Historical event annotations ---
ann_kw = dict(
    xycoords="data",
    textcoords="offset points",
    fontsize=8,
    color="#444444",
    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
)

# Fall of Rome ~476 CE
ax.annotate(
    "Fall of Rome\n~476 CE",
    xy=(476, ax.get_ylim()[0] * 1.5),
    xytext=(30, 18),
    ha="left",
    **ann_kw,
)
ax.axvline(476, color="#888888", lw=0.7, ls="--", zorder=1, alpha=0.6)

# Black Death ~1350 CE
ax.annotate(
    "Black Death\n~1350 CE",
    xy=(1350, ax.get_ylim()[0] * 1.5),
    xytext=(-80, 18),
    ha="left",
    **ann_kw,
)
ax.axvline(1350, color="#888888", lw=0.7, ls="--", zorder=1, alpha=0.6)

# Legend — place in upper left (inside, low clutter)
legend_handles = [
    Line2D(
        [0], [0],
        color=CLUSTER_COLORS[cl],
        linestyle=CLUSTER_LINESTYLES[cl],
        linewidth=CLUSTER_LINEWIDTHS[cl],
        label=CLUSTER_LABELS[cl],
    )
    for cl in [0, 1, 3, 4]
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
    handlelength=2.5,
    borderpad=0.4,
    labelspacing=0.4,
)

# ---------------------------------------------------------------------------
# 8. Panel B — Density ratio (High-density / Pastoral)
# ---------------------------------------------------------------------------
ax = ax_b

ax.plot(
    ratio_years, ratio_vals,
    color=CLUSTER_COLORS[3],
    linewidth=2.0,
    linestyle="solid",
    zorder=3,
    label="High-density / Pastoral ratio",
)
ax.fill_between(
    ratio_years, ratio_lo, ratio_hi,
    color=CLUSTER_COLORS[3],
    alpha=0.20,
    zorder=2,
    label="±1 SE",
)

# Reference line at ratio = 1
ax.axhline(1.0, color="#555555", lw=1.0, ls="--", zorder=1, alpha=0.8,
           label="Ratio = 1 (parity)")

ax.set_xlim(0, 1750)
ax.set_ylim(bottom=0)

ax.set_xlabel("Year (CE)", fontsize=10)
ax.set_ylabel("Density ratio\n(High-density intensive / Pastoral–mixed)", fontsize=10)
ax.set_title(
    "(B)  Divergence Ratio: High-density Intensive vs. Pastoral/Mixed",
    fontweight="bold", loc="left", pad=8,
)

# Annotate widening gap near 1500-1750
peak_idx  = np.argmax(ratio_vals)
peak_year = ratio_years[peak_idx]
peak_val  = ratio_vals[peak_idx]

ax.annotate(
    f"Widening gap\n(ratio ≈ {peak_val:.1f}×)",
    xy=(peak_year, peak_val),
    xytext=(-90, -30),
    textcoords="offset points",
    fontsize=8,
    color="#333333",
    arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9),
    ha="left",
)

ax.legend(
    loc="upper left",
    frameon=False,
    handlelength=2.5,
    borderpad=0.4,
    labelspacing=0.4,
)

# ---------------------------------------------------------------------------
# 9. Shared x-tick formatting (every 250 years)
# ---------------------------------------------------------------------------
for ax_ in (ax_a, ax_b):
    ax_.set_xticks(range(0, 1751, 250))
    ax_.tick_params(axis="both", which="both", direction="out", length=4, width=0.7)

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
out_png = "analysis/figures/fig2_density_divergence.png"
out_pdf = "analysis/figures/fig2_density_divergence.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close(fig)
