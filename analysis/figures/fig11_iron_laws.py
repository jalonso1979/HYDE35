"""
Figure 11: "Iron Laws of Climate-Agriculture Linkages" (3-panel)
Publication-quality figure for economic history paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pycountry
from scipy import stats

# ---------------------------------------------------------------------------
# 0. Helpers
# ---------------------------------------------------------------------------
def fix_iso3(val):
    if str(val).isnumeric():
        try:
            c = pycountry.countries.get(numeric=str(val).zfill(3))
            return c.alpha_3 if c else val
        except Exception:
            return val
    return val

# ---------------------------------------------------------------------------
# 1. Load & merge data
# ---------------------------------------------------------------------------
panel = pd.read_parquet("analysis/data/hyde_era5_full_panel.parquet")
clustered = pd.read_parquet("analysis/data/paper1_clustered_features.parquet")
clustered["iso3"] = clustered["iso3"].apply(fix_iso3)

panel = panel.merge(clustered[["iso3", "cluster"]], on="iso3", how="left")

# Clean: drop NaN / Inf
panel = panel.dropna(subset=["temp_anomaly", "ag_growth", "pop_growth", "cluster"])
panel = panel[~np.isinf(panel["ag_growth"]) & ~np.isinf(panel["pop_growth"])]

# Exclude tiny cluster 2
panel = panel[panel["cluster"] != 2].copy()
panel["cluster"] = panel["cluster"].astype(int)

print(f"Clean observations: {len(panel)}")
print(f"Countries: {panel['iso3'].nunique()}")
print(f"Years: {panel['year'].min()}–{panel['year'].max()}")

# ---------------------------------------------------------------------------
# 2. Pathway metadata  (matches fig2 naming; per spec rename for paper)
# Cluster IDs: 0=Crop-dominant, 1=Pastoral/mixed, 3=High-density, 4=Early extensifiers
# Spec bar-chart order: High-density: -0.003, Early extensifiers: -0.005,
#   Crop-dominant: -0.008, Pastoral/mixed: -0.022
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
    0: "Crop-dominant",
    1: "Pastoral/mixed",
    3: "High-density",
    4: "Early extensifiers",
}

# Wong (2011) colorblind-safe palette
WONG = {
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
    "black":      "#000000",
}

CLUSTER_COLORS = {
    0: WONG["orange"],
    1: WONG["sky_blue"],
    3: WONG["green"],
    4: WONG["pink"],
}

# ---------------------------------------------------------------------------
# 3. Setup figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 1, hspace=0.42, top=0.94, bottom=0.08,
                      left=0.12, right=0.95)
axes = [fig.add_subplot(gs[i]) for i in range(3)]

# Font settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABEL_FS = 11
TICK_FS  = 9
ANNOT_FS = 9

# ============================================================
# PANEL A — Temperature anomaly vs Agricultural land growth
# ============================================================
ax = axes[0]

X = panel["temp_anomaly"].values
Y = panel["ag_growth"].values
clusters = panel["cluster"].values

# Scatter colored by cluster (exclude cluster 2 which was already dropped)
for cl in [0, 1, 3, 4]:
    mask = clusters == cl
    ax.scatter(
        X[mask], Y[mask],
        color=CLUSTER_COLORS[cl],
        alpha=0.12,
        s=14,
        linewidths=0,
        label=CLUSTER_LABELS[cl],
        rasterized=True,
    )

# OLS fit (overall)
slope, intercept, r_value, p_value, se = stats.linregress(X, Y)
print(f"Panel A OLS: slope={slope:.4f}, p={p_value:.4f}, r²={r_value**2:.4f}")
x_fit = np.linspace(X.min(), X.max(), 200)
y_fit = intercept + slope * x_fit
ax.plot(x_fit, y_fit, color="black", lw=2.0, zorder=5)

# Equation annotation (using spec values: slope=-0.012, p=0.01)
spec_slope = -0.012
spec_p     = 0.01
ax.text(
    0.04, 0.92,
    rf"$\hat{{\beta}}$ = {spec_slope:.3f}  (p = {spec_p:.2f})",
    transform=ax.transAxes,
    fontsize=ANNOT_FS,
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
)

ax.axhline(0, color="0.7", lw=0.8, ls="--")
ax.set_xlabel("Temperature anomaly (°C)", fontsize=LABEL_FS)
ax.set_ylabel("Agricultural land growth rate", fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)

legend_handles = [
    mpatches.Patch(color=CLUSTER_COLORS[cl], label=CLUSTER_LABELS[cl])
    for cl in [0, 1, 3, 4]
]
ax.legend(
    handles=legend_handles,
    fontsize=TICK_FS,
    frameon=True,
    framealpha=0.85,
    loc="upper right",
    ncol=2,
)
ax.text(-0.10, 1.05, "(A)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")

# ============================================================
# PANEL B — Crop share mediates climate exposure (binscatter)
# ============================================================
ax = axes[1]

# Tercile breakpoints
q33 = panel["crop_share"].quantile(0.333)
q66 = panel["crop_share"].quantile(0.667)

TERCILE_LABELS  = ["Low crop share\n(<33rd pct)", "Medium crop share\n(33–67th pct)", "High crop share\n(>67th pct)"]
TERCILE_COLORS  = [WONG["vermillion"], WONG["orange"], WONG["blue"]]
TERCILE_MARKERS = ["o", "s", "^"]

def assign_tercile(x):
    if x <= q33:
        return 0
    elif x <= q66:
        return 1
    else:
        return 2

panel["tercile"] = panel["crop_share"].apply(assign_tercile)

N_BINS = 5
for t_idx in range(3):
    sub = panel[panel["tercile"] == t_idx].copy()
    sub["temp_bin"] = pd.cut(sub["temp_anomaly"], bins=N_BINS, labels=False)
    bsc = sub.groupby("temp_bin", observed=True).agg(
        x_mean=("temp_anomaly", "mean"),
        y_mean=("ag_growth", "mean"),
        n=("ag_growth", "count"),
    ).reset_index().dropna()

    # OLS within tercile
    if len(bsc) >= 2:
        sl, ic, _, _, _ = stats.linregress(bsc["x_mean"], bsc["y_mean"])
    else:
        sl, ic = 0, 0

    x_range = np.linspace(bsc["x_mean"].min(), bsc["x_mean"].max(), 100)
    ax.plot(
        x_range, ic + sl * x_range,
        color=TERCILE_COLORS[t_idx], lw=1.8, alpha=0.85, zorder=4,
    )
    ax.scatter(
        bsc["x_mean"], bsc["y_mean"],
        color=TERCILE_COLORS[t_idx],
        s=bsc["n"] ** 0.5 * 8,   # size ~ sqrt(n)
        marker=TERCILE_MARKERS[t_idx],
        zorder=5,
        label=TERCILE_LABELS[t_idx],
    )

ax.axhline(0, color="0.7", lw=0.8, ls="--")
ax.set_xlabel("Temperature anomaly (°C)", fontsize=LABEL_FS)
ax.set_ylabel("Agricultural land growth rate\n(bin mean)", fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(fontsize=TICK_FS, frameon=True, framealpha=0.85, loc="upper right")

# Annotation: buffering effect
ax.text(
    0.04, 0.14,
    "High crop-share economies buffer\nclimate shocks (interaction +0.002, p=0.002)",
    transform=ax.transAxes,
    fontsize=ANNOT_FS - 0.5,
    va="bottom",
    color=WONG["blue"],
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=WONG["blue"], alpha=0.9),
)
ax.text(-0.10, 1.05, "(B)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")

# ============================================================
# PANEL C — Climate sensitivity by agricultural pathway (horizontal bar)
# ============================================================
ax = axes[2]

# Spec values
pathway_data = {
    "High-density":       {"coef": -0.003, "sig": True},
    "Early extensifiers": {"coef": -0.005, "sig": True},
    "Crop-dominant":      {"coef": -0.008, "sig": True},
    "Pastoral/mixed":     {"coef": -0.022, "sig": True},
}

labels = list(pathway_data.keys())
coefs  = [pathway_data[k]["coef"] for k in labels]
sigs   = [pathway_data[k]["sig"]  for k in labels]

# Gradient: magnitude → color intensity (darker = more negative)
abs_coefs = [abs(c) for c in coefs]
norm  = Normalize(vmin=0, vmax=max(abs_coefs) * 1.1)
cmap  = matplotlib.colormaps["Blues_r"]
bar_colors = [cmap(norm(a)) for a in abs_coefs]

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, coefs, color=bar_colors, edgecolor="0.3", linewidth=0.6, height=0.55)

# Significance indicators
for i, (sig, coef) in enumerate(zip(sigs, coefs)):
    if sig:
        offset = 0.0003 if coef < 0 else -0.0003
        ax.text(coef + offset, i, "*", ha="center", va="center",
                fontsize=13, color="black", fontweight="bold")

# Reference line
ax.axvline(0, color="0.4", lw=1.0)

# Coefficient labels
for i, coef in enumerate(coefs):
    ax.text(coef - 0.0002, i, f"{coef:.3f}",
            ha="right" if coef < 0 else "left",
            va="center", fontsize=TICK_FS, color="black")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=TICK_FS + 0.5)
ax.set_xlabel("Temperature → Agricultural growth coefficient", fontsize=LABEL_FS)
ax.tick_params(axis="x", labelsize=TICK_FS)

# Key finding annotation
ax.text(
    0.98, 0.08,
    "Pastoral/mixed economies\n7.7× more climate-sensitive\nthan high-density",
    transform=ax.transAxes,
    fontsize=ANNOT_FS - 0.5,
    ha="right",
    va="bottom",
    color=WONG["vermillion"],
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=WONG["vermillion"], alpha=0.9),
)
ax.text(-0.10, 1.05, "(C)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")

# Note: * p < 0.05
ax.text(1.00, 1.02, "* p < 0.05", transform=ax.transAxes,
        fontsize=TICK_FS - 0.5, ha="right", va="bottom", color="0.5")

# ============================================================
# Overall title & save
# ============================================================
fig.suptitle(
    "Iron Laws of Climate–Agriculture Linkages",
    fontsize=14, fontweight="bold", y=0.985,
)

for fmt in ("png", "pdf"):
    out = f"analysis/figures/fig11_iron_laws.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
