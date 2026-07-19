"""
Figure 12: "The Malthusian Climate Channel" (2-panel)
Publication-quality figure for economic history paper.

The 1950-1967 draft claimed a "warming paradox" (temp -> population growth
positive despite temp -> agriculture negative). On the full 1950-2025 panel
the population channel REVERSES: warming reduces both agricultural land and
population growth; the positive response is confined to 1950-1969.

Loads the same panel as analysis/run_ag_impact_final.py and computes every
annotated estimate in-script with the same FE estimator (within-country
demeaning, SEs clustered by country), so the figure cannot drift from the
paper's quoted numbers.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# 0. FE helper (mirrors run_ag_impact_final.py)
# ---------------------------------------------------------------------------
def run_fe(panel, dep, indeps, entity="country_id"):
    cols = [dep] + indeps + [entity]
    sub = panel[cols].dropna()
    sub = sub[~np.isinf(sub[dep])]
    if len(sub) < 50:
        return None
    for c in [dep] + indeps:
        sub[c] = sub[c] - sub.groupby(entity)[c].transform("mean")
    y = sub[dep]
    X = sm.add_constant(sub[indeps])
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub[entity]})


def p_fmt(p):
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}" if p < 0.01 else f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# 1. Load data & estimate (same sample rules as run_ag_impact_final.py)
# ---------------------------------------------------------------------------
panel = pd.read_parquet("analysis/data/hyde_era5_extended_panel.parquet")
panel = panel[panel["temperature_c"].notna() & (panel["pop"] > 0)
              & (panel["area_km2"] >= 500)].copy()
panel["ag_land_growth"] = panel.groupby("country_id")["log_ag_land"].transform(
    lambda s: s.diff())

print(f"Panel: {len(panel):,} obs, {panel['iso3'].nunique()} countries, "
      f"{panel['year'].min()}–{panel['year'].max()}")

CLIM = ["temp_anomaly", "precip_anomaly"]

m_ag = run_fe(panel, "ag_land_growth", CLIM)
b_ag, p_ag = m_ag.params["temp_anomaly"], m_ag.pvalues["temp_anomaly"]

m_pop = run_fe(panel, "pop_growth", CLIM)
b_pop, p_pop = m_pop.params["temp_anomaly"], m_pop.pvalues["temp_anomaly"]

m_early = run_fe(panel[panel["year"] <= 1969], "pop_growth", CLIM)
b_early, p_early = m_early.params["temp_anomaly"], m_early.pvalues["temp_anomaly"]

m_late = run_fe(panel[panel["year"] >= 1970], "pop_growth", CLIM)
b_late, p_late = m_late.params["temp_anomaly"], m_late.pvalues["temp_anomaly"]

print(f"FE temp → ag land growth : β={b_ag:+.6f}, p={p_ag:.6f}, N={int(m_ag.nobs)}")
print(f"FE temp → pop growth     : β={b_pop:+.6f}, p={p_pop:.6f}, N={int(m_pop.nobs)}")
print(f"  1950–1969 subsample    : β={b_early:+.6f}, p={p_early:.6f}, N={int(m_early.nobs)}")
print(f"  1970–2025 subsample    : β={b_late:+.6f}, p={p_late:.6f}, N={int(m_late.nobs)}")

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

# ---------------------------------------------------------------------------
# 2. Setup figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(1, 2, wspace=0.36, top=0.88, bottom=0.10,
                      left=0.09, right=0.97)
ax_scatter = fig.add_subplot(gs[0, 0])
ax_schema  = fig.add_subplot(gs[0, 1])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABEL_FS = 11
TICK_FS  = 9
ANNOT_FS = 9

# ============================================================
# PANEL A — Temperature anomaly vs Population growth (scatter + regime fits)
# ============================================================
ax = ax_scatter

sc = panel.dropna(subset=["temp_anomaly", "pop_growth"])
sc = sc[~np.isinf(sc["pop_growth"])]
X, Y = sc["temp_anomaly"].values, sc["pop_growth"].values

ax.scatter(
    X, Y,
    color=WONG["sky_blue"],
    alpha=0.10,
    s=14,
    linewidths=0,
    rasterized=True,
    label="Country-year obs.",
)

# Within-country FE fits drawn through subsample means
x_fit = np.linspace(X.min(), X.max(), 200)
ax.plot(x_fit, Y.mean() + b_pop * (x_fit - X.mean()),
        color=WONG["blue"], lw=2.4, zorder=6, label="FE fit, 1950–2025")

early = sc[sc["year"] <= 1969]
ax.plot(x_fit, early["pop_growth"].mean() + b_early * (x_fit - early["temp_anomaly"].mean()),
        color=WONG["vermillion"], lw=1.8, ls="--", zorder=5,
        label="FE fit, 1950–1969")

late = sc[sc["year"] >= 1970]
ax.plot(x_fit, late["pop_growth"].mean() + b_late * (x_fit - late["temp_anomaly"].mean()),
        color=WONG["green"], lw=1.8, ls=":", zorder=5,
        label="FE fit, 1970–2025")

ax.text(
    0.04, 0.97,
    rf"FE $\hat{{\beta}}$ = {b_pop:+.4f}  ({p_fmt(p_pop)})",
    transform=ax.transAxes,
    fontsize=ANNOT_FS,
    va="top",
    color=WONG["blue"],
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=WONG["blue"], alpha=0.9),
)

ax.text(
    0.97, 0.03,
    "The positive 1950–69 response\nreverses after 1970:\n"
    rf"$\beta$ = {b_early:+.4f} → {b_late:+.4f}",
    transform=ax.transAxes,
    fontsize=ANNOT_FS - 0.5,
    ha="right",
    va="bottom",
    color=WONG["vermillion"],
    bbox=dict(boxstyle="round,pad=0.35", fc="#fff8f0", ec=WONG["vermillion"],
              alpha=0.92, lw=1.2),
)

ax.axhline(0, color="0.7", lw=0.8, ls="--")
ax.set_xlabel("Temperature anomaly (°C)", fontsize=LABEL_FS)
ax.set_ylabel("Population growth rate", fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
ax.legend(fontsize=TICK_FS - 0.5, frameon=True, framealpha=0.85, loc="upper right")
ax.text(-0.12, 1.06, "(A)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")

# ============================================================
# PANEL B — Schematic: both channels now negative
# ============================================================
ax = ax_schema
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(-0.08, 1.06, "(B)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top")

def draw_channel(ax, x, y_top, y_bot, label_top, label_bot,
                 coef_text, p_text, color):
    """Draw a Temperature box → outcome box with arrow."""
    box_w, box_h = 2.4, 0.9

    rect_top = mpatches.FancyBboxPatch(
        (x - box_w / 2, y_top - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.08",
        facecolor="#f5e6cc" if color == "red" else "#ddeeff",
        edgecolor="#888888",
        linewidth=1.0,
    )
    ax.add_patch(rect_top)
    ax.text(x, y_top, label_top,
            ha="center", va="center", fontsize=10, fontweight="bold")

    arrow_color = "#CC3311" if color == "red" else "#0044AA"
    ax.annotate(
        "",
        xy=(x, y_bot + box_h / 2 + 0.15),
        xytext=(x, y_top - box_h / 2 - 0.15),
        arrowprops=dict(
            arrowstyle="-|>",
            color=arrow_color,
            lw=2.5,
            mutation_scale=18,
        ),
    )

    mid_y = (y_top + y_bot) / 2
    ax.text(x + 0.55, mid_y, coef_text,
            ha="left", va="center", fontsize=8.5,
            color=arrow_color, fontweight="bold")

    rect_bot = mpatches.FancyBboxPatch(
        (x - box_w / 2, y_bot - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.08",
        facecolor="#ffe5e5" if color == "red" else "#e5f0ff",
        edgecolor="#888888",
        linewidth=1.0,
    )
    ax.add_patch(rect_bot)
    ax.text(x, y_bot, label_bot,
            ha="center", va="center", fontsize=10, fontweight="bold")

    ax.text(x, y_bot - box_h / 2 - 0.35, p_text,
            ha="center", va="top", fontsize=8,
            color="0.4",
            style="italic")

# ---- Left channel: Temperature → Agriculture (negative) ----
draw_channel(
    ax,
    x=2.5,
    y_top=8.0, y_bot=5.5,
    label_top="Temperature ↑",
    label_bot="Agriculture ↓",
    coef_text=f"β = {b_ag:+.4f}",
    p_text=f"{p_fmt(p_ag)}",
    color="red",
)

# ---- Right channel: Temperature → Population (negative on full sample) ----
draw_channel(
    ax,
    x=7.5,
    y_top=8.0, y_bot=5.5,
    label_top="Temperature ↑",
    label_bot="Population ↓",
    coef_text=f"β = {b_pop:+.4f}",
    p_text=f"{p_fmt(p_pop)}",
    color="blue",
)

# ---- Regime annotation under the population channel ----
ax.text(7.5, 4.15,
        f"1950–69: β = {b_early:+.4f}\n1970–2025: β = {b_late:+.4f}",
        ha="center", va="top", fontsize=8.5,
        color="#0044AA", fontweight="bold")

# ---- Divider line ----
ax.axvline(5.0, ymin=0.40, ymax=0.95,
           color="0.75", lw=1.2, ls="--")

# ---- Bottom caption ----
caption = (
    "No Malthusian climate paradox on the full sample:\n"
    "warming reduces both agricultural land and\n"
    "population growth; the positive population\n"
    "response is confined to 1950–1969"
)
ax.text(5.0, 3.1, caption,
        ha="center", va="top", fontsize=9.5,
        color="#333333",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9",
                  ec="0.7", alpha=0.95, lw=1.0))

# ---- Panel labels ----
ax.text(2.5, 9.4, "Channel I", ha="center", va="top",
        fontsize=10, fontweight="bold", color="#CC3311")
ax.text(7.5, 9.4, "Channel II", ha="center", va="top",
        fontsize=10, fontweight="bold", color="#0044AA")

# ============================================================
# Overall title & save
# ============================================================
fig.suptitle(
    "The Malthusian Climate Channel",
    fontsize=14, fontweight="bold", y=0.96,
)

for fmt in ("png", "pdf"):
    out = f"analysis/figures/fig12_climate_channel.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
