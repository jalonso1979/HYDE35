"""
Figure 6: Pathway-Specific Malthusian Dynamics

Panel A: Rolling-window density coefficients by agricultural pathway (4 pathways)
Panel B: Bar chart of net density coefficients from the interaction model
"""

import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Volumes/BIGDATA/HYDE35')

import numpy as np
import pandas as pd
import pycountry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from analysis.shared.config import ANALYSIS_DATA
from analysis.paper2_malthus.panels import build_malthusian_panel
from analysis.paper2_malthus.regressions import run_rolling_window

# ── Wong 2011 colorblind palette (same as Figures 1–5) ───────────────────────
COLORS = {
    0: '#E69F00',   # orange  — Crop-dominant late
    1: '#56B4E9',   # sky blue — Pastoral/mixed late
    3: '#009E73',   # green   — High-density intensive
    4: '#0072B2',   # blue    — Early extensifiers
}
LABELS = {
    0: 'Crop-dominant late',
    1: 'Pastoral/mixed late',
    3: 'High-density intensive',
    4: 'Early extensifiers',
}
PLOT_ORDER = [3, 4, 0, 1]   # visual order: high-density, early ext., crop-dom., pastoral

# ── Interaction model net density coefficients (provided) ────────────────────
NET_COEFS = {
    'Pastoral/mixed late':       0.00006140,
    'Crop-dominant late':        0.00015459,
    'High-density intensive':    0.00001974,
    'Early extensifiers':       -0.00000532,
}
# Significance: * p<0.05 (add star for those the paper reports as significant)
# Based on provided values — Early extensifiers is not significant (negative near 0)
SIG_FLAGS = {
    'Pastoral/mixed late':       True,
    'Crop-dominant late':        True,
    'High-density intensive':    True,
    'Early extensifiers':        False,
}

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'figure.dpi':        150,
})

# ── 1. Load and prepare data ──────────────────────────────────────────────────
print("Loading data …")
cp = pd.read_parquet(ANALYSIS_DATA / 'country_analysis_panel.parquet')
clustered = pd.read_parquet(ANALYSIS_DATA / 'paper1_clustered_features.parquet')


def iso_num_to_alpha3(num):
    try:
        c = pycountry.countries.get(numeric=str(int(num)).zfill(3))
        return c.alpha_3 if c else str(int(num))
    except Exception:
        return str(int(num))


code_map = {c: iso_num_to_alpha3(c) for c in cp.country.unique()}
cp['iso3'] = cp['country'].map(code_map)
cp = cp.merge(clustered[['iso3', 'cluster']], on='iso3', how='inner')

cp_f = cp[(cp.year >= 0) & (cp.cluster != 2)]
high_pop = cp_f.groupby('iso3')['popdens_p_km2_mean'].max()
valid = high_pop[high_pop > 1.0].index
cp_f = cp_f[cp_f.iso3.isin(valid)]

# ── 2. Rolling windows per pathway ───────────────────────────────────────────
print("Running rolling window regressions by pathway …")
pathway_rolling = {}
for cl in PLOT_ORDER:
    pw_data = cp_f[cp_f.cluster == cl]
    n_countries = pw_data.iso3.nunique()
    print(f"  Cluster {cl} ({LABELS[cl]}): {n_countries} countries")
    if n_countries < 5:
        print(f"    -> Skipped (< 5 countries)")
        continue
    malthus = build_malthusian_panel(pw_data, entity_col='iso3')
    rolling = run_rolling_window(
        malthus, dep_var='pop_growth_rate', key_var='popdens_lag',
        control_vars=['land_labor_ratio_lag'],
        entity_col='iso3', window_years=400, step_years=200,
    )
    if len(rolling) > 0:
        rolling['pathway'] = LABELS[cl]
        rolling['cluster'] = cl
        pathway_rolling[cl] = rolling
        print(f"    -> {len(rolling)} windows")

print(f"Pathways with results: {list(pathway_rolling.keys())}")

# ── 3. Build figure ───────────────────────────────────────────────────────────
fig, (ax_A, ax_B) = plt.subplots(
    2, 1, figsize=(10, 10),
    gridspec_kw={'hspace': 0.45}
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Rolling coefficients by pathway
# ─────────────────────────────────────────────────────────────────────────────
all_x, all_y = [], []
for cl in PLOT_ORDER:
    if cl not in pathway_rolling:
        continue
    df = pathway_rolling[cl]
    all_x.extend(df['center_year'].tolist())
    all_y.extend(df['coefficient'].tolist())

if all_y:
    y_min_global = min(min(all_y) * 1.3, -1e-9)
    y_max_global = max(max(all_y) * 1.3, 1e-9)
else:
    y_min_global, y_max_global = -1e-4, 1e-4

x_min_global = min(all_x) - 50 if all_x else 0
x_max_global = max(all_x) + 50 if all_x else 1800

# Malthusian zone shading (below y=0)
ax_A.fill_between(
    [x_min_global, x_max_global], y_min_global, 0,
    color='#ffcccc', alpha=0.30, zorder=0, linewidth=0
)

# Horizontal zero reference
ax_A.axhline(0, color='#888888', linewidth=0.8, linestyle='--', zorder=1)

legend_handles = []
for cl in PLOT_ORDER:
    if cl not in pathway_rolling:
        continue
    df = pathway_rolling[cl].sort_values('center_year')
    x = df['center_year'].values
    y = df['coefficient'].values
    p = df['pvalue'].values
    color = COLORS[cl]
    label = LABELS[cl]

    # Line
    ax_A.plot(x, y, color=color, linewidth=1.8, alpha=0.85, zorder=2)

    # Significant points (filled circles)
    sig = p < 0.05
    if sig.any():
        ax_A.scatter(x[sig], y[sig], color=color, s=55, zorder=4,
                     edgecolors='white', linewidths=0.5)
    # Non-significant (open circles)
    if (~sig).any():
        ax_A.scatter(x[~sig], y[~sig], facecolors='none', edgecolors=color,
                     s=50, zorder=3, linewidths=1.2)

    legend_handles.append(
        mpatches.Patch(color=color, label=label)
    )

# Malthusian zone label
ax_A.text(
    x_min_global + 10, y_min_global * 0.55,
    'Malthusian\nzone',
    fontsize=8, color='#bb4444', alpha=0.75,
    va='center', ha='left', style='italic'
)

ax_A.set_xlim(x_min_global, x_max_global)
ax_A.set_ylim(y_min_global, y_max_global)
ax_A.set_xlabel('Center year of 400-year window', fontsize=10)
ax_A.set_ylabel('Density -> Growth coefficient (β)', fontsize=10)
ax_A.set_title('(A)  Rolling Malthusian Coefficient by Agricultural Pathway',
               loc='left', fontsize=11, fontweight='bold', pad=8)
ax_A.legend(
    handles=legend_handles,
    fontsize=8.5, frameon=False,
    loc='upper right',
    ncol=2, handlelength=1.4, columnspacing=1.0
)
ax_A.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax_A.ticklabel_format(axis='y', style='sci', scilimits=(-5, -3))
ax_A.spines['left'].set_linewidth(0.8)
ax_A.spines['bottom'].set_linewidth(0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B — Bar chart of net density coefficients from interaction model
# ─────────────────────────────────────────────────────────────────────────────
bar_labels_order = [LABELS[cl] for cl in PLOT_ORDER]
bar_values = [NET_COEFS[lbl] for lbl in bar_labels_order]
bar_sigs = [SIG_FLAGS[lbl] for lbl in bar_labels_order]

# Color: positive = blue (#0072B2), negative = red (#D55E00)
bar_colors = ['#0072B2' if v >= 0 else '#D55E00' for v in bar_values]

x_pos = np.arange(len(bar_labels_order))
bars = ax_B.bar(
    x_pos, bar_values,
    color=bar_colors,
    width=0.6,
    edgecolor='white',
    linewidth=0.8,
    zorder=3,
)

# Zero reference line
ax_B.axhline(0, color='#555555', linewidth=0.9, zorder=2)

# Significance stars
for i, (val, sig) in enumerate(zip(bar_values, bar_sigs)):
    if sig:
        y_text = val + (abs(max(bar_values, key=abs)) * 0.04) * np.sign(val)
        va = 'bottom' if val >= 0 else 'top'
        ax_B.text(x_pos[i], y_text, '*', ha='center', va=va,
                  fontsize=13, color='#222222', fontweight='bold')

# Axis formatting
ax_B.set_xticks(x_pos)
ax_B.set_xticklabels(bar_labels_order, fontsize=9, rotation=15, ha='right')
ax_B.set_ylabel('Net density -> growth coefficient', fontsize=10)
ax_B.set_title('(B)  Net Density Coefficient by Pathway (Interaction Model)',
               loc='left', fontsize=11, fontweight='bold', pad=8)
ax_B.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax_B.ticklabel_format(axis='y', style='sci', scilimits=(-6, -4))
ax_B.set_xlim(-0.5, len(bar_labels_order) - 0.5)
ax_B.spines['left'].set_linewidth(0.8)
ax_B.spines['bottom'].set_linewidth(0.8)

# Legend for bar colors
pos_patch = mpatches.Patch(color='#0072B2', label='Positive (non-Malthusian)')
neg_patch = mpatches.Patch(color='#D55E00', label='Negative (Malthusian)')
ax_B.legend(handles=[pos_patch, neg_patch], fontsize=8.5, frameon=False,
            loc='upper right')

# Note about significance
ax_B.text(
    0.01, 0.02, '* p < 0.05',
    transform=ax_B.transAxes, fontsize=8, color='#555555',
    va='bottom', ha='left', style='italic'
)

# ── Figure title & save ───────────────────────────────────────────────────────
fig.suptitle(
    'Figure 6  |  Pathway-Specific Malthusian Dynamics',
    fontsize=11, fontweight='bold', y=1.01
)

OUT_BASE = '/Volumes/BIGDATA/HYDE35/analysis/figures/fig6_pathway_malthusian'
fig.savefig(OUT_BASE + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_BASE + '.pdf', bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT_BASE}.png")
print(f"Saved: {OUT_BASE}.pdf")
plt.close(fig)
