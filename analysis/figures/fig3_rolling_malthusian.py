"""
Figure 3: Time-Varying Malthusian Coefficient (0–1750 CE)

Rolling 200-year window OLS with entity fixed effects.
Dependent variable: pop_growth_rate
Key variable: popdens_lag (Malthusian density coefficient)
Control: land_labor_ratio_lag
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pycountry

# Ensure project root is on path
sys.path.insert(0, '/Volumes/BIGDATA/HYDE35')

from analysis.shared.config import ANALYSIS_DATA
from analysis.paper2_malthus.panels import build_malthusian_panel
from analysis.paper2_malthus.regressions import run_rolling_window

# ── 1. Load and filter panel ────────────────────────────────────────────────
cp = pd.read_parquet(ANALYSIS_DATA / 'country_analysis_panel.parquet')

def iso_num_to_alpha3(num):
    try:
        c = pycountry.countries.get(numeric=str(int(num)).zfill(3))
        return c.alpha_3 if c else str(int(num))
    except Exception:
        return str(int(num))

code_map = {c: iso_num_to_alpha3(c) for c in cp.country.unique()}
cp['iso3'] = cp['country'].map(code_map)
cp_post0 = cp[cp.year >= 0]
high_pop = cp_post0.groupby('iso3')['popdens_p_km2_mean'].max()
valid = high_pop[high_pop > 1.0].index
cp_f = cp_post0[cp_post0.iso3.isin(valid)]
malthus = build_malthusian_panel(cp_f, entity_col='iso3')

# ── 2. Rolling window regression ────────────────────────────────────────────
rolling = run_rolling_window(
    malthus, dep_var='pop_growth_rate', key_var='popdens_lag',
    control_vars=['land_labor_ratio_lag'],
    entity_col='iso3', window_years=200, step_years=50,
)

print(f"Rolling windows computed: {len(rolling)}")
print(rolling.to_string(index=False))

# ── 3. Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x = rolling['center_year'].values
y = rolling['coefficient'].values
p = rolling['pvalue'].values

sig = p < 0.05

# Shaded Malthusian zone (y < 0)
y_min = min(y.min(), 0) * 1.25
ax.fill_between([x.min() - 30, x.max() + 30], y_min, 0,
                color='#ffcccc', alpha=0.35, zorder=0, linewidth=0)

# Horizontal reference line at 0
ax.axhline(0, color='#888888', linewidth=0.9, linestyle='--', zorder=1)

# Line connecting all points
ax.plot(x, y, color='#555555', linewidth=1.2, zorder=2, alpha=0.7)

# Significant points (red filled circles)
ax.scatter(x[sig], y[sig], color='#cc2222', s=60, zorder=4,
           label='p < 0.05', edgecolors='#990000', linewidths=0.6)

# Non-significant points (gray open circles)
ax.scatter(x[~sig], y[~sig], facecolors='none', edgecolors='#777777',
           s=55, zorder=3, linewidths=1.1, label='p ≥ 0.05')

# ── 4. Annotations ───────────────────────────────────────────────────────────
arrowprops_left = dict(arrowstyle='->', color='#333333', lw=1.2,
                       connectionstyle='arc3,rad=-0.2')
arrowprops_right = dict(arrowstyle='->', color='#333333', lw=1.2,
                        connectionstyle='arc3,rad=0.2')

# Find the actual trough near 1050 CE
window_1050 = rolling[(rolling['center_year'] >= 950) & (rolling['center_year'] <= 1150)]
if not window_1050.empty:
    trough_row = window_1050.loc[window_1050['coefficient'].idxmin()]
    trough_x = trough_row['center_year']
    trough_y = trough_row['coefficient']
else:
    trough_x, trough_y = 1050, y.min()

# Find the actual spike near 1250 CE
window_1250 = rolling[(rolling['center_year'] >= 1150) & (rolling['center_year'] <= 1350)]
if not window_1250.empty:
    spike_row = window_1250.loc[window_1250['coefficient'].idxmin()]
    spike_x = spike_row['center_year']
    spike_y = spike_row['coefficient']
else:
    spike_x, spike_y = 1250, y.min() * 0.5

# Find the relief near 1350 CE
window_1350 = rolling[(rolling['center_year'] >= 1300) & (rolling['center_year'] <= 1450)]
if not window_1350.empty:
    relief_row = window_1350.loc[window_1350['coefficient'].idxmax()]
    relief_x = relief_row['center_year']
    relief_y = relief_row['coefficient']
else:
    relief_x, relief_y = 1350, 0.0

# Annotation 1: Peak Malthusian constraint ~1050
ax.annotate(
    'Peak Malthusian\nconstraint ~1050',
    xy=(trough_x, trough_y),
    xytext=(trough_x - 180, trough_y - abs(trough_y) * 0.5),
    fontsize=9, color='#222222',
    arrowprops=arrowprops_left,
    ha='center', va='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85)
)

# Annotation 2: Pre-Black Death pressure ~1250
ax.annotate(
    'Pre-Black Death\npressure ~1250',
    xy=(spike_x, spike_y),
    xytext=(spike_x + 160, spike_y - abs(spike_y) * 0.45),
    fontsize=9, color='#222222',
    arrowprops=arrowprops_right,
    ha='center', va='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85)
)

# Annotation 3: Post-plague relief ~1350
ax.annotate(
    'Post-plague\nrelief ~1350',
    xy=(relief_x, relief_y),
    xytext=(relief_x + 155, relief_y + abs(y).max() * 0.3),
    fontsize=9, color='#222222',
    arrowprops=arrowprops_right,
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85)
)

# ── 5. Axes and labels ───────────────────────────────────────────────────────
ax.set_xlabel('Center year of 200-year window', fontsize=11)
ax.set_ylabel('Density → Growth coefficient (β)', fontsize=11)
ax.tick_params(axis='both', labelsize=9)

# Pad x-axis slightly
ax.set_xlim(x.min() - 40, x.max() + 40)

# Legend
legend = ax.legend(fontsize=9, framealpha=0.9, loc='upper right',
                   handletextpad=0.5, borderpad=0.7)

# Subtle Malthusian zone label inside the shaded region
ax.text(x.min() - 20, y_min * 0.55, 'Malthusian\nzone',
        fontsize=8, color='#bb4444', alpha=0.7, va='center', ha='left',
        style='italic')

# Clean spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()

# ── 6. Save ──────────────────────────────────────────────────────────────────
out_base = '/Volumes/BIGDATA/HYDE35/analysis/figures/fig3_rolling_malthusian'
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out_base + '.pdf', bbox_inches='tight')
print(f"\nSaved: {out_base}.png")
print(f"Saved: {out_base}.pdf")
plt.close(fig)
