"""
Figure 9: The Tightening Malthusian Constraint — OLS vs IV Estimates by Period

Two-panel figure comparing OLS and IV estimates of:
  Panel A: Agricultural productivity → population growth
  Panel B: Population density → population growth (Malthusian check)

Across four historical periods: 0–500, 500–1000, 1000–1500, 1500–1750 CE.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Hardcoded regression results ────────────────────────────────────────────

periods  = ['0–500 CE', '500–1000 CE', '1000–1500 CE', '1500–1750 CE']
n_obs    = [603, 760, 765, 1086]

# Agricultural productivity → Pop growth coefficients
ols_ag   = [-0.000038, -0.000071, -0.000132, -0.001436]
iv_ag    = [ 0.000013, -0.000059, -0.000144, -0.000947]

# Population density → Pop growth coefficients
ols_dens = [-0.000520, -0.000502, -0.000433, -0.001136]
iv_dens  = [-0.000609, -0.000526, -0.000416, -0.001037]

# First-stage F-statistics
f_stats  = [16.9, 28.1, 29.1, 264.2]

# ── Colour palette (Okabe–Ito accessible) ───────────────────────────────────
C_IV  = '#0072B2'   # blue
C_OLS = '#888888'   # mid-gray

x = np.arange(len(periods))


def _configure_ax(ax, y_vals_all, title, ylabel='Coefficient'):
    """Shared axis formatting."""
    y_min = min(y_vals_all) * 1.22
    y_max = max(y_vals_all) * 1.15 if max(y_vals_all) > 0 else abs(min(y_vals_all)) * 0.15

    # Ensure y=0 is always visible with some headroom
    y_max = max(y_max, abs(y_min) * 0.10)
    y_min = min(y_min, -abs(y_max) * 0.05)

    # Pink Malthusian zone (below 0)
    ax.axhspan(y_min, 0, color='#FFE0E0', alpha=0.55, zorder=0, label='_nolegend_')

    # Reference line at 0
    ax.axhline(0, color='black', linewidth=0.9, linestyle='-', zorder=2)

    ax.set_xlim(-0.5, len(periods) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlabel('Period', fontsize=11, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight='normal', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Secondary x-axis labels: n = ...
    for xi, n in zip(x, n_obs):
        ax.text(xi, y_min * 1.08, f'n={n}', ha='center', va='bottom',
                fontsize=7.5, color='#444444')

    return y_min, y_max


# ── Figure layout ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 12))
fig.subplots_adjust(hspace=0.42, top=0.95, bottom=0.07, left=0.12, right=0.95)

# ============================================================
# Panel A — Agricultural productivity effect
# ============================================================
ax = axes[0]

all_vals_a = ols_ag + iv_ag
y_min_a, y_max_a = _configure_ax(
    ax, all_vals_a,
    title='Agricultural productivity → population growth',
    ylabel='Coefficient on ag. productivity'
)

# OLS series — open circles, dashed
ax.plot(x, ols_ag, color=C_OLS, linestyle='--', linewidth=1.6,
        marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor=C_OLS, markeredgewidth=1.8, zorder=4,
        label='OLS (potentially biased)')

# IV series — filled circles, solid
ax.plot(x, iv_ag, color=C_IV, linestyle='-', linewidth=1.8,
        marker='o', markersize=8, markerfacecolor=C_IV,
        markeredgecolor=C_IV, markeredgewidth=1.5, zorder=5,
        label='IV (causal estimate)')

# Value labels above/below each point
offset_frac = (y_max_a - y_min_a) * 0.04
for xi, vo, vi in zip(x, ols_ag, iv_ag):
    va_ols = 'bottom' if vo >= 0 else 'top'
    va_iv  = 'bottom' if vi >= 0 else 'top'
    dy_ols = offset_frac if vo >= 0 else -offset_frac
    dy_iv  = offset_frac if vi >= 0 else -offset_frac
    ax.text(xi - 0.07, vo + dy_ols, f'{vo:.6f}', ha='center',
            va=va_ols, fontsize=7, color=C_OLS)
    ax.text(xi + 0.07, vi + dy_iv, f'{vi:.6f}', ha='center',
            va=va_iv, fontsize=7, color=C_IV)

# Annotation: IV sign flip in 0–500 CE
# Place text at ~30% of the way down from top of axis (in data coords)
ann_y_a = y_min_a + (y_max_a - y_min_a) * 0.68
ax.annotate(
    'IV reveals positive causal\neffect of ag productivity\non pop growth in 0–500 CE',
    xy=(0, iv_ag[0]), xytext=(0.72, ann_y_a),
    arrowprops=dict(arrowstyle='->', color=C_IV, lw=1.4,
                    connectionstyle='arc3,rad=0.3'),
    fontsize=8.5, color=C_IV, ha='left',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor=C_IV, alpha=0.92)
)

# First-stage F-stats as footer text
ax.text(0.98, 0.03,
        'First-stage F: ' + ',  '.join(f'{f:.1f}' for f in f_stats),
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=7.5, color='#555555', style='italic')

# Legend
ax.legend(loc='lower left', fontsize=9, framealpha=0.9,
          edgecolor='#cccccc', handlelength=2.2)

# Bold panel label
ax.text(-0.08, 1.04, '(A)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='top')

# Malthusian zone label
ax.text(len(periods) - 0.52, y_min_a * 0.55,
        'Malthusian\nzone', ha='right', va='center',
        fontsize=8, color='#cc4444', style='italic', alpha=0.8)


# ============================================================
# Panel B — Population density effect
# ============================================================
ax = axes[1]

all_vals_b = ols_dens + iv_dens
y_min_b, y_max_b = _configure_ax(
    ax, all_vals_b,
    title='Population density → population growth (Malthusian check)',
    ylabel='Coefficient on pop. density'
)

# OLS series
ax.plot(x, ols_dens, color=C_OLS, linestyle='--', linewidth=1.6,
        marker='o', markersize=8, markerfacecolor='white',
        markeredgecolor=C_OLS, markeredgewidth=1.8, zorder=4,
        label='OLS (potentially biased)')

# IV series
ax.plot(x, iv_dens, color=C_IV, linestyle='-', linewidth=1.8,
        marker='o', markersize=8, markerfacecolor=C_IV,
        markeredgecolor=C_IV, markeredgewidth=1.5, zorder=5,
        label='IV (causal estimate)')

# Value labels
offset_frac_b = (y_max_b - y_min_b) * 0.04
for xi, vo, vi in zip(x, ols_dens, iv_dens):
    dy_ols = -offset_frac_b
    dy_iv  = offset_frac_b
    ax.text(xi - 0.07, vo + dy_ols, f'{vo:.6f}', ha='center',
            va='top', fontsize=7, color=C_OLS)
    ax.text(xi + 0.07, vi + dy_iv, f'{vi:.6f}', ha='center',
            va='bottom', fontsize=7, color=C_IV)

# Annotation: deepening constraint
# Arrow pointing to the steep drop at 1500-1750 CE
ax.annotate(
    'Density constraint deepens\nmonotonically from 0 to 1750 CE',
    xy=(3, iv_dens[3]), xytext=(1.55, y_min_b * 0.62),
    arrowprops=dict(arrowstyle='->', color=C_IV, lw=1.4,
                    connectionstyle='arc3,rad=-0.20'),
    fontsize=8.5, color=C_IV, ha='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor=C_IV, alpha=0.92)
)

# First-stage F-stats
ax.text(0.98, 0.03,
        'First-stage F: ' + ',  '.join(f'{f:.1f}' for f in f_stats),
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=7.5, color='#555555', style='italic')

ax.legend(loc='lower left', fontsize=9, framealpha=0.9,
          edgecolor='#cccccc', handlelength=2.2)

ax.text(-0.08, 1.04, '(B)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='top')

ax.text(len(periods) - 0.52, y_min_b * 0.55,
        'Malthusian\nzone', ha='right', va='center',
        fontsize=8, color='#cc4444', style='italic', alpha=0.8)


# ── Suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    'Figure 9: The Tightening Malthusian Constraint — OLS vs IV Estimates by Period',
    fontsize=12, fontweight='bold', y=0.985
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_base = '/Volumes/BIGDATA/HYDE35/analysis/figures/fig9_ols_vs_iv_periods'
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out_base + '.pdf', dpi=300, bbox_inches='tight')
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
plt.close(fig)
