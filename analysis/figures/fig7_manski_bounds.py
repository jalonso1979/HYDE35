"""
Figure 7: Manski Bounds and Pre/Post Structural Break Comparison

Panel A: Manski partial identification bounds on the density coefficient
Panel B: Pre vs post structural break density coefficients
"""

import sys
sys.path.insert(0, '/Volumes/BIGDATA/HYDE35')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── Wong 2011 colorblind palette ──────────────────────────────────────────────
C_BLUE   = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN  = '#009E73'
C_RED    = '#D55E00'
C_GRAY   = '#888888'

# ── Provided values ───────────────────────────────────────────────────────────
MANSKI_LOWER  = -0.00001276
MANSKI_BASE   = -0.00000955
MANSKI_UPPER  = -0.00000229

PRE_COEF   =  0.0         # p=0.96, not significant
PRE_PVAL   =  0.96
POST_COEF  = -0.00004477  # p=0.002, significant
POST_PVAL  =  0.002

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

fig, (ax_A, ax_B) = plt.subplots(
    2, 1, figsize=(8, 8),
    gridspec_kw={'hspace': 0.52}
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Manski Bounds
# ─────────────────────────────────────────────────────────────────────────────
points = {
    'Lower\nbound':  MANSKI_LOWER,
    'Baseline\nestimate': MANSKI_BASE,
    'Upper\nbound':  MANSKI_UPPER,
}
y_pos = 1   # single horizontal strip — all on y=1

labels = list(points.keys())
values = list(points.values())
dot_colors = [C_RED, C_BLUE, C_ORANGE]   # lower=red, base=blue, upper=orange

# Line segment connecting lower to upper bound
ax_A.hlines(y=y_pos, xmin=MANSKI_LOWER, xmax=MANSKI_UPPER,
            color=C_GRAY, linewidth=2.5, zorder=2)

# Shaded interval band
ax_A.fill_betweenx(
    [y_pos - 0.18, y_pos + 0.18],
    MANSKI_LOWER, MANSKI_UPPER,
    color=C_GRAY, alpha=0.12, zorder=1
)

# Dots
for val, lbl, col in zip(values, labels, dot_colors):
    ax_A.scatter([val], [y_pos], color=col, s=110, zorder=4,
                 edgecolors='white', linewidths=1.0)
    # Label below each dot
    offset_y = -0.22
    ax_A.text(val, y_pos + offset_y, lbl,
              ha='center', va='top', fontsize=8.5, color='#333333')
    # Numeric value above dot
    ax_A.text(val, y_pos + 0.17,
              f'{val:.2e}',
              ha='center', va='bottom', fontsize=7.5, color=col, fontweight='bold')

# Vertical dashed line at zero
ax_A.axvline(0, color='#444444', linewidth=1.1, linestyle='--', zorder=3)
ax_A.text(0, y_pos + 0.38, 'Zero', ha='center', va='bottom',
          fontsize=8, color='#444444', style='italic')

# "All bounds < 0" annotation
ax_A.annotate(
    'All bounds negative\n-> robust Malthusian effect',
    xy=(MANSKI_UPPER, y_pos),
    xytext=(MANSKI_UPPER + abs(MANSKI_LOWER) * 0.35, y_pos + 0.5),
    fontsize=8.5, color='#222222', ha='left', va='center',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0,
                    connectionstyle='arc3,rad=-0.25'),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor='#cccccc', alpha=0.9)
)

ax_A.set_xlim(MANSKI_LOWER * 1.5, abs(MANSKI_LOWER) * 0.9)
ax_A.set_ylim(0.3, 1.8)
ax_A.set_xlabel('Density -> Growth coefficient (β)', fontsize=10)
ax_A.set_yticks([])
ax_A.spines['left'].set_visible(False)
ax_A.spines['bottom'].set_linewidth(0.8)
ax_A.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax_A.ticklabel_format(axis='x', style='sci', scilimits=(-6, -4))
ax_A.set_title('(A)  Manski Partial Identification Bounds',
               loc='left', fontsize=11, fontweight='bold', pad=8)

# Legend
dot_handles = [
    mpatches.Patch(color=C_RED,    label=f'Lower bound ({MANSKI_LOWER:.2e})'),
    mpatches.Patch(color=C_BLUE,   label=f'Baseline ({MANSKI_BASE:.2e})'),
    mpatches.Patch(color=C_ORANGE, label=f'Upper bound ({MANSKI_UPPER:.2e})'),
]
ax_A.legend(handles=dot_handles, fontsize=8, frameon=False,
            loc='upper left', handlelength=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B — Pre vs Post Structural Break
# ─────────────────────────────────────────────────────────────────────────────
bar_vals   = [PRE_COEF, POST_COEF]
bar_labels = ['Pre-break\n(p = 0.96)', 'Post-break\n(p = 0.002)']
bar_colors = [C_GRAY, C_BLUE]   # gray for n.s., blue for sig (negative = Malthusian)
bar_sig    = [False, True]

x_pos = np.array([0, 1])
bars = ax_B.bar(
    x_pos, bar_vals,
    color=bar_colors,
    width=0.5,
    edgecolor='white',
    linewidth=0.8,
    zorder=3
)

# Zero reference
ax_B.axhline(0, color='#555555', linewidth=0.9, zorder=2)

# Significance stars and value labels
y_scale = abs(POST_COEF)
for i, (val, sig, lbl_txt) in enumerate(zip(bar_vals, bar_sig, bar_labels)):
    if sig:
        y_star = val - y_scale * 0.06
        ax_B.text(x_pos[i], y_star, '**', ha='center', va='top',
                  fontsize=13, color='#222222', fontweight='bold')
    else:
        ax_B.text(x_pos[i], y_scale * 0.03, 'n.s.', ha='center', va='bottom',
                  fontsize=9, color='#666666', style='italic')
    # Numeric value inside/near bar
    if abs(val) > y_scale * 0.02:
        offset = -y_scale * 0.06 if val < 0 else y_scale * 0.03
        va = 'top' if val < 0 else 'bottom'
        ax_B.text(x_pos[i], val + offset,
                  f'{val:.2e}',
                  ha='center', va=va, fontsize=8, color='white' if val < 0 else '#444444',
                  fontweight='bold')

ax_B.set_xticks(x_pos)
ax_B.set_xticklabels(bar_labels, fontsize=9)
ax_B.set_ylabel('Density -> Growth coefficient (β)', fontsize=10)
ax_B.set_title('(B)  Density Coefficient: Pre vs Post Structural Break',
               loc='left', fontsize=11, fontweight='bold', pad=8)
ax_B.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax_B.ticklabel_format(axis='y', style='sci', scilimits=(-6, -4))
ax_B.set_xlim(-0.5, 1.5)
# Pad y range so labels fit
y_lo = POST_COEF * 1.35
y_hi = abs(POST_COEF) * 0.55
ax_B.set_ylim(y_lo, y_hi)
ax_B.spines['left'].set_linewidth(0.8)
ax_B.spines['bottom'].set_linewidth(0.8)

# Note
ax_B.text(
    0.99, 0.98,
    '** p < 0.01  |  n.s. = not significant',
    transform=ax_B.transAxes, fontsize=8, color='#555555',
    va='top', ha='right', style='italic'
)

# ── Figure title & save ───────────────────────────────────────────────────────
fig.suptitle(
    'Figure 7  |  Manski Bounds and Pre/Post Structural Break Comparison',
    fontsize=11, fontweight='bold', y=1.01
)

OUT_BASE = '/Volumes/BIGDATA/HYDE35/analysis/figures/fig7_manski_bounds'
fig.savefig(OUT_BASE + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_BASE + '.pdf', bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT_BASE}.png")
print(f"Saved: {OUT_BASE}.pdf")
plt.close(fig)
