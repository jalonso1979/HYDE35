"""
Figure 10: The Climate-Agriculture-Population Nexus — Identification Strategy
Two-panel figure: (A) DAG-style causal diagram, (B) Summary results visual table.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── colour palette ───────────────────────────────────────────────────────────
C_CLIMATE_FILL   = '#dce8f5'
C_CLIMATE_EDGE   = '#2166ac'
C_AGPROD_FILL    = '#e8f5e8'
C_AGPROD_EDGE    = '#1a7a1a'
C_POP_FILL       = '#fff3cd'
C_POP_EDGE       = '#b07d00'

C_SOLID_ARROW    = '#2166ac'
C_ENDOG_ARROW    = '#888888'
C_EXCL_ARROW     = '#cc0000'
C_TEXT_DARK      = '#1a1a1a'

ROW_GREEN  = '#d4edda'
ROW_RED    = '#f8d7da'
ROW_BLUE   = '#d1ecf1'
ROW_LIGHT  = '#f8f9fa'
BORDER_COL = '#cccccc'

# ── figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 12), dpi=300)
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.08,
                      left=0.04, right=0.96, top=0.96, bottom=0.03)

ax_dag   = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[1])

for ax in (ax_dag, ax_table):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# ═══════════════════════════════════════════════════════════════════════════
# PANEL (A) — DAG
# ═══════════════════════════════════════════════════════════════════════════

ax_dag.text(0.012, 0.97, '(A)', fontsize=13, fontweight='bold',
            va='top', ha='left', color=C_TEXT_DARK,
            transform=ax_dag.transAxes)
ax_dag.text(0.5, 0.97,
            'Causal Identification Strategy (IV/2SLS)',
            fontsize=12, fontweight='bold', ha='center', va='top',
            color=C_TEXT_DARK, transform=ax_dag.transAxes)

# ── helper to draw a FancyBboxPatch with centred multi-line text ─────────
def draw_box(ax, cx, cy, w, h, title, subtitle_lines,
             fill, edge, title_size=11, sub_size=9):
    """Draw box centred at (cx,cy) in axes-fraction coords."""
    x0 = cx - w / 2
    y0 = cy - h / 2
    box = FancyBboxPatch((x0, y0), w, h,
                         boxstyle='round,pad=0.015',
                         facecolor=fill, edgecolor=edge,
                         linewidth=1.8, zorder=3,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    # title
    n_sub = len(subtitle_lines)
    total_lines = 1 + n_sub
    dy_title = (n_sub * 0.012) if n_sub else 0
    ax.text(cx, cy + dy_title + 0.005, title,
            fontsize=title_size, fontweight='bold',
            ha='center', va='center', color=edge,
            transform=ax.transAxes, zorder=4)
    for i, line in enumerate(subtitle_lines):
        y_sub = cy + dy_title - 0.022 * (i + 1)
        ax.text(cx, y_sub, line,
                fontsize=sub_size, ha='center', va='center',
                color='#444444', style='italic',
                transform=ax.transAxes, zorder=4)

# node positions (axes coords)
BOX_W, BOX_H = 0.26, 0.22
Y_MID = 0.575

CX_CLIM = 0.15
CX_AG   = 0.50
CX_POP  = 0.85

draw_box(ax_dag, CX_CLIM, Y_MID, BOX_W, BOX_H,
         'Climate Endowments',
         ['Crop composition', 'Rice suitability', 'Irrigation potential'],
         C_CLIMATE_FILL, C_CLIMATE_EDGE)

draw_box(ax_dag, CX_AG, Y_MID, BOX_W, BOX_H,
         'Agricultural\nProductivity',
         ['Land per capita', 'Crop yields'],
         C_AGPROD_FILL, C_AGPROD_EDGE)

draw_box(ax_dag, CX_POP, Y_MID, BOX_W, BOX_H,
         'Population\nGrowth',
         ['Malthusian dynamics'],
         C_POP_FILL, C_POP_EDGE)

# ── arrow helper using annotate ──────────────────────────────────────────
def draw_arrow(ax, x0, y0, x1, y1, color, lw=2.2, ls='-',
               arrowstyle='->', label=None, label_xy=None,
               label_size=9, label_color=None, label_rotation=0,
               connectionstyle='arc3,rad=0.0', zorder=2):
    ax.annotate('',
                xy=(x1, y1), xycoords='axes fraction',
                xytext=(x0, y0), textcoords='axes fraction',
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    color=color,
                    lw=lw,
                    linestyle=ls,
                    connectionstyle=connectionstyle,
                    shrinkA=4, shrinkB=4,
                ),
                zorder=zorder)
    if label and label_xy:
        lc = label_color if label_color else color
        ax.text(label_xy[0], label_xy[1], label,
                fontsize=label_size, ha='center', va='center',
                color=lc, fontweight='semibold',
                transform=ax.transAxes, zorder=5,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5))

# 1. Solid blue: Climate → AgProd (first stage)
draw_arrow(ax_dag,
           CX_CLIM + BOX_W/2, Y_MID,
           CX_AG   - BOX_W/2, Y_MID,
           color=C_SOLID_ARROW, lw=2.8,
           label='First stage\nF-stat = 59',
           label_xy=(0.325, Y_MID + 0.10),
           label_color=C_SOLID_ARROW)

# 2. Solid blue: AgProd → Population (second stage)
draw_arrow(ax_dag,
           CX_AG  + BOX_W/2, Y_MID,
           CX_POP - BOX_W/2, Y_MID,
           color=C_SOLID_ARROW, lw=2.8,
           label='Second stage (IV)',
           label_xy=(0.675, Y_MID + 0.10),
           label_color=C_SOLID_ARROW)

# 3. Dashed gray double-headed: AgProd ↔ Pop (endogeneity / reverse causality)
#    draw two arrows at a slight vertical offset to simulate double-headed
draw_arrow(ax_dag,
           CX_AG  + BOX_W/2, Y_MID - 0.06,
           CX_POP - BOX_W/2, Y_MID - 0.06,
           color=C_ENDOG_ARROW, lw=1.6, ls='dashed',
           arrowstyle='<->',
           label='Endogenous: reverse causality',
           label_xy=(0.675, Y_MID - 0.135),
           label_color=C_ENDOG_ARROW,
           connectionstyle='arc3,rad=0.0')

# 4. Dashed red: Climate → Pop direct path (exclusion restriction — crossed out)
#    Draw as curved dashed red arrow passing above
draw_arrow(ax_dag,
           CX_CLIM + BOX_W/2 - 0.01, Y_MID + 0.04,
           CX_POP  - BOX_W/2 + 0.01, Y_MID + 0.04,
           color=C_EXCL_ARROW, lw=1.8, ls=(0, (4, 3)),
           arrowstyle='-|>',
           connectionstyle='arc3,rad=-0.35',
           zorder=1)

# Red X mark at midpoint of the arc (roughly top of arc)
ax_dag.text(0.50, Y_MID + 0.245, '✕',
            fontsize=22, color=C_EXCL_ARROW, ha='center', va='center',
            fontweight='bold', transform=ax_dag.transAxes, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1))

ax_dag.text(0.50, Y_MID + 0.31,
            'Exclusion restriction (blocked)',
            fontsize=8.5, ha='center', va='center',
            color=C_EXCL_ARROW, style='italic',
            transform=ax_dag.transAxes, zorder=6)

# ── exclusion restriction text at bottom of panel ────────────────────────
restrict_y = 0.06
ax_dag.add_patch(FancyBboxPatch((0.05, restrict_y - 0.048), 0.90, 0.09,
                                boxstyle='round,pad=0.01',
                                facecolor='#fff8e1', edgecolor='#e6a817',
                                linewidth=1.4, transform=ax_dag.transAxes,
                                zorder=2, clip_on=False))
ax_dag.text(0.50, restrict_y,
            'Exclusion restriction: climate determines WHAT you grow, '
            'not HOW FAST you grow\n'
            '— except through agricultural productivity',
            fontsize=9.5, ha='center', va='center', style='italic',
            color='#5a3e00', transform=ax_dag.transAxes, zorder=3,
            linespacing=1.55)

# ── legend for arrow types ────────────────────────────────────────────────
legend_x, legend_y = 0.03, 0.30
ax_dag.annotate('', xy=(legend_x + 0.06, legend_y),
                xycoords='axes fraction',
                xytext=(legend_x, legend_y), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=C_SOLID_ARROW, lw=2.2))
ax_dag.text(legend_x + 0.072, legend_y, 'Causal path (IV)',
            fontsize=8, va='center', color=C_SOLID_ARROW,
            transform=ax_dag.transAxes)

legend_y2 = legend_y - 0.06
ax_dag.annotate('', xy=(legend_x + 0.06, legend_y2),
                xycoords='axes fraction',
                xytext=(legend_x, legend_y2), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color=C_ENDOG_ARROW,
                                lw=1.6, linestyle='dashed'))
ax_dag.text(legend_x + 0.072, legend_y2, 'Endogenous relationship',
            fontsize=8, va='center', color=C_ENDOG_ARROW,
            transform=ax_dag.transAxes)

legend_y3 = legend_y2 - 0.06
ax_dag.annotate('', xy=(legend_x + 0.06, legend_y3),
                xycoords='axes fraction',
                xytext=(legend_x, legend_y3), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=C_EXCL_ARROW,
                                lw=1.6, linestyle=(0, (4, 3))))
ax_dag.text(legend_x + 0.072, legend_y3, 'Blocked / excluded path',
            fontsize=8, va='center', color=C_EXCL_ARROW,
            transform=ax_dag.transAxes)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL (B) — Summary Results Visual Table
# ═══════════════════════════════════════════════════════════════════════════

ax_table.text(0.012, 0.97, '(B)', fontsize=13, fontweight='bold',
              va='top', ha='left', color=C_TEXT_DARK,
              transform=ax_table.transAxes)
ax_table.text(0.5, 0.97, 'Summary of Key Results',
              fontsize=12, fontweight='bold', ha='center', va='top',
              color=C_TEXT_DARK, transform=ax_table.transAxes)

# ── table geometry ────────────────────────────────────────────────────────
col_xs   = [0.03, 0.42, 0.59, 0.72, 0.87]   # left edge of each column
col_mids = [0.225, 0.505, 0.655, 0.795, 0.935]  # label x-centres
col_ws   = [col_xs[i+1] - col_xs[i] for i in range(len(col_xs)-1)] + [0.10]

headers  = ['Statistic / Estimator', 'Coefficient', 'p-value', 'Interpretation']
h_mids   = [(col_xs[i] + col_xs[i+1]) / 2 for i in range(len(col_xs)-1)]
h_mids.append((col_xs[-1] + 0.97) / 2)

# adjusted column boundaries including right edge
col_rights = col_xs[1:] + [0.97]

row_data = [
    # (label, coef, pval, interp, bg_color)
    ('First-stage F-statistic',  '59.0',       '—',       'Strong instruments\n(Stock-Yogo: >10)',   ROW_BLUE),
    ('OLS: agricultural land → pop. growth', '−0.000062', '<0.001', 'Biased downward\n(reverse causality)', ROW_RED),
    ('IV: agricultural land → pop. growth',  '+0.000187', '0.038',  'Causal: positive\nMalthusian effect',  ROW_GREEN),
    ('IV: pop. density → pop. growth',       '−0.000615', '<0.001', 'Malthusian constraint\n(crowding check)', ROW_LIGHT),
]

TABLE_TOP    = 0.86
ROW_H        = 0.155
HEADER_H     = 0.10
TABLE_LEFT   = col_xs[0]
TABLE_RIGHT  = 0.97
TABLE_BORDER = '#555555'

# header row
header_y = TABLE_TOP - HEADER_H
ax_table.add_patch(FancyBboxPatch(
    (TABLE_LEFT, header_y), TABLE_RIGHT - TABLE_LEFT, HEADER_H,
    boxstyle='square,pad=0', facecolor='#343a40', edgecolor=TABLE_BORDER,
    linewidth=1.2, transform=ax_table.transAxes, zorder=2, clip_on=False))

for i, (hdr, hx) in enumerate(zip(headers, h_mids)):
    ax_table.text(hx, header_y + HEADER_H / 2, hdr,
                  fontsize=9.5, fontweight='bold', color='white',
                  ha='center', va='center', transform=ax_table.transAxes,
                  zorder=3)

# data rows
for r, (label, coef, pval, interp, bg) in enumerate(row_data):
    ry = header_y - (r + 1) * ROW_H
    # row background
    ax_table.add_patch(FancyBboxPatch(
        (TABLE_LEFT, ry), TABLE_RIGHT - TABLE_LEFT, ROW_H,
        boxstyle='square,pad=0', facecolor=bg, edgecolor=BORDER_COL,
        linewidth=0.8, transform=ax_table.transAxes, zorder=2, clip_on=False))

    row_cy = ry + ROW_H / 2

    # column 0: label
    ax_table.text(col_xs[0] + 0.01, row_cy, label,
                  fontsize=9, ha='left', va='center',
                  color=C_TEXT_DARK, transform=ax_table.transAxes, zorder=3)

    # column 1: coefficient
    coef_color = '#155724' if bg == ROW_GREEN else ('#721c24' if bg == ROW_RED else C_TEXT_DARK)
    ax_table.text(h_mids[1], row_cy, coef,
                  fontsize=10, ha='center', va='center',
                  fontweight='bold', color=coef_color,
                  transform=ax_table.transAxes, zorder=3,
                  fontfamily='monospace')

    # column 2: p-value
    ax_table.text(h_mids[2], row_cy, pval,
                  fontsize=9.5, ha='center', va='center',
                  color=C_TEXT_DARK, transform=ax_table.transAxes, zorder=3)

    # column 3: interpretation
    ax_table.text(h_mids[3], row_cy, interp,
                  fontsize=8.5, ha='center', va='center',
                  color='#333333', style='italic',
                  transform=ax_table.transAxes, zorder=3,
                  linespacing=1.4)

    # vertical dividers
    for cx in col_xs[1:]:
        ax_table.plot([cx, cx], [ry, ry + ROW_H], color=BORDER_COL,
                      lw=0.7, transform=ax_table.transAxes, zorder=3)

# outer border
table_bot = header_y - len(row_data) * ROW_H
ax_table.add_patch(FancyBboxPatch(
    (TABLE_LEFT, table_bot), TABLE_RIGHT - TABLE_LEFT,
    TABLE_TOP - table_bot - (TABLE_TOP - (header_y + HEADER_H)),
    boxstyle='square,pad=0', facecolor='none', edgecolor=TABLE_BORDER,
    linewidth=1.6, transform=ax_table.transAxes, zorder=4, clip_on=False))

# colour-key note
note_y = table_bot - 0.065
ax_table.text(TABLE_LEFT, note_y,
              'Note:  ',
              fontsize=8, va='center', color='#555555',
              transform=ax_table.transAxes)

for i, (label, color) in enumerate([
        ('IV causal estimate', ROW_GREEN),
        ('Biased OLS', ROW_RED),
        ('Instrument strength', ROW_BLUE),
]):
    nx = TABLE_LEFT + 0.07 + i * 0.26
    ax_table.add_patch(FancyBboxPatch(
        (nx, note_y - 0.022), 0.016, 0.035,
        boxstyle='round,pad=0.004', facecolor=color,
        edgecolor=BORDER_COL, linewidth=0.8,
        transform=ax_table.transAxes, zorder=3))
    ax_table.text(nx + 0.025, note_y, label,
                  fontsize=7.5, va='center', color='#444444',
                  transform=ax_table.transAxes)

# ── save ─────────────────────────────────────────────────────────────────
out_base = '/Volumes/BIGDATA/HYDE35/analysis/figures/fig10_iv_dag'
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(out_base + '.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print('Saved:', out_base + '.png')
print('Saved:', out_base + '.pdf')
