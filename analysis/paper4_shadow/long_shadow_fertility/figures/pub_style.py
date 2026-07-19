"""Publication-quality grayscale matplotlib stylesheet for Long Shadow on Fertility.

Rationale
---------
Journal print robustness: most economics journals print in black-and-white.
Colorblind accessibility: ~8% of male readers cannot distinguish red/green.

Strategy: distinguish series by BOTH gray level AND linestyle (+ optional marker),
never by color alone. This ensures figures are readable in:
  - Grayscale print
  - Colorblind conditions (all common types)
  - Low-resolution screen renders

Usage
-----
    from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import (
        apply_pub_style, style_lines, GRAYS, LINESTYLES, MARKERS, SEQUENTIAL_CMAP
    )

    apply_pub_style()
    fig, ax = plt.subplots()
    style_lines(ax)
    for label, data in series.items():
        ax.plot(data["x"], data["y"], label=label)
"""
from __future__ import annotations

import itertools
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

# ---------------------------------------------------------------------------
# Grayscale palette: 6 levels, from black to light gray
# ---------------------------------------------------------------------------
GRAYS: list[str] = [
    "#000000",  # black
    "#444444",  # dark gray
    "#777777",  # medium gray
    "#aaaaaa",  # light-medium gray
    "#cccccc",  # light gray
    "#eeeeee",  # very light gray (use only on white background, with thick lines)
]

# ---------------------------------------------------------------------------
# Linestyles: 6 distinct styles for unambiguous series identification
# ---------------------------------------------------------------------------
LINESTYLES: list = [
    "-",                # solid
    "--",               # dashed
    ":",                # dotted
    "-.",               # dash-dot
    (0, (3, 1, 1, 1)),  # densely dash-dot-dot
    (0, (5, 1)),        # densely dashed
]

# ---------------------------------------------------------------------------
# Markers: 6 distinct markers (used for scatter / annotated line plots)
# ---------------------------------------------------------------------------
MARKERS: list[str] = ["o", "s", "^", "D", "v", "x"]

# ---------------------------------------------------------------------------
# Sequential colormap recommendation for stacked-area / FEVD charts
# ---------------------------------------------------------------------------
# "Greys" is pure grayscale — best for true B&W print.
# "cividis" is colorblind-safe and converts to near-monotone grayscale.
SEQUENTIAL_CMAP: str = "Greys"
SEQUENTIAL_CMAP_COLORBLIND: str = "cividis"


def apply_pub_style(
    font_size: int = 10,
    dpi: int = 300,
    serif: bool = True,
) -> None:
    """Set matplotlib rcParams for publication-quality output.

    Parameters
    ----------
    font_size:
        Base font size in points. Axes labels and ticks scale from this.
    dpi:
        Raster DPI for any PNG saves. Vector PDF is unaffected by DPI.
    serif:
        If True (default), use Computer Modern / DejaVu Serif for a TeX-like look.
        Set False to use a clean sans-serif (DejaVu Sans) for slide figures.
    """
    font_family = "serif" if serif else "sans-serif"
    font_list = (
        ["Computer Modern", "DejaVu Serif", "Times New Roman", "serif"]
        if serif
        else ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    )

    matplotlib.rcParams.update(
        {
            # --- fonts ---
            "font.family": font_family,
            "font.serif": font_list if serif else [],
            "font.sans-serif": font_list if not serif else [],
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            # --- PDF vector output: embed fonts (Type 42 = TrueType, editable in Illustrator) ---
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # --- raster DPI (for PNG saves) ---
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            # --- tight layout by default ---
            "savefig.bbox": "tight",
            "figure.autolayout": False,  # we call tight_layout() explicitly
            # --- thin, clean axes ---
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # --- grid (light, off by default — enable per-figure if needed) ---
            "axes.grid": False,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.4,
            "grid.color": "#cccccc",
            # --- lines ---
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.6,
            # --- legend ---
            "legend.framealpha": 0.7,
            "legend.edgecolor": "#cccccc",
            "legend.borderpad": 0.4,
            # --- text rendering ---
            "text.usetex": False,  # avoid LaTeX dependency; use mathtext
            "mathtext.fontset": "dejavuserif" if serif else "dejavusans",
        }
    )


def style_lines(
    ax: "matplotlib.axes.Axes",
    n: Optional[int] = None,
    use_markers: bool = False,
    marker_every: int = 10,
) -> None:
    """Apply a gray-level + linestyle property cycle to the axes.

    Call this BEFORE plotting so that subsequent `ax.plot(...)` calls
    automatically cycle through the grayscale+linestyle combinations.

    Parameters
    ----------
    ax:
        Target axes.
    n:
        Number of series expected. If None, cycles through all 6 levels.
        Useful to clip to a shorter palette when n < 6.
    use_markers:
        If True, also cycle through MARKERS (good for scatter-heavy figures).
    marker_every:
        Plot a marker every this many data points (only used if use_markers=True).
    """
    n_levels = min(n, len(GRAYS)) if n is not None else len(GRAYS)
    grays = GRAYS[:n_levels]
    ls = LINESTYLES[:n_levels]
    mk = MARKERS[:n_levels]

    if use_markers:
        prop = cycler(color=grays) + cycler(linestyle=ls) + cycler(marker=mk)
    else:
        prop = cycler(color=grays) + cycler(linestyle=ls)

    ax.set_prop_cycle(prop)


def get_sequential_cmap(colorblind_safe: bool = False) -> str:
    """Return the recommended sequential colormap name.

    Parameters
    ----------
    colorblind_safe:
        If True, return 'cividis' (colorblind-safe, near-monotone in grayscale).
        If False (default), return 'Greys' (true grayscale).
    """
    return SEQUENTIAL_CMAP_COLORBLIND if colorblind_safe else SEQUENTIAL_CMAP
