"""Sober grayscale figure style for the Long Shadow paper.

Use:
    from figstyle import set_style, GRAYS, LINESTYLES
    set_style()
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt


GRAYS = ["#000000", "#404040", "#707070", "#A0A0A0", "#C8C8C8", "#E0E0E0"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def set_style() -> None:
    """Configure matplotlib for sober academic figures (sans-serif).

    Matches the paper's typographic style: clean sans-serif, restrained,
    generous white space, no rainbow colors.
    """
    mpl.rcParams.update({
        # Fonts — sans-serif throughout to match paper typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue",
                             "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#202020",
        "axes.labelcolor": "#202020",
        "axes.titlecolor": "#000000",
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        # Ticks
        "xtick.color": "#202020",
        "ytick.color": "#202020",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid
        "axes.grid": True,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        # Patches
        "patch.linewidth": 0.6,
        "patch.edgecolor": "#202020",
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        # Save / output
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def gray_palette(n: int) -> list[str]:
    """Return n shades of gray, dark to light."""
    if n <= len(GRAYS):
        return GRAYS[:n]
    import numpy as np
    return [f"#{int(g):02x}{int(g):02x}{int(g):02x}" for g in
            np.linspace(0, 200, n).astype(int)]


def style_box(ax, bp, palette: list[str] | None = None) -> None:
    """Apply consistent boxplot styling."""
    if palette is None:
        palette = gray_palette(len(bp["boxes"]))
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_edgecolor("#202020")
        patch.set_linewidth(0.6)
    for elem in ["whiskers", "caps", "medians"]:
        for line in bp[elem]:
            line.set_color("#202020")
            line.set_linewidth(0.7)
    if "means" in bp:
        for m in bp["means"]:
            m.set_marker("o")
            m.set_markerfacecolor("white")
            m.set_markeredgecolor("#202020")
            m.set_markersize(4)
    for f in bp["fliers"]:
        f.set_marker("+")
        f.set_markersize(3)
        f.set_markeredgecolor("#606060")
