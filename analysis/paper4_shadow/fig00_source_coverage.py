"""Figure 0: source-coverage timeline.

Horizontal-bar chart showing the temporal extent of every data source the
paper assembles into the monthly paleo-economic panel. Plotted at calendar
resolution, sorted by start year, with each source coded by category
(climate, land use & population, vital statistics, volcanic, conflict,
prices & wages).

The plot sells the synthesis at a glance and is meant for the opening of
Section 2 (the data section).

Output: analysis/figures/paper4_v2/fig00_source_coverage.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

FIG = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/paper4_v2")

# (label, start_year, end_year, category, note)
SOURCES = [
    ("Harper Roman-Egypt wheat prices",          45, 650, "prices",    None),
    ("HYDE 3.5 land use & population (10ka BCE-)",     -100, 2025, "landuse", "shown from 100 BCE for legibility"),
    ("Brecke pre-1400 European conflicts",       900, 1402, "conflict", None),
    ("Allen-Nuffield wage and price archive",    1259, 1914, "prices",   None),
    ("Brecke Conflict Catalogue v18",            1400, 1999, "conflict", None),
    ("ModE-RA paleo-reanalysis (monthly)",       1421, 2008, "climate",  None),
    ("eVolv2k volcanic record (500 BCE-)",       1421, 1900, "volcanic", "shown from 1421 for legibility"),
    ("CamPop 26-parish family reconstitution",   1538, 1851, "vital",    None),
    ("HMD/HFD vital statistics (Sweden lead)",   1751, 2022, "vital",    None),
    ("CRU TS 4.09 climatology (1901-1950 ref)",  1901, 1950, "climate",  None),
    ("ERA5 reanalysis (monthly)",                1950, 2025, "climate",  None),
]

CATEGORY_COLORS = {
    "climate":  "#1f77b4",  # blue
    "landuse":  "#2ca02c",  # green
    "volcanic": "#d62728",  # red
    "conflict": "#9467bd",  # purple
    "vital":    "#ff7f0e",  # orange
    "prices":   "#8c564b",  # brown
}

CATEGORY_LABELS = {
    "climate":  "Climate (ModE-RA, CRU, ERA5)",
    "landuse":  "Land use & population (HYDE 3.5)",
    "volcanic": "Volcanic record (eVolv2k)",
    "conflict": "Conflict catalogues (Brecke)",
    "vital":    "Vital statistics (CamPop, HMD/HFD)",
    "prices":   "Prices & wages (Allen, Harper)",
}


def main() -> None:
    n = len(SOURCES)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    # Sort by start year ascending for a clean look
    sources = sorted(SOURCES, key=lambda r: r[1])

    y_positions = np.arange(n)
    for y, (label, start, end, cat, note) in zip(y_positions, sources):
        color = CATEGORY_COLORS[cat]
        ax.barh(y, end - start, left=start, color=color, alpha=0.85,
                edgecolor="#202020", linewidth=0.5, height=0.65)
        # Annotate start/end years inside the bar where space allows
        bar_len = end - start
        if bar_len > 250:
            ax.text(start + bar_len/2, y, f"{start}–{end}",
                    ha="center", va="center", fontsize=8.0,
                    color="white", weight="bold")
        else:
            ax.text(end + 25, y, f"{start}–{end}",
                    ha="left", va="center", fontsize=8.0,
                    color="#202020")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in sources], fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlabel("Calendar year (CE)")
    ax.set_xlim(-200, 2100)

    # Add vertical reference lines for the headline analysis windows
    ax.axvline(1421, color="#404040", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(1500, color="#404040", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(1750, color="#404040", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(1900, color="#404040", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(1950, color="#404040", linestyle=":", linewidth=0.8, alpha=0.6)
    # Annotate window labels at the top
    for x, txt in [(1421, "ModE-RA"), (1500, "Joint VAR"),
                    (1750, "HMD/HFD"), (1900, "Modern"), (1950, "ERA5")]:
        ax.text(x, -1.0, txt, rotation=90, ha="center", va="bottom",
                fontsize=7.5, color="#606060")

    # Legend by category
    handles = [mpatches.Patch(color=CATEGORY_COLORS[c],
                              label=CATEGORY_LABELS[c])
               for c in ["climate", "landuse", "vital", "volcanic",
                          "conflict", "prices"]]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8.5, framealpha=0.95, ncol=2, bbox_to_anchor=(1.0, -0.05))
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Temporal coverage of the data sources assembled into the paleo-economic panel",
                 y=1.005, x=0.04, ha="left", fontsize=11.5)
    plt.tight_layout()
    fig.savefig(FIG / "fig00_source_coverage.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig00_source_coverage.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG/'fig00_source_coverage.pdf'}")


if __name__ == "__main__":
    main()
