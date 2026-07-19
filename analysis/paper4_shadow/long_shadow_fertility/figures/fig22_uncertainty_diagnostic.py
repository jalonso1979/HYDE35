"""Fig 22 -- Uncertainty diagnostic: within-season realized SD vs ModE-RA ensemble SD.

Two panels:
  Top:    Within-growing-season realized SD of monthly temperature anomaly (HEADLINE).
          Expected: roughly trendless — reflects true climate variability.
  Bottom: ModE-RA ensemble SD of growing-season temperature anomaly (COMPARATOR).
          Expected: clear secular decline driven by increasing paleo observation density.

This figure motivates using within-season SD as the headline uncertainty measure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import (
    apply_pub_style,
    style_lines,
)

# Apply publication style globally
apply_pub_style()

PANEL = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig22_uncertainty_diagnostic"
)

# The 12 Long Shadow countries
LS7 = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP", "NOR", "DNK", "FIN", "ISL", "CHE"]

COUNTRY_LABELS = {
    "GBR": "England",
    "FRA": "France",
    "ITA": "Italy",
    "SWE": "Sweden",
    "BEL": "Belgium",
    "NLD": "Netherlands",
    "ESP": "Spain",
    "NOR": "Norway",
    "DNK": "Denmark",
    "FIN": "Finland",
    "ISL": "Iceland",
    "CHE": "Switzerland",
}


ERUPTIONS = [
    (1600, "Huaynaputina"),
    (1660, "Long Island"),
    (1783, "Laki"),
    (1815, "Tambora"),
    (1835, "Coseguina"),
    (1883, "Krakatoa"),
    (1902, "Santa María"),
    (1963, "Agung"),
    (1991, "Pinatubo"),
]


def make_fig22() -> None:
    df = pd.read_parquet(PANEL)

    # Confirm columns exist
    cols_needed = ["t_anom_c_within_season_sd", "ensstd_t_growing"]
    for c in cols_needed:
        if c not in df.columns:
            raise RuntimeError(
                f"Missing required column: {c}; available: {df.columns.tolist()}"
            )

    df = df[df["iso3"].isin(LS7)].copy()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    # Apply grayscale + linestyle cycler to both panels (12 countries)
    style_lines(axes[0], n=12)
    style_lines(axes[1], n=12)

    # --- Top panel: within-season realized SD (HEADLINE) ---
    for iso3, sub in df.groupby("iso3"):
        sub = sub.sort_values("year").dropna(subset=["t_anom_c_within_season_sd"])
        axes[0].plot(
            sub["year"],
            sub["t_anom_c_within_season_sd"],
            label=COUNTRY_LABELS.get(iso3, iso3),
            alpha=0.85,
            lw=1.1,
        )

    # --- Eruption reference lines (top panel): thin gray dotted vertical lines ---
    first = True
    for ev_year, ev_name in ERUPTIONS:
        if first:
            axes[0].axvline(
                ev_year,
                color="#999999",
                lw=0.7,
                alpha=0.6,
                linestyle=":",
                zorder=0,
                label="Major eruptions",
            )
            first = False
        else:
            axes[0].axvline(
                ev_year,
                color="#999999",
                lw=0.7,
                alpha=0.6,
                linestyle=":",
                zorder=0,
            )

    axes[0].set_title(
        "Headline: within-growing-season realized SD of monthly temperature anomaly",
        fontsize=10,
    )
    axes[0].set_ylabel(r"SD (${}^\circ$C)")
    axes[0].annotate(
        "Trendless: reflects true climate variability",
        xy=(0.02, 0.93),
        xycoords="axes fraction",
        fontsize=8,
        color="dimgray",
        va="top",
    )
    axes[0].legend(ncol=6, loc="upper right", fontsize=7, framealpha=0.7)

    # --- Bottom panel: ModE-RA ensemble SD (COMPARATOR — data-density artifact) ---
    for iso3, sub in df.groupby("iso3"):
        sub = sub.sort_values("year").dropna(subset=["ensstd_t_growing"])
        axes[1].plot(
            sub["year"],
            sub["ensstd_t_growing"],
            label=COUNTRY_LABELS.get(iso3, iso3),
            alpha=0.85,
            lw=1.1,
        )

    # --- Eruption reference lines (bottom panel): thin gray dotted vertical lines ---
    for ev_year, ev_name in ERUPTIONS:
        axes[1].axvline(
            ev_year,
            color="#999999",
            lw=0.7,
            alpha=0.6,
            linestyle=":",
            zorder=0,
        )

    axes[1].set_title(
        "Comparator: ModE-RA ensemble SD of growing-season temperature anomaly",
        fontsize=10,
    )
    axes[1].set_ylabel(r"SD (${}^\circ$C)")
    axes[1].set_xlabel("Year")
    axes[1].annotate(
        "Secular decline: data-assimilation density artifact (not used as headline)",
        xy=(0.02, 0.93),
        xycoords="axes fraction",
        fontsize=8,
        color="dimgray",
        va="top",
    )
    axes[1].legend(ncol=6, loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle(
        "Two measures of climate uncertainty: realized vs paleo-ensemble",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")

    # Copy PDF to paper repo figures/
    import shutil
    copy_pdf = Path(
        "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
        "/My Drive/Fertility/long_shadow/figures/fig22_uncertainty_diagnostic.pdf"
    )
    copy_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f"{OUT}.pdf", str(copy_pdf))
    print(f"Copied PDF to {copy_pdf}")


if __name__ == "__main__":
    make_fig22()
