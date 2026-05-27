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

PANEL = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig22_uncertainty_diagnostic"
)

# The 7 Long Shadow countries
LS7 = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"]

COUNTRY_LABELS = {
    "GBR": "England",
    "FRA": "France",
    "ITA": "Italy",
    "SWE": "Sweden",
    "BEL": "Belgium",
    "NLD": "Netherlands",
    "ESP": "Spain",
}


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

    # --- Top panel: within-season realized SD (HEADLINE) ---
    for iso3, sub in df.groupby("iso3"):
        sub = sub.sort_values("year").dropna(subset=["t_anom_c_within_season_sd"])
        axes[0].plot(
            sub["year"],
            sub["t_anom_c_within_season_sd"],
            label=COUNTRY_LABELS.get(iso3, iso3),
            alpha=0.75,
            lw=0.9,
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
    axes[0].legend(ncol=7, loc="upper right", fontsize=7.5, framealpha=0.7)

    # --- Bottom panel: ModE-RA ensemble SD (COMPARATOR — data-density artifact) ---
    for iso3, sub in df.groupby("iso3"):
        sub = sub.sort_values("year").dropna(subset=["ensstd_t_growing"])
        axes[1].plot(
            sub["year"],
            sub["ensstd_t_growing"],
            label=COUNTRY_LABELS.get(iso3, iso3),
            alpha=0.75,
            lw=0.9,
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
    axes[1].legend(ncol=7, loc="upper right", fontsize=7.5, framealpha=0.7)

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


if __name__ == "__main__":
    make_fig22()
