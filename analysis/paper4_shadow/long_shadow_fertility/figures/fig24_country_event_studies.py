"""Fig 24 -- Country event studies aligned to country-specific log w = 9.97 crossing.

For each of the 7 countries in the balanced panel, we:
  1. Identify the first year where log_real_wage > 9.97 (event time = 0).
  2. Compute a rolling-window (15 yr) OLS elasticity of log_cbr on t_growing.
  3. Plot elasticity vs event time (event-time 0 = threshold crossing).

If the elasticity discontinuity appears around event time 0 in all panels, the
pooled Hansen threshold result is robust to country pooling.

Outputs
-------
/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig24_country_event_studies.pdf
/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig24_country_event_studies.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.event_study_threshold import (
    crossing_year,
    rolling_elasticity,
)
from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import (
    apply_pub_style,
    GRAYS,
    LINESTYLES,
)

# Apply publication style globally
apply_pub_style()

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
PAPER_FIG_DIR = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility/long_shadow/figures"
)
C_THRESHOLD = 9.97   # Phase 9 pooled Hansen estimate
ROLLING_WINDOW = 15  # years, centred


def make_fig24():
    # -----------------------------------------------------------------
    # 1. Build merged panel (fertility + climate + real wage)
    # -----------------------------------------------------------------
    panel = assemble_panel_multi()
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(wage, on=["iso3", "year"], how="left")

    countries = sorted(df["iso3"].unique())
    n_c = len(countries)

    # -----------------------------------------------------------------
    # 2. Grid: 2 rows x 4 cols (7 panels + 1 spare)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    flat_axes = axes.ravel()

    # Hide all panels by default; turn on as we fill them
    for ax in flat_axes:
        ax.set_visible(False)

    crossing_info = {}  # iso3 -> crossing year or None

    for idx, (ax, iso3) in enumerate(zip(flat_axes, countries)):
        ax.set_visible(True)

        # Per-country grayscale + linestyle (cycle through GRAYS/LINESTYLES)
        gray = GRAYS[idx % len(GRAYS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]

        sub = df[df["iso3"] == iso3].copy()

        # --- crossing year ---
        cross = crossing_year(sub, z="log_real_wage", c=C_THRESHOLD)
        crossing_info[iso3] = cross

        if cross is None:
            ax.text(0.5, 0.5, f"{iso3}\nnever crossed",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{iso3}: never crossed", fontsize=9)
            continue

        # --- rolling elasticity ---
        elas = rolling_elasticity(sub, y="log_cbr", x="t_growing",
                                  window=ROLLING_WINDOW)

        if elas.empty:
            ax.text(0.5, 0.5, f"{iso3}\ninsufficient data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{iso3}: no data", fontsize=9)
            continue

        elas = elas.copy()
        elas["event_time"] = elas["year"] - cross

        # --- plot (grayscale-safe: distinct gray + linestyle per country) ---
        ax.axhline(0, color="#aaaaaa", lw=0.7, linestyle=":")
        # Threshold-crossing marker: densely dashed dark gray
        ax.axvline(0, color="#333333", lw=1.2, linestyle=(0, (5, 1)),
                   label=f"crossing ({cross})")
        # CI shading in matching gray
        ax.fill_between(
            elas["event_time"],
            elas["beta"] - 1.96 * elas["se"],
            elas["beta"] + 1.96 * elas["se"],
            alpha=0.20, color=gray,
        )
        ax.plot(elas["event_time"], elas["beta"], color=gray, lw=1.4, linestyle=ls)

        ax.set_xlim(-60, 60)
        ax.set_title(f"{iso3}  (crossed {cross})", fontsize=9)
        ax.legend(loc="best", fontsize=7, framealpha=0.5)

    # Shared axis labels — bottom row visible panels
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("event time (years from threshold crossing)", fontsize=8)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("rolling climate elasticity\n(log CBR on growing-season T)", fontsize=8)

    fig.suptitle(
        "Fig 24 — Climate-fertility elasticity aligned to country-specific log $W$ = 9.97 crossing\n"
        r"(rolling 15-yr OLS; 95% CI shaded; red dashed = threshold-crossing year)",
        fontsize=10,
    )
    fig.tight_layout()

    # -----------------------------------------------------------------
    # 3. Save
    # -----------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / "fig24_country_event_studies.pdf"
    png_path = FIG_DIR / "fig24_country_event_studies.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")

    # -----------------------------------------------------------------
    # 4. Copy PDF to paper folder
    # -----------------------------------------------------------------
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    paper_pdf = PAPER_FIG_DIR / "fig24_country_event_studies.pdf"
    import shutil
    shutil.copy2(pdf_path, paper_pdf)
    print(f"Copied to {paper_pdf}")

    # -----------------------------------------------------------------
    # 5. Print crossing year summary
    # -----------------------------------------------------------------
    print("\nCrossing year summary (log W > 9.97):")
    for iso3, yr in crossing_info.items():
        print(f"  {iso3}: {yr if yr is not None else 'never crossed'}")

    return crossing_info


if __name__ == "__main__":
    make_fig24()
