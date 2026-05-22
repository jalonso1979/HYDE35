"""Fig 1 -- 4-panel rolling-window growing-season-T elasticity by country."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.rolling_window import (
    rolling_elasticity,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ERUPTIONS = (1600, 1641, 1815, 1883, 1991)
ORDER = ["GBR", "FRA", "ITA", "SWE"]
TITLES = {"GBR": "England (1541-2022)", "FRA": "France (1816-2022)",
          "ITA": "Italy (1862-2022)", "SWE": "Sweden (1749-2022)"}


def make_fig1_multi(window: int = 40):
    df = assemble_panel_multi()
    estimates = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, iso in zip(axes.flat, ORDER):
        sub = df.loc[df["iso3"] == iso]
        est = rolling_elasticity(sub, y="log_cbr", x="t_growing", window=window)
        estimates[iso] = est
        ax.fill_between(est["center_year"], est["ci_low"], est["ci_high"],
                         alpha=0.25, color="steelblue")
        ax.plot(est["center_year"], est["beta"], color="steelblue", lw=1.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        for v in ERUPTIONS:
            ax.axvline(v, color="gray", lw=0.4, alpha=0.5)
        ax.set_title(TITLES[iso], fontsize=11)
    axes[1, 0].set_xlabel("Window center year (40-yr)")
    axes[1, 1].set_xlabel("Window center year (40-yr)")
    axes[0, 0].set_ylabel(r"$\beta_w$")
    axes[1, 0].set_ylabel(r"$\beta_w$")
    fig.suptitle("Rolling-window climate-fertility elasticity, 4 countries", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig1_rolling_multi_country.pdf"
    png = FIG_DIR / "fig1_rolling_multi_country.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, _ = make_fig1_multi()
    print(f"wrote {pdf} and {png}")
