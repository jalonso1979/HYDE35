"""Fig 3 -- multi-country stacked volcanic event studies (1815/1883/1991)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.stacked_event_study import (
    stacked_event_study,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ERUPTIONS = [("Tambora", 1815), ("Krakatoa", 1883), ("Pinatubo", 1991)]
WINDOW_RADIUS = 30


def make_fig3_stacked():
    df = assemble_panel_multi()
    estimates = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (name, yr) in zip(axes, ERUPTIONS):
        sub = df.loc[df["year"].between(yr - WINDOW_RADIUS, yr + WINDOW_RADIUS)]
        res = stacked_event_study(sub, y="log_cbr", eruption_year=yr, pre=5, post=10)
        estimates[name] = res
        ax.fill_between(res["h"], res["ci_low"], res["ci_high"],
                         alpha=0.25, color="steelblue")
        ax.plot(res["h"], res["delta"], marker="o", color="steelblue", lw=1.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{name} ({yr})")
        ax.set_xlabel("Years from eruption")
    axes[0].set_ylabel(r"$\Delta$ log CBR (h = -1 ref.)")
    fig.suptitle("Multi-country stacked volcanic event studies", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig3_stacked_volcanic.pdf"
    png = FIG_DIR / "fig3_stacked_volcanic.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, est = make_fig3_stacked()
    print(f"wrote {pdf} and {png}")
