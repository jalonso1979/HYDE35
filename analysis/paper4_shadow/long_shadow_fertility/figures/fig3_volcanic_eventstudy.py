"""Figure 3: side-by-side volcanic event studies on log CBR.

Three panels: 1600 Huaynaputina (Malthusian), 1815 Tambora (transition),
1991 Pinatubo (modern). h=-1 is the reference year; trend partialled out.

Each event uses a +/-30-year window around the eruption to keep the linear
trend assumption credible over a manageable span.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel import (
    assemble_england_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.volcanic_event_study import (
    event_study_single_event,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")

ERUPTIONS_HEADLINE = [("Huaynaputina", 1600), ("Tambora", 1815), ("Pinatubo", 1991)]

# Window radius around each eruption for the estimation sample
WINDOW_RADIUS = 30  # years on each side


def make_fig3():
    df = assemble_england_panel()
    estimates: dict[str, pd.DataFrame] = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (name, yr) in zip(axes, ERUPTIONS_HEADLINE):
        # Slice to +/-30 years around the eruption -- keeps the linear trend
        # assumption credible. The event_study estimator no longer clips.
        sub = df.loc[df["year"].between(yr - WINDOW_RADIUS, yr + WINDOW_RADIUS)]
        res = event_study_single_event(sub, y="log_cbr", eruption_year=yr, pre=5, post=10)
        estimates[name] = res
        ax.fill_between(res["h"], res["ci_low"], res["ci_high"], alpha=0.25, color="steelblue")
        ax.plot(res["h"], res["delta"], marker="o", color="steelblue", lw=1.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{name} ({yr})")
        ax.set_xlabel("Years from eruption")
    axes[0].set_ylabel(r"$\Delta$ log CBR (h = -1 ref.)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig3_volcanic_eventstudy_england.pdf"
    png = FIG_DIR / "fig3_volcanic_eventstudy_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, est = make_fig3()
    print(f"wrote {pdf} and {png}")
