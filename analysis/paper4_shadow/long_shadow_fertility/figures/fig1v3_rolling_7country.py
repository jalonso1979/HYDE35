"""Fig 1v3 -- 7-panel rolling-window IRF (gap-filled England + 6 other countries)."""
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
ORDER = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"]


def make_fig1v3(window: int = 40):
    df = assemble_panel_multi()
    estimates: dict[str, pd.DataFrame] = {}
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)
    for ax, iso in zip(axes.flat, ORDER):
        sub = df.loc[df["iso3"] == iso]
        est = rolling_elasticity(sub, y="log_cbr", x="t_growing", window=window)
        estimates[iso] = est
        ax.fill_between(est["center_year"], est["ci_low"], est["ci_high"],
                         alpha=0.25, color="steelblue")
        ax.plot(est["center_year"], est["beta"], color="steelblue", lw=1.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        for v in (1600, 1641, 1815, 1883, 1991):
            ax.axvline(v, color="gray", lw=0.4, alpha=0.5)
        ax.set_title(iso)
    for j in (7, 8):
        axes.flat[j].axis("off")
    fig.suptitle("Phase 5 rolling-window IRF -- 7 countries (gap-filled England via HMD)", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig1v3_rolling_7country.pdf"
    png = FIG_DIR / "fig1v3_rolling_7country.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, _ = make_fig1v3()
    print(f"wrote {pdf}")
