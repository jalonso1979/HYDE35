"""Figure 1: rolling-window OLS elasticity of log CBR on growing-season T,
England 1541-2020 (40-year window centered on year w)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel import (
    assemble_england_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.rolling_window import (
    rolling_elasticity,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig1(window: int = 40, return_estimates: bool = False):
    df = assemble_england_panel()
    est = rolling_elasticity(df, y="log_cbr", x="t_growing", window=window)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(est["center_year"], est["ci_low"], est["ci_high"],
                     alpha=0.25, color="steelblue", label="95% CI")
    ax.plot(est["center_year"], est["beta"], color="steelblue", lw=1.5,
             label="$\\beta_w$ (40-yr window)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    for v in (1600, 1641, 1815, 1883, 1991):
        ax.axvline(v, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Window center year (40-yr window)")
    ax.set_ylabel("Elasticity of log CBR on growing-season T anom.")
    ax.set_title("England rolling-window climate-fertility elasticity, 1541-2020")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig1_rolling_elasticity_england.pdf"
    png = FIG_DIR / "fig1_rolling_elasticity_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    if return_estimates:
        return pdf, png, est
    return pdf, png


if __name__ == "__main__":
    pdf, png = make_fig1()
    print(f"wrote {pdf} and {png}")
