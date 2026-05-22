"""Figure 2: smooth-transition function for climate-fertility elasticity,
with log Maddison GDPpc as transition variable Z, England 1700-2020."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel import (
    assemble_england_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.smooth_transition import (
    fit_smooth_transition,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig2():
    df = assemble_england_panel()
    sub = df.loc[df["year"].between(1700, 2008)].dropna(subset=["log_cbr", "t_growing", "log_gdppc"])
    fit = fit_smooth_transition(sub, y="log_cbr", x="t_growing", z="log_gdppc")
    z_grid = np.linspace(sub["log_gdppc"].min(), sub["log_gdppc"].max(), 200)
    g = 1.0 / (1.0 + np.exp(-fit["theta"] * (z_grid - fit["c"])))
    beta = fit["beta_M"] * (1 - g) + fit["beta_T"] * g
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_grid, beta, color="firebrick", lw=2.0,
             label=f"$\\beta(Z)$: Malthus={fit['beta_M']:.3f}, Modern={fit['beta_T']:.3f}")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(fit["c"], color="gray", lw=0.8, ls=":", label=f"threshold $c$={fit['c']:.2f}")
    ax.set_xlabel("log Maddison GDP per capita (England)")
    ax.set_ylabel("Elasticity $\\beta(Z)$ of log CBR on growing-season T anom.")
    ax.set_title("England smooth-transition climate-fertility elasticity")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig2_smooth_transition_england.pdf"
    png = FIG_DIR / "fig2_smooth_transition_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, fit


if __name__ == "__main__":
    pdf, png, fit = make_fig2()
    print(f"wrote {pdf} and {png}; beta_M={fit['beta_M']:.4f}, beta_T={fit['beta_T']:.4f}")
