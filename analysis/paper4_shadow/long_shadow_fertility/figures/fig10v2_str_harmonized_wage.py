"""Fig 10v2 -- Pooled STR with HARMONIZED real-wage Z."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_smooth_transition import (
    fit_pooled_smooth_transition,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig10v2():
    panel = assemble_panel_multi()
    wages = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(wages, on=["iso3", "year"], how="left")
    sub = df.dropna(subset=["log_cbr", "t_growing", "log_real_wage"])
    fit = fit_pooled_smooth_transition(sub, y="log_cbr", x="t_growing", z="log_real_wage", theta_max=3.0)

    z_grid = np.linspace(sub["log_real_wage"].min(), sub["log_real_wage"].max(), 200)
    g = 1.0 / (1.0 + np.exp(-fit["theta"] * (z_grid - fit["c"])))
    beta = fit["beta_M"] * (1 - g) + fit["beta_T"] * g

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_grid, beta, color="darkgreen", lw=2.0,
             label=f"$\\beta(Z)$: $\\beta_M$={fit['beta_M']:+.3f}, $\\beta_T$={fit['beta_T']:+.3f}")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(fit["c"], color="gray", lw=0.8, ls=":", label=f"c={fit['c']:.2f}")
    ax.set_xlabel("log real wage (harmonized Allen$\\leftrightarrow$Maddison)")
    ax.set_ylabel(r"$\beta(Z)$, pooled elasticity")
    ax.set_title(f"Phase 5 STR with harmonized wage Z, N={fit['n']} across {fit['n_countries']} countries")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig10v2_str_harmonized_wage.pdf"
    png = FIG_DIR / "fig10v2_str_harmonized_wage.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, fit


if __name__ == "__main__":
    pdf, png, fit = make_fig10v2()
    print(f"wrote {pdf}; beta_M={fit['beta_M']:.4f}, beta_T={fit['beta_T']:.4f}, c={fit['c']:.2f}, theta={fit['theta']:.2f}")
