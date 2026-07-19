"""Figure 2: smooth-transition function for climate-fertility elasticity
(left panel) and raw within-era OLS scatter (right panel), England 1700-2008.

The right-panel within-era OLS is a robust complement to the STR fit when
theta pegs at its upper bound — it shows the slope magnitudes implied by
the data inside each era without any logistic-shape assumption.
"""
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

MALTHUS_YEARS = (1700, 1837)
MODERN_YEARS = (1938, 2008)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) for y on x via numpy.polyfit (degree=1)."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def make_fig2():
    df = assemble_england_panel()
    sub = df.loc[df["year"].between(1700, 2008)].dropna(subset=["log_cbr", "t_growing", "log_gdppc"])
    fit = fit_smooth_transition(sub, y="log_cbr", x="t_growing", z="log_gdppc")
    z_grid = np.linspace(sub["log_gdppc"].min(), sub["log_gdppc"].max(), 200)
    g = 1.0 / (1.0 + np.exp(-fit["theta"] * (z_grid - fit["c"])))
    beta = fit["beta_M"] * (1 - g) + fit["beta_T"] * g

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: STR beta(Z)
    axL.plot(z_grid, beta, color="firebrick", lw=2.0,
             label=f"$\\beta(Z)$: Malthus={fit['beta_M']:.3f}, Modern={fit['beta_T']:.3f}")
    axL.axhline(0, color="black", lw=0.6, ls="--")
    axL.axvline(fit["c"], color="gray", lw=0.8, ls=":", label=f"threshold $c$={fit['c']:.2f}")
    axL.set_xlabel("log Maddison GDP per capita (England)")
    axL.set_ylabel("Elasticity $\\beta(Z)$ of log CBR on growing-season T anom.")
    axL.set_title("Smooth-transition $\\beta(Z)$")
    axL.legend(loc="best", frameon=False)

    # Right panel: raw within-era scatter + OLS lines
    mal_mask = sub["year"].between(*MALTHUS_YEARS)
    mod_mask = sub["year"].between(*MODERN_YEARS)
    mal = sub.loc[mal_mask]
    mod = sub.loc[mod_mask]

    bM, aM = _ols_slope(mal["t_growing"].to_numpy(), mal["log_cbr"].to_numpy())
    bT, aT = _ols_slope(mod["t_growing"].to_numpy(), mod["log_cbr"].to_numpy())

    axR.scatter(mal["t_growing"], mal["log_cbr"], s=12, alpha=0.55,
                color="firebrick",
                label=f"Malthus {MALTHUS_YEARS[0]}-{MALTHUS_YEARS[1]} "
                      f"(N={len(mal)}, $\\beta_M$={bM:+.3f})")
    axR.scatter(mod["t_growing"], mod["log_cbr"], s=12, alpha=0.55,
                color="steelblue",
                label=f"Modern {MODERN_YEARS[0]}-{MODERN_YEARS[1]} "
                      f"(N={len(mod)}, $\\beta_T$={bT:+.3f})")
    # OLS fit lines spanning each era's t_growing range
    if len(mal) > 1:
        xg = np.linspace(mal["t_growing"].min(), mal["t_growing"].max(), 50)
        axR.plot(xg, aM + bM * xg, color="firebrick", lw=2.0)
    if len(mod) > 1:
        xg = np.linspace(mod["t_growing"].min(), mod["t_growing"].max(), 50)
        axR.plot(xg, aT + bT * xg, color="steelblue", lw=2.0)
    axR.set_xlabel("Growing-season temperature anomaly (deg C)")
    axR.set_ylabel("log CBR (England)")
    axR.set_title("Raw within-era OLS")
    axR.legend(loc="best", frameon=False, fontsize=8)

    fig.suptitle("England smooth-transition (left) vs raw within-era OLS (right)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig2_smooth_transition_england.pdf"
    png = FIG_DIR / "fig2_smooth_transition_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    out_fit = dict(fit)
    out_fit["beta_M_within_era"] = bM
    out_fit["beta_T_within_era"] = bT
    out_fit["n_malthus"] = int(len(mal))
    out_fit["n_modern"] = int(len(mod))
    plt.close(fig)
    return pdf, png, out_fit


if __name__ == "__main__":
    pdf, png, fit = make_fig2()
    print(
        f"wrote {pdf} and {png}; "
        f"STR beta_M={fit['beta_M']:.4f}, beta_T={fit['beta_T']:.4f}, "
        f"theta={fit['theta']:.3f}, c={fit['c']:.3f}; "
        f"within-era beta_M={fit['beta_M_within_era']:+.4f} (N={fit['n_malthus']}), "
        f"beta_T={fit['beta_T_within_era']:+.4f} (N={fit['n_modern']})"
    )
