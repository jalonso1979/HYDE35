"""Fig 2 -- pooled smooth-transition climate-fertility elasticity (HEADLINE).

Pooled STR across all 4 countries with controls residualized out
of log_cbr and t_growing before fitting.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_smooth_transition import (
    fit_pooled_smooth_transition,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")

CONTROL_COLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]


def _residualize_on_controls(df: pd.DataFrame, target: str) -> pd.Series:
    sub = df.dropna(subset=[target] + CONTROL_COLS)
    if sub.empty:
        return pd.Series(dtype=float, index=df.index)
    dums = pd.get_dummies(sub["iso3"], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([sub[CONTROL_COLS], dums], axis=1))
    res = sm.OLS(sub[target].to_numpy(), X.to_numpy()).fit()
    out = pd.Series(np.nan, index=df.index)
    out.loc[sub.index] = sub[target].to_numpy() - res.fittedvalues
    return out


def make_fig2_pooled():
    df = assemble_panel_multi()
    df = df.dropna(subset=["log_cbr", "t_growing", "log_gdppc"])
    for c in CONTROL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    df["log_cbr_resid"] = _residualize_on_controls(df, "log_cbr")
    df["t_growing_resid"] = _residualize_on_controls(df, "t_growing")
    sub = df.dropna(subset=["log_cbr_resid", "t_growing_resid", "log_gdppc"])
    fit = fit_pooled_smooth_transition(sub, y="log_cbr_resid", x="t_growing_resid", z="log_gdppc")

    z_grid = np.linspace(sub["log_gdppc"].min(), sub["log_gdppc"].max(), 200)
    g = 1.0 / (1.0 + np.exp(-fit["theta"] * (z_grid - fit["c"])))
    beta = fit["beta_M"] * (1 - g) + fit["beta_T"] * g

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_grid, beta, color="firebrick", lw=2.0,
             label=f"$\\beta(Z)$: Malthus={fit['beta_M']:+.3f} (SE {fit['beta_M_se']:.3f}), "
                   f"Modern={fit['beta_T']:+.3f} (SE {fit['beta_T_se']:.3f})")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(fit["c"], color="gray", lw=0.8, ls=":", label=f"threshold c={fit['c']:.2f}")
    ax.set_xlabel("log Maddison GDP per capita")
    ax.set_ylabel(r"$\beta(Z)$, pooled elasticity (controls partialled out)")
    ax.set_title(f"Phase 2 pooled smooth-transition, N={fit['n']} country-years across {fit['n_countries']} countries")
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig2_pooled_smooth_transition.pdf"
    png = FIG_DIR / "fig2_pooled_smooth_transition.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, fit


if __name__ == "__main__":
    pdf, png, fit = make_fig2_pooled()
    print(f"wrote {pdf}; beta_M={fit['beta_M']:.4f}, beta_T={fit['beta_T']:.4f}, c={fit['c']:.2f}, theta={fit['theta']:.2f}")
