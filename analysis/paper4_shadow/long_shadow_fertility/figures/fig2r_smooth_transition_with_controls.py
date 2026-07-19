"""Fig 2-R — Phase 1 England STR with controls partialled out."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.smooth_transition import (
    fit_smooth_transition,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")

CONTROL_COLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]


def _residualize(df: pd.DataFrame, target: str) -> pd.Series:
    sub = df.dropna(subset=[target] + CONTROL_COLS)
    if sub.empty:
        return pd.Series(dtype=float, index=df.index)
    X = sm.add_constant(sub[CONTROL_COLS])
    res = sm.OLS(sub[target].to_numpy(), X.to_numpy()).fit()
    out = pd.Series(np.nan, index=df.index)
    out.loc[sub.index] = sub[target].to_numpy() - res.fittedvalues
    return out


def make_fig2r():
    df = assemble_panel_multi()
    eng = df.loc[df["iso3"] == "GBR"].copy()
    eng = eng.loc[eng["year"].between(1700, 2008)]
    for c in CONTROL_COLS:
        if c in eng.columns:
            eng[c] = eng[c].fillna(0)
    eng["log_cbr_resid"] = _residualize(eng, "log_cbr")
    eng["t_growing_resid"] = _residualize(eng, "t_growing")
    sub = eng.dropna(subset=["log_cbr_resid", "t_growing_resid", "log_gdppc"])

    fit = fit_smooth_transition(sub, y="log_cbr_resid", x="t_growing_resid", z="log_gdppc")
    z_grid = np.linspace(sub["log_gdppc"].min(), sub["log_gdppc"].max(), 200)
    g = 1.0 / (1.0 + np.exp(-fit["theta"] * (z_grid - fit["c"])))
    beta = fit["beta_M"] * (1 - g) + fit["beta_T"] * g
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_grid, beta, color="firebrick", lw=2.0,
             label=f"β(Z): β_M={fit['beta_M']:+.3f}, β_T={fit['beta_T']:+.3f}")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(fit["c"], color="gray", lw=0.8, ls=":", label=f"c={fit['c']:.2f}")
    ax.set_xlabel("log Maddison GDP per capita (England)")
    ax.set_ylabel("Elasticity β(Z), controls partialled out")
    ax.set_title("England STR with controls (Phase 1 retrospective)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig2r_smooth_transition_with_controls_england.pdf"
    png = FIG_DIR / "fig2r_smooth_transition_with_controls_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, fit


if __name__ == "__main__":
    pdf, png, fit = make_fig2r()
    print(f"wrote {pdf}; β_M={fit['beta_M']:.4f}, β_T={fit['beta_T']:.4f}")
