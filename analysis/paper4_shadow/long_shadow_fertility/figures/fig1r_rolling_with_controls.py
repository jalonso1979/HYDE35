"""Fig 1-R — Phase 1 England rolling-window with controls partialled out."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.rolling_window import (
    rolling_elasticity,
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


def make_fig1r(window: int = 40):
    df = assemble_panel_multi()
    eng = df.loc[df["iso3"] == "GBR"].copy()
    for c in CONTROL_COLS:
        if c in eng.columns:
            eng[c] = eng[c].fillna(0)
    eng["log_cbr_resid"] = _residualize(eng, "log_cbr")
    eng["t_growing_resid"] = _residualize(eng, "t_growing")
    est = rolling_elasticity(eng, y="log_cbr_resid", x="t_growing_resid", window=window)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(est["center_year"], est["ci_low"], est["ci_high"],
                     alpha=0.25, color="firebrick", label="With controls 95% CI")
    ax.plot(est["center_year"], est["beta"], color="firebrick", lw=1.5,
             label="β (controls partialled out)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    for v in (1600, 1641, 1815, 1883, 1991):
        ax.axvline(v, color="gray", lw=0.4, alpha=0.5)
    ax.set_xlabel("Window center year (40-yr)")
    ax.set_ylabel("Elasticity, partialled out")
    ax.set_title("England rolling-window WITH controls (Phase 1 retrospective)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig1r_rolling_with_controls_england.pdf"
    png = FIG_DIR / "fig1r_rolling_with_controls_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, est


if __name__ == "__main__":
    pdf, png, _ = make_fig1r()
    print(f"wrote {pdf}")
