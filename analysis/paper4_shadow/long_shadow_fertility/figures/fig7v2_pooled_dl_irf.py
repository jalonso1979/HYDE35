"""Fig 7v2 - Pooled DL IRF on log_cbr (country FE + year FE + controls + cluster SE)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_distributed_lag import (
    fit_pooled_distributed_lag,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
CONTROLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]


def make_fig7v2(lags: int = 3):
    df = assemble_panel_multi()
    sub = df.dropna(subset=["log_cbr", "t_growing"] + CONTROLS)
    est = fit_pooled_distributed_lag(sub, y="log_cbr", x="t_growing", lags=lags,
                                       controls=CONTROLS, year_fe=True)
    per_lag = est.loc[est["lag"] != "cumulative"]
    cum = est.loc[est["lag"] == "cumulative"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(per_lag["lag"].astype(int), per_lag["beta"],
            yerr=[per_lag["beta"] - per_lag["ci_low"], per_lag["ci_high"] - per_lag["beta"]],
            color="steelblue", alpha=0.7, label=r"$\beta_k$")
    ax.errorbar([lags + 1.2], [cum["beta"]],
                 yerr=[[cum["beta"] - cum["ci_low"]], [cum["ci_high"] - cum["beta"]]],
                 fmt="o", color="firebrick", label=f"cum = {cum['beta']:+.3f} (SE {cum['se']:.3f})")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks(list(range(lags + 1)) + [lags + 1.2])
    ax.set_xticklabels([f"k={k}" for k in range(lags + 1)] + ["cum"])
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel(r"$\beta_k$, log CBR on T_growing")
    ax.set_title("Pooled DL IRF (country FE + year FE + 9 controls + cluster SE), 4-country panel")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig7v2_pooled_dl_irf.pdf"
    png = FIG_DIR / "fig7v2_pooled_dl_irf.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, est


if __name__ == "__main__":
    pdf, png, est = make_fig7v2()
    print(f"wrote {pdf}; cum beta = {est.loc[est['lag']=='cumulative','beta'].iloc[0]:.4f}")
