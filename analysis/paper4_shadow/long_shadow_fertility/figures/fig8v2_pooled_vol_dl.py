"""Fig 8v2 - Pooled level + volatility DL (country FE + year FE + controls + cluster SE)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_volatility_dl import (
    fit_pooled_volatility_dl,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
# Controls - vol is the treatment, exclude vol_t_10y AND vol_p_10y from controls
CONTROLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought",
]


def make_fig8v2(lags: int = 3):
    df = assemble_panel_multi()
    sub = df.dropna(subset=["log_cbr", "t_growing", "vol_t_10y"] + CONTROLS)
    est = fit_pooled_volatility_dl(sub, y="log_cbr", x_level="t_growing", x_vol="vol_t_10y",
                                     lags=lags, controls=CONTROLS, year_fe=True)
    fig, (axL, axV) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, reg, color, ttl in [(axL, "level", "steelblue", "T level beta_k"),
                                  (axV, "vol", "firebrick", "T vol beta^V_k")]:
        e = est.loc[est["regressor"] == reg]
        ax.bar(e["lag"].astype(int), e["beta"],
                yerr=[e["beta"] - e["ci_low"], e["ci_high"] - e["beta"]],
                color=color, alpha=0.7)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(ttl)
        ax.set_xticks(list(range(lags + 1)))
        ax.set_xticklabels([f"k={k}" for k in range(lags + 1)])
    axL.set_ylabel(r"$\beta_k$, log CBR")
    fig.suptitle("Pooled level + volatility DL (FE + controls + cluster SE)", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig8v2_pooled_vol_dl.pdf"
    png = FIG_DIR / "fig8v2_pooled_vol_dl.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, est


if __name__ == "__main__":
    pdf, png, _ = make_fig8v2()
    print(f"wrote {pdf}")
