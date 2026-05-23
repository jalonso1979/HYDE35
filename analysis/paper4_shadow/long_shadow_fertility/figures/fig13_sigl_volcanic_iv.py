"""Fig 13 — Sigl volcanic-IV 2SLS estimate of log_cbr ~ T_growing using log_vssi
(and 1-2 yr lags) as exogenous instruments. Country FE only (no year FE — V is
constant within year, would be absorbed)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_sigl_volcanic_panel import (
    build_sigl_volcanic_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.iv_2sls import (
    fit_iv_2sls,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
CONTROLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]


def _add_lags(df: pd.DataFrame, var: str, lags: int, unit_col: str = "iso3") -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values([unit_col, "year"]).copy()
    cols = []
    for k in range(lags + 1):
        c = f"{var}_lag{k}"
        df[c] = df.groupby(unit_col)[var].shift(k)
        cols.append(c)
    return df, cols


def make_fig13():
    panel = assemble_panel_multi()
    sigl = build_sigl_volcanic_panel()[["iso3", "year", "log_vssi"]]
    df = panel.merge(sigl, on=["iso3", "year"], how="left")
    df, vssi_lags = _add_lags(df, "log_vssi", lags=2)
    df = df.dropna(subset=["log_cbr", "t_growing"] + vssi_lags + CONTROLS)

    res = fit_iv_2sls(df, y="log_cbr", x="t_growing",
                       instruments=vssi_lags, controls=CONTROLS)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    sample = df.sample(min(500, len(df)), random_state=0)
    axL.scatter(sample["log_vssi_lag0"], sample["t_growing"], alpha=0.3, s=8, color="steelblue")
    axL.set_xlabel("log VSSI (Tg S, k=0)")
    axL.set_ylabel("T_growing (anomaly)")
    axL.set_title(f"First-stage scatter (first-stage F={res['first_stage_f']:.1f})")
    axR.bar([0], [res["beta"]], yerr=[1.96 * res["se"]], color="firebrick", alpha=0.7, width=0.5)
    axR.axhline(0, color="black", lw=0.6, ls="--")
    axR.set_xticks([0]); axR.set_xticklabels([r"2SLS $\beta$"])
    axR.set_title(f"2SLS β = {res['beta']:+.4f} (SE {res['se']:.4f})  |  AR p = {res['ar_pvalue']:.3g}")
    fig.suptitle(f"Sigl volcanic-IV 2SLS  —  N = {res['n']}", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig13_sigl_volcanic_iv.pdf"
    png = FIG_DIR / "fig13_sigl_volcanic_iv.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig13()
    print(f"wrote {pdf}; β={res['beta']:.4f}, first-stage F={res['first_stage_f']:.1f}")
