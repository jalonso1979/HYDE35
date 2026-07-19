"""Fig 7 - Distributed-lag IRF on log_cbr, 4-panel by country.

For each country, runs a SINGLE-COUNTRY distributed-lag regression
(no country FE - there's only one unit). Uses HAC SEs with lag=k.
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

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ORDER = ["GBR", "FRA", "ITA", "SWE"]


def _single_country_dl(sub: pd.DataFrame, y: str, x: str, lags: int) -> pd.DataFrame:
    """Distributed-lag for a single country, no FE."""
    sub = sub.sort_values("year").copy()
    lag_cols = []
    for k in range(lags + 1):
        col = f"{x}_lag{k}"
        sub[col] = sub[x].shift(k)
        lag_cols.append(col)
    keep = sub[[y] + lag_cols].dropna()
    if len(keep) < 30:
        return pd.DataFrame()
    X = sm.add_constant(keep[lag_cols].astype(float))
    res = sm.OLS(keep[y].astype(float).to_numpy(), X.to_numpy()).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(1, lags)}
    )
    rows = []
    for k, col in enumerate(lag_cols):
        idx = 1 + k
        b = float(res.params[idx])
        s = float(res.bse[idx])
        rows.append({"lag": k, "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    # Cumulative
    lag_idx = np.arange(1, 1 + len(lag_cols))
    cum_b = float(res.params[lag_idx].sum())
    cov_block = res.cov_params()[np.ix_(lag_idx, lag_idx)]
    cum_se = float(np.sqrt(np.ones(len(lag_idx)) @ cov_block @ np.ones(len(lag_idx))))
    rows.append({"lag": "cumulative", "beta": cum_b, "se": cum_se,
                  "ci_low": cum_b - 1.96 * cum_se, "ci_high": cum_b + 1.96 * cum_se})
    return pd.DataFrame(rows)


def make_fig7_dl_irf(lags: int = 3):
    df = assemble_panel_multi()
    estimates: dict[str, pd.DataFrame] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, iso in zip(axes.flat, ORDER):
        sub = df.loc[df["iso3"] == iso, ["year", "log_cbr", "t_growing"]]
        est = _single_country_dl(sub, y="log_cbr", x="t_growing", lags=lags)
        estimates[iso] = est
        if est.empty:
            ax.set_title(f"{iso} (insufficient data)")
            continue
        per_lag = est.loc[est["lag"] != "cumulative"]
        cum = est.loc[est["lag"] == "cumulative"].iloc[0]
        ax.bar(per_lag["lag"].astype(int), per_lag["beta"],
                yerr=[per_lag["beta"] - per_lag["ci_low"], per_lag["ci_high"] - per_lag["beta"]],
                color="steelblue", alpha=0.7, label=r"$\beta_k$")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.errorbar([lags + 1.2], [cum["beta"]],
                     yerr=[[cum["beta"] - cum["ci_low"]], [cum["ci_high"] - cum["beta"]]],
                     fmt="o", color="firebrick", label=f"cum={cum['beta']:+.3f}")
        ax.set_title(iso)
        ax.set_xticks(list(range(lags + 1)) + [lags + 1.2])
        ax.set_xticklabels([f"k={k}" for k in range(lags + 1)] + ["cum"])
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Distributed-lag climate-fertility IRF (single-country, lags 0..{lags})", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig7_distributed_lag_irf.pdf"
    png = FIG_DIR / "fig7_distributed_lag_irf.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, _ = make_fig7_dl_irf()
    print(f"wrote {pdf}")
