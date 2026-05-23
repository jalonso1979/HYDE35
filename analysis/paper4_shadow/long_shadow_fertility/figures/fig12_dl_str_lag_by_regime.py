"""Fig 12 - Distributed-lag beta_k split by Malthus vs Modern STR regime."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.distributed_lag import (
    fit_distributed_lag,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
STR_THRESHOLD = 7.17  # Phase 2 STR threshold c on log_gdppc


def make_fig12_dl_regime(lags: int = 3):
    df = assemble_panel_multi()
    df = df.dropna(subset=["log_cbr", "t_growing", "log_gdppc", "iso3"])
    mal = df.loc[df["log_gdppc"] < STR_THRESHOLD]
    mod = df.loc[df["log_gdppc"] >= STR_THRESHOLD]
    est = {}
    for name, sub in [("Malthus", mal), ("Modern", mod)]:
        if sub["iso3"].nunique() < 2:
            est[name] = pd.DataFrame()
            continue
        est[name] = fit_distributed_lag(sub, y="log_cbr", x="t_growing", lags=lags, unit_col="iso3")

    fig, (axM, axT) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name, color in [(axM, "Malthus", "steelblue"), (axT, "Modern", "firebrick")]:
        e = est[name]
        if e.empty:
            ax.set_title(f"{name} (insufficient data)"); continue
        per_lag = e.loc[e["lag"] != "cumulative"]
        cum = e.loc[e["lag"] == "cumulative"].iloc[0]
        ax.bar(per_lag["lag"].astype(int), per_lag["beta"],
                yerr=[per_lag["beta"] - per_lag["ci_low"], per_lag["ci_high"] - per_lag["beta"]],
                color=color, alpha=0.7, label=r"$\beta_k$")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.errorbar([lags + 1.2], [cum["beta"]],
                     yerr=[[cum["beta"] - cum["ci_low"]], [cum["ci_high"] - cum["beta"]]],
                     fmt="o", color="black", label=f"cum={cum['beta']:+.3f}")
        ax.set_xticks(list(range(lags + 1)) + [lags + 1.2])
        ax.set_xticklabels([f"k={k}" for k in range(lags + 1)] + ["cum"])
        ax.set_title(f"{name} (log_gdppc {'<' if name == 'Malthus' else '>='} {STR_THRESHOLD})")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Distributed-lag fertility-T elasticity by STR regime (lags 0..{lags})", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig12_dl_str_lag_by_regime.pdf"
    png = FIG_DIR / "fig12_dl_str_lag_by_regime.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, est


if __name__ == "__main__":
    pdf, png, _ = make_fig12_dl_regime()
    print(f"wrote {pdf}")
