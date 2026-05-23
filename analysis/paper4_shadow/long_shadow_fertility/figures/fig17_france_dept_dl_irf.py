"""Fig 17 — France dept-year pooled DL IRF.

Four panels:
- A: Spec 1 (dept FE only, no year FE). Cumulative beta across h=0..5 with 95% CI.
- B: Spec 2 (dept + year FE). Cumulative beta across h=0..5 with 95% CI.
- C: Precipitation IRF for both specs.
- D: Cross-country beta_F from Fig 7v3 (read from pooled DL output) overlaid on
     Spec 1 dept-level beta_F for visual comparison.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

from analysis.paper4_shadow.long_shadow_fertility.data.build_france_dept_annual import (
    build_france_dept_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_dl_france_dept import (
    fit_dept_distributed_lag,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_distributed_lag import (
    fit_pooled_distributed_lag,
)
from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
LAGS = 5


def _lag_irf(df, x, controls, year_fe):
    res = fit_dept_distributed_lag(
        df, y="log_cbr", x=x, lags=LAGS, controls=controls, year_fe=year_fe,
    )
    lag_rows = res[res["lag"] != "cumulative"].copy()
    lag_rows["lag"] = lag_rows["lag"].astype(int)
    lag_rows = lag_rows.sort_values("lag")
    return lag_rows


def make_fig17():
    panel = build_france_dept_annual()
    sub = panel.dropna(subset=["log_cbr", "t_growing", "p_growing"]).copy()

    spec1_t = _lag_irf(sub, x="t_growing", controls=["p_growing"], year_fe=False)
    spec2_t = _lag_irf(sub, x="t_growing", controls=["p_growing"], year_fe=True)
    spec1_p = _lag_irf(sub, x="p_growing", controls=["t_growing"], year_fe=False)
    spec2_p = _lag_irf(sub, x="p_growing", controls=["t_growing"], year_fe=True)

    # Cross-country beta_F via Fig 7v3 pipeline (recomputed on the fly)
    pooled = assemble_panel_multi()
    pooled_sub = pooled.dropna(subset=["log_cbr", "t_growing", "p_growing"]).copy()
    cross = fit_pooled_distributed_lag(
        pooled_sub, y="log_cbr", x="t_growing", lags=LAGS,
        unit_col="iso3", controls=["p_growing"], year_fe=True,
    )
    cross_lag = cross[cross["lag"] != "cumulative"].copy()
    cross_lag["lag"] = cross_lag["lag"].astype(int)
    cross_lag = cross_lag.sort_values("lag")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def _panel(ax, df, title, color="steelblue"):
        ax.errorbar(df["lag"], df["beta"], yerr=1.96 * df["se"],
                    fmt="o-", color=color, capsize=3)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xlabel("Lag (years)")
        ax.set_ylabel(r"$\beta$")
        ax.set_title(title, fontsize=10)

    _panel(axes[0, 0], spec1_t,
            "A. Spec 1 (dept FE) — T -> log CBR", color="steelblue")
    _panel(axes[0, 1], spec2_t,
            "B. Spec 2 (dept + year FE) — T -> log CBR", color="navy")
    _panel(axes[1, 0], spec1_p,
            "C. Spec 1+2 — P -> log CBR (precip)", color="darkgreen")
    _panel(axes[1, 0], spec2_p, "", color="forestgreen")

    # Panel D — overlay
    ax = axes[1, 1]
    ax.errorbar(spec1_t["lag"], spec1_t["beta"], yerr=1.96 * spec1_t["se"],
                fmt="o-", color="steelblue", capsize=3, label="France dept Spec 1")
    ax.errorbar(cross_lag["lag"], cross_lag["beta"], yerr=1.96 * cross_lag["se"],
                fmt="s-", color="firebrick", capsize=3, label="Cross-country (Fig 7v3)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.legend(fontsize=8)
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"D. France dept vs. cross-country $\beta_F$", fontsize=10)

    fig.suptitle("Fig 17 — France dept-year distributed-lag IRF (Cassini 1851-1897)",
                  fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig17_france_dept_dl_irf.pdf"
    png = FIG_DIR / "fig17_france_dept_dl_irf.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    summary = {
        "spec1": {"beta": spec1_t["beta"].tolist(), "se": spec1_t["se"].tolist()},
        "spec2": {"beta": spec2_t["beta"].tolist(), "se": spec2_t["se"].tolist()},
        "cross": {"beta": cross_lag["beta"].tolist(), "se": cross_lag["se"].tolist()},
    }
    return pdf, png, summary


if __name__ == "__main__":
    pdf, png, summary = make_fig17()
    print(f"wrote {pdf}")
    print(f"Spec 1 beta_h=0 = {summary['spec1']['beta'][0]:+.4f}")
    print(f"Spec 2 beta_h=0 = {summary['spec2']['beta'][0]:+.4f}")
    print(f"Cross-country beta_h=0 = {summary['cross']['beta'][0]:+.4f}")
