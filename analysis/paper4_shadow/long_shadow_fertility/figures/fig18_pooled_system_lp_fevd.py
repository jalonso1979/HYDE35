"""Fig 18 — System LP-FEVD on pooled 7-country panel.

Four panels:
A. IRF of log CBR to a 1-sigma climate shock (h=0..15)
B. IRFs of log W and log CDR to the same climate shock (transmission)
C. FEVD stacked-area of log CBR variance shares by shock (T, W, M, F)
D. Comparison of FEVD with Cholesky [T,W,M,F] vs [T,F,W,M] -> wage-mediation share
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
HORIZONS = list(range(0, 16))


def _build_full_panel():
    panel = assemble_panel_multi()
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(mort, on=["iso3", "year"], how="left").merge(
        wage, on=["iso3", "year"], how="left"
    )
    df = df.rename(columns={
        "t_growing": "T", "log_real_wage": "W",
        "log_cdr": "M", "log_cbr": "F",
    })
    df = df.dropna(subset=["T", "W", "M", "F"]).copy()
    return df[["iso3", "year", "T", "W", "M", "F"]]


def make_fig18():
    df = _build_full_panel()

    base = fit_system_lp_fevd(df, variables=["T", "W", "M", "F"],
                                horizons=HORIZONS, p=2)
    alt = fit_system_lp_fevd(df, variables=["T", "F", "W", "M"],
                              horizons=HORIZONS, p=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: IRF F <- T
    ax = axes[0, 0]
    irf_F_T = np.array(base["irf"][("F", "T")])
    se_F_T = np.array(base["irf_se"][("F", "T")])
    ax.errorbar(HORIZONS, irf_F_T, yerr=1.96 * se_F_T,
                fmt="o-", color="steelblue", capsize=3)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel(r"$\partial \log\,\mathrm{CBR}/\partial u_T$")
    ax.set_title("A. IRF: log CBR response to 1-sigma climate shock", fontsize=10)

    # Panel B: IRFs W, M <- T
    ax = axes[0, 1]
    ax.plot(HORIZONS, base["irf"][("W", "T")],
            "o-", color="darkgreen", label="log wage")
    ax.plot(HORIZONS, base["irf"][("M", "T")],
            "s-", color="firebrick", label="log CDR")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Response to 1-sigma T shock")
    ax.set_title("B. Transmission IRFs (wage + mortality)", fontsize=10)
    ax.legend()

    # Panel C: FEVD stacked area for F
    ax = axes[1, 0]
    fevd_F = base["fevd"]["F"]  # shape (4 shocks, H)
    colors = ["steelblue", "darkgreen", "firebrick", "gold"]
    labels = ["T (climate)", "W (wage)", "M (mortality)", "F (own)"]
    ax.stackplot(HORIZONS, fevd_F, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Variance share")
    ax.set_ylim(0, 1)
    ax.set_title("C. FEVD of log CBR (Cholesky [T,W,M,F])", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    # Panel D: ordering comparison
    ax = axes[1, 1]
    fevd_F_alt = alt["fevd"]["F"]
    # alt ordering [T, F, W, M]: T=0, F=1, W=2, M=3
    t_base = fevd_F[0]
    t_alt = fevd_F_alt[0]
    w_base = fevd_F[1]
    w_alt = fevd_F_alt[2]
    ax.plot(HORIZONS, t_base, "o-", color="steelblue", label="T share (T,W,M,F)")
    ax.plot(HORIZONS, t_alt, "s--", color="navy", label="T share (T,F,W,M)")
    ax.plot(HORIZONS, w_base, "o-", color="darkgreen", label="W share (T,W,M,F)")
    ax.plot(HORIZONS, w_alt, "s--", color="forestgreen", label="W share (T,F,W,M)")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Variance share")
    ax.set_title("D. Mediation share via ordering swap", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Fig 18 — System LP-FEVD on pooled 7-country panel (Cholesky-identified)",
                  fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig18_pooled_system_lp_fevd.pdf"
    png = FIG_DIR / "fig18_pooled_system_lp_fevd.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    summary = {
        "irf_F_T": list(map(float, irf_F_T)),
        "fevd_F": {
            "T": [float(x) for x in fevd_F[0]],
            "W": [float(x) for x in fevd_F[1]],
            "M": [float(x) for x in fevd_F[2]],
            "F": [float(x) for x in fevd_F[3]],
        },
        "fevd_F_alt_ordering": {
            "T": [float(x) for x in fevd_F_alt[0]],
        },
    }
    return pdf, png, summary


if __name__ == "__main__":
    pdf, png, summary = make_fig18()
    print(f"wrote {pdf}")
    print(f"IRF F<-T at h=0: {summary['irf_F_T'][0]:+.4f}, h=15: {summary['irf_F_T'][-1]:+.4f}")
    print(f"FEVD share at h=15:  T={summary['fevd_F']['T'][-1]:.3f}, "
          f"W={summary['fevd_F']['W'][-1]:.3f}, "
          f"M={summary['fevd_F']['M'][-1]:.3f}, "
          f"F={summary['fevd_F']['F'][-1]:.3f}")
