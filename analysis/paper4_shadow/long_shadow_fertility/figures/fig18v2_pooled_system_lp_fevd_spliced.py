"""Fig 18v2 — System LP-FEVD on pooled panel with spliced ModE-RA+ERA5 climate."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_climate_spliced import (
    build_country_climate_spliced,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
HORIZONS = list(range(0, 16))


def _build_panel_spliced():
    panel = assemble_panel_multi()
    spliced = build_country_climate_spliced(write=False)
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    panel_t = (panel.drop(columns=["t_growing"], errors="ignore")
                .merge(spliced[["iso3", "year", "t_growing"]], on=["iso3", "year"], how="left"))
    df = (panel_t.merge(mort, on=["iso3", "year"], how="left")
                 .merge(wage, on=["iso3", "year"], how="left"))
    df = df.rename(columns={"t_growing": "T", "log_real_wage": "W",
                              "log_cdr": "M", "log_cbr": "F"})
    df = df.dropna(subset=["T", "W", "M", "F"]).copy()
    return df[["iso3", "year", "T", "W", "M", "F"]]


def make_fig18v2():
    df = _build_panel_spliced()
    res = fit_system_lp_fevd(df, variables=["T", "W", "M", "F"], horizons=HORIZONS, p=2)
    fevd_F = res["fevd"]["F"]
    irf_F_T = np.array(res["irf"][("F", "T")])
    se_F_T = np.array(res["irf_se"][("F", "T")])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(HORIZONS, irf_F_T, yerr=1.96 * se_F_T,
                fmt="o-", color="purple", capsize=3)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel(r"$\partial \log\,\mathrm{CBR}/\partial u_T$")
    ax.set_title("A. IRF F<-T (ModE-RA+ERA5 spliced T)", fontsize=10)

    ax = axes[1]
    colors = ["steelblue", "darkgreen", "firebrick", "gold"]
    labels = ["T (climate)", "W (wage)", "M (mortality)", "F (own)"]
    ax.stackplot(HORIZONS, fevd_F, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Variance share")
    ax.set_ylim(0, 1)
    ax.set_title("B. FEVD of log CBR (spliced T)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Fig 18v2 — System LP-FEVD on pooled panel with spliced ModE-RA<1950 + ERA5 1950+",
                  fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig18v2_pooled_system_lp_fevd_spliced.pdf"
    png = FIG_DIR / "fig18v2_pooled_system_lp_fevd_spliced.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    summary = {
        "irf_F_T": list(map(float, irf_F_T)),
        "fevd_F_at_h15": {v: float(fevd_F[i][-1]) for i, v in enumerate(["T", "W", "M", "F"])},
        "n_obs": int(len(df)),
    }
    return pdf, png, summary


if __name__ == "__main__":
    pdf, png, summary = make_fig18v2()
    print(f"wrote {pdf}")
    print(f"IRF F<-T at h=15: {summary['irf_F_T'][-1]:+.4f}")
    print(f"FEVD shares at h=15: {summary['fevd_F_at_h15']}")
    print(f"N (panel rows): {summary['n_obs']}")
