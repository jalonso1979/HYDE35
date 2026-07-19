"""Fig 15 — 3-equation SUR: log_cbr, log_cdr, log_real_wage on T_growing (Phase 5 harmonized wages)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
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
from analysis.paper4_shadow.long_shadow_fertility.estimators.triple_sur import (
    fit_triple_sur,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig15():
    panel = assemble_panel_multi()
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(mort, on=["iso3", "year"], how="left").merge(wage, on=["iso3", "year"], how="left")
    sub = df.dropna(subset=["log_cbr", "log_cdr", "log_real_wage", "t_growing"])
    res = fit_triple_sur(sub, ys=["log_cbr", "log_cdr", "log_real_wage"], x="t_growing")

    labels = ["log CBR (fertility)", "log CDR (mortality)", "log wage"]
    colors = ["steelblue", "firebrick", "darkgreen"]
    betas = [res["beta_y1"], res["beta_y2"], res["beta_y3"]]
    ses = [res["se_y1"], res["se_y2"], res["se_y3"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, lbl, b, s, c in zip(axes, labels, betas, ses, colors):
        ax.bar([0], [b], yerr=[1.96 * s], color=c, alpha=0.7, width=0.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xticks([])
        ax.set_title(f"{lbl}: β = {b:+.4f} (SE {s:.4f})")

    fig.suptitle(f"Phase 5 3-eq SUR — T_growing → {{fertility, mortality, wage}}  |  "
                  f"Wald p(F=M)={res['wald_eq12_pvalue']:.3g}, "
                  f"p(F=W)={res['wald_eq13_pvalue']:.3g}, "
                  f"p(M=W)={res['wald_eq23_pvalue']:.3g}",
                  fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig15_triple_sur.pdf"
    png = FIG_DIR / "fig15_triple_sur.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig15()
    print(f"wrote {pdf}; β_F={res['beta_y1']:.4f}, β_M={res['beta_y2']:.4f}, β_W={res['beta_y3']:.4f}")
