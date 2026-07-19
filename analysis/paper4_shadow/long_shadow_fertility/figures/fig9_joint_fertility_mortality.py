"""Fig 9 — Joint fertility + mortality SUR response to T (with cross-equation test)."""
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
from analysis.paper4_shadow.long_shadow_fertility.estimators.bivariate_sur import (
    fit_bivariate_sur,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig9_joint():
    panel = assemble_panel_multi()
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    df = panel.merge(mort, on=["iso3", "year"], how="left")
    sub = df.dropna(subset=["log_cbr", "log_cdr", "t_growing"])
    res = fit_bivariate_sur(sub, y1="log_cbr", y2="log_cdr", x="t_growing")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, label, beta, se, color in [
        (axL, "log CBR (fertility)", res["beta_y1"], res["se_y1"], "steelblue"),
        (axR, "log CDR (mortality)", res["beta_y2"], res["se_y2"], "firebrick"),
    ]:
        ax.bar([0], [beta], yerr=[1.96 * se], color=color, alpha=0.7, width=0.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xticks([]); ax.set_title(f"{label}: β = {beta:+.4f} (SE {se:.4f})")
    fig.suptitle(f"Joint fertility + mortality response to growing-season T  —  "
                 f"Wald test β_F = β_M: p = {res['wald_eq_pvalue']:.3g}",
                 fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig9_joint_fertility_mortality.pdf"
    png = FIG_DIR / "fig9_joint_fertility_mortality.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig9_joint()
    print(f"wrote {pdf}; β_F={res['beta_y1']:.4f}, β_M={res['beta_y2']:.4f}, p={res['wald_eq_pvalue']:.3g}")
