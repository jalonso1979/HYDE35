"""Fig 5 - EFP cross-section pooled STR (stub; activates when EFP data arrives)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
EFP = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "efp_province_decade.parquet")


def make_fig5_efp():
    from analysis.paper4_shadow.long_shadow_fertility.estimators.efp_cross_section import (
        fit_efp_str,
    )
    df = pd.read_parquet(EFP)
    sub = df.dropna(subset=["If", "t_growing", "log_gdppc"])
    sub["log_If"] = np.log(sub["If"])
    fit = fit_efp_str(sub, y="log_If", x="t_growing", z="log_gdppc")

    z_grid = np.linspace(sub["log_gdppc"].min(), sub["log_gdppc"].max(), 200)
    g = 1.0 / (1.0 + np.exp(-fit["theta"] * (z_grid - fit["c"])))
    beta = fit["beta_M"] * (1 - g) + fit["beta_T"] * g

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_grid, beta, color="seagreen", lw=2.0,
             label=f"beta(Z): Malthus={fit['beta_M']:+.3f}, Modern={fit['beta_T']:+.3f}")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(fit["c"], color="gray", lw=0.8, ls=":", label=f"c={fit['c']:.2f}")
    ax.set_xlabel("log GDP per capita")
    ax.set_ylabel(r"$\beta(Z)$, EFP If on $T_{growing}$")
    ax.set_title(f"EFP cross-section pooled STR, N={fit['n']} province-decades")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig5_efp_cross_section.pdf"
    png = FIG_DIR / "fig5_efp_cross_section.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, fit


if __name__ == "__main__":
    pdf, png, fit = make_fig5_efp()
    print(f"wrote {pdf}")
