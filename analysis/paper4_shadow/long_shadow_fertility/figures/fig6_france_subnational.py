"""Fig 6 - France dept rolling-window robustness."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.build_france_dept_annual import (
    build_france_dept_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_rolling_window import (
    pooled_rolling_elasticity,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig6_france_subnational(window: int = 40):
    df = build_france_dept_annual()
    # estimator expects 'iso3' as the unit identifier; drop the country-level
    # 'iso3' ("FRA") so the dept code can be renamed in without colliding.
    df = df.drop(columns=["iso3"]).rename(columns={"dep": "iso3"})
    est = pooled_rolling_elasticity(df, y="log_cbr", x="t_growing", window=window, min_obs=100)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(est["center_year"], est["ci_low"], est["ci_high"],
                     alpha=0.25, color="steelblue")
    ax.plot(est["center_year"], est["beta"], color="steelblue", lw=1.5)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Window center year (40-yr)")
    ax.set_ylabel(r"$\beta_w$")
    ax.set_title("France dept-level rolling-window elasticity (within-dept)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig6_france_subnational.pdf"
    png = FIG_DIR / "fig6_france_subnational.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, est


if __name__ == "__main__":
    pdf, png, _ = make_fig6_france_subnational()
    print(f"wrote {pdf}")
