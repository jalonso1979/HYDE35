"""Fig 19 — France dept system LP IRF with year FE.

Overlays:
- Within-country LP β_h^{within} from year-FE-absorbed identification.
- Fig 17 Spec 2 (dept + year FE) DL β_h for cross-validation that LP and DL
  agree on the same identifying variation.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

from analysis.paper4_shadow.long_shadow_fertility.data.build_france_dept_annual import (
    build_france_dept_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_france_dept import (
    fit_dept_local_projection,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_dl_france_dept import (
    fit_dept_distributed_lag,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
HORIZONS = range(0, 6)


def make_fig19():
    panel = build_france_dept_annual()
    sub = panel.dropna(subset=["log_cbr", "t_growing", "p_growing"]).copy()

    lp_df = fit_dept_local_projection(sub, y="log_cbr", shock="t_growing",
                                        horizons=HORIZONS)

    # DL Spec 2 for overlay
    dl = fit_dept_distributed_lag(sub, y="log_cbr", x="t_growing", lags=5,
                                    controls=["p_growing"], year_fe=True)
    dl_lag = dl[dl["lag"] != "cumulative"].copy()
    dl_lag["lag"] = dl_lag["lag"].astype(int)
    dl_lag = dl_lag.sort_values("lag")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(lp_df["h"], lp_df["beta"], yerr=1.96 * lp_df["se"],
                fmt="o-", color="steelblue", capsize=3, label="LP (year FE)")
    ax.errorbar(dl_lag["lag"], dl_lag["beta"], yerr=1.96 * dl_lag["se"],
                fmt="s--", color="firebrick", capsize=3, label="DL Spec 2 (year FE)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Horizon / lag (years)")
    ax.set_ylabel(r"$\beta_h^{within}$")
    ax.set_title("Fig 19 — France dept system LP IRF (within-country, year-FE identified)",
                  fontsize=11)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig19_france_dept_system_lp.pdf"
    png = FIG_DIR / "fig19_france_dept_system_lp.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png, lp_df


if __name__ == "__main__":
    pdf, png, lp_df = make_fig19()
    print(f"wrote {pdf}")
    print(lp_df.to_string(index=False))
