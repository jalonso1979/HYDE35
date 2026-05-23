"""Fig 10v3 -- Hansen 1996/2000 threshold regression on the 7-country pooled panel.

Two panels:
A. (z, y_residual) scatter with vertical c_hat line + shaded 95% CI + two regression lines
B. LR path across candidate thresholds with the 95% LR cutoff (7.35) marked
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression import (
    fit_threshold_regression,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
N_BOOT = 500


def make_fig10v3():
    panel = assemble_panel_multi()
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(wage, on=["iso3", "year"], how="left").dropna(
        subset=["log_cbr", "t_growing", "log_real_wage"]
    )

    res = fit_threshold_regression(
        df, y="log_cbr", x="t_growing", z="log_real_wage",
        n_boot=N_BOOT, seed=0,
    )

    # Residualize log_cbr by country FE for the scatter
    dums = pd.get_dummies(df["iso3"], drop_first=True, dtype=float)
    Xfe = pd.concat([pd.Series(1.0, index=df.index, name="const"), dums], axis=1).to_numpy()
    y_vec = df["log_cbr"].astype(float).to_numpy()
    coefs = np.linalg.lstsq(Xfe, y_vec, rcond=None)[0]
    resid = y_vec - Xfe @ coefs

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A
    ax = axes[0]
    ax.scatter(df["log_real_wage"], resid, alpha=0.3, s=10, color="grey")
    c_hat = res["c_hat"]
    ax.axvline(c_hat, color="firebrick", lw=1.5, label=f"$\\hat{{c}}={c_hat:.2f}$")
    ax.axvspan(res["c_ci_lo"], res["c_ci_hi"], color="firebrick", alpha=0.15,
                label=f"95% CI [{res['c_ci_lo']:.2f}, {res['c_ci_hi']:.2f}]")
    z_lo = np.linspace(df["log_real_wage"].min(), c_hat, 50)
    z_hi = np.linspace(c_hat, df["log_real_wage"].max(), 50)
    t_ref = 1.0
    ax.plot(z_lo, [res["beta_M"] * t_ref] * len(z_lo),
            color="steelblue", lw=2,
            label=f"$\\beta_M={res['beta_M']:+.3f}$ (low-wage regime)")
    ax.plot(z_hi, [res["beta_T"] * t_ref] * len(z_hi),
            color="darkorange", lw=2,
            label=f"$\\beta_T={res['beta_T']:+.3f}$ (high-wage regime)")
    ax.set_xlabel(r"$\log W$ (harmonized real wage)")
    ax.set_ylabel(r"$\log\,\mathrm{CBR}$ (country-FE residual)")
    ax.set_title(f"A. Threshold regression (sup-Wald p={res['sup_wald_pvalue']:.3f}, N={res['n']})", fontsize=10)
    ax.legend(fontsize=8, loc="best")

    # Panel B
    ax = axes[1]
    lr_path = res["lr_path"]
    cs = [p[0] for p in lr_path]
    lrs = [p[1] for p in lr_path]
    ax.plot(cs, lrs, "o-", color="steelblue", markersize=4)
    ax.axhline(7.35, color="firebrick", lw=1.0, ls="--", label="95% CI cutoff (LR=7.35)")
    ax.axvline(c_hat, color="firebrick", lw=1.5, label=f"$\\hat{{c}}={c_hat:.2f}$")
    ax.set_xlabel(r"Candidate threshold $c$ in $\log W$")
    ax.set_ylabel(r"LR statistic")
    ax.set_title("B. LR path across candidate thresholds", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Fig 10v3 -- Hansen (1996, 2000) threshold regression on harmonized real wage",
                  fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig10v3_hansen_threshold.pdf"
    png = FIG_DIR / "fig10v3_hansen_threshold.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig10v3()
    print(f"wrote {pdf}")
    print(f"c_hat = {res['c_hat']:.3f} [{res['c_ci_lo']:.3f}, {res['c_ci_hi']:.3f}]")
    print(f"beta_M = {res['beta_M']:+.4f}, beta_T = {res['beta_T']:+.4f}")
    print(f"sup-Wald p-value = {res['sup_wald_pvalue']:.4f}")
    print(f"N = {res['n']}")
