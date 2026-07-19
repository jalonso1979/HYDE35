"""Fig 13v2 — Volcanic IV first stage + LATE on the extended Sigl+Sato series.

Reuses the Phase 3 iv_2sls estimator with the Phase 7 spliced volcanic panel
(1700-2012). Returns BLOCKED if the splice was unavailable.

Phase 10 addition: Anderson-Rubin identification-robust CI (Wright 2003) added
alongside the 2SLS LATE. The AR CI is computed after partialling out country FEs,
using the single scalar 'vssi' instrument (just-identified case). Under weak
first-stage identification (F < 10), the AR CI may be very wide or unbounded —
this is the correct output and should be reported in the paper.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_volcanic_panel_v2 import (
    build_volcanic_panel_v2,
    BLOCKED,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.iv_2sls import (
    fit_iv_2sls,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.ar_inference import (
    anderson_rubin_ci,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ERUPTION_LABELS = {1815: "Tambora", 1883: "Krakatau", 1902: "Sta. Maria",
                    1912: "Katmai", 1963: "Agung", 1982: "El Chichón", 1991: "Pinatubo"}


def _partial_out_fe(pnl: pd.DataFrame, cols: list[str], unit_col: str = "iso3") -> list[np.ndarray]:
    """Return residuals after partialling out country fixed effects."""
    unit_dums = pd.get_dummies(pnl[unit_col], drop_first=True, dtype=float)
    W = sm.add_constant(unit_dums).to_numpy()
    result = []
    for c in cols:
        v = pnl[c].astype(float).to_numpy()
        resid = v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
        result.append(resid)
    return result


def make_fig13v2():
    volc = build_volcanic_panel_v2()
    if volc is BLOCKED:
        return BLOCKED

    panel = assemble_panel_multi()
    pnl = panel.merge(volc[["year", "vssi"]], on="year", how="inner")
    pnl = pnl.dropna(subset=["log_cbr", "t_growing", "vssi"]).copy()

    iv_res = fit_iv_2sls(
        pnl, y="log_cbr", x="t_growing", instruments=["vssi"], unit_col="iso3"
    )

    # --- Phase 10: Anderson-Rubin identification-robust CI ---
    # Partial out country FEs from y, x, z; then apply scalar AR CI.
    # Just-identified (single instrument), so AR CI is exact even under weak F.
    y_res, x_res, z_res = _partial_out_fe(pnl, ["log_cbr", "t_growing", "vssi"])
    ar_lo, ar_hi = anderson_rubin_ci(y_res, x_res, z_res, alpha=0.05,
                                      beta_range=(-15.0, 15.0), grid_n=3001)
    iv_res["ar_ci_lo"] = ar_lo
    iv_res["ar_ci_hi"] = ar_hi

    _lo_str = f"{ar_lo:+.3f}" if np.isfinite(ar_lo) else "-inf"
    _hi_str = f"{ar_hi:+.3f}" if np.isfinite(ar_hi) else "+inf"
    print(
        f"[Fig13v2 AR CI] 2SLS LATE β = {iv_res['beta']:+.4f}  "
        f"95% CI (standard) = [{iv_res['beta'] - 1.96*iv_res['se']:+.4f}, "
        f"{iv_res['beta'] + 1.96*iv_res['se']:+.4f}]"
    )
    print(
        f"[Fig13v2 AR CI] Anderson-Rubin 95% CI (Wright 2003, just-identified vssi) = "
        f"[{_lo_str}, {_hi_str}]  |  first-stage F = {iv_res['first_stage_f']:.2f}"
    )
    if not (np.isfinite(ar_lo) and np.isfinite(ar_hi)):
        print(
            "[Fig13v2 AR CI] WARNING: AR CI is unbounded — identification is too weak "
            "to bound the IV estimate. Report as 'AR CI: (-inf, +inf)' in paper. "
            "NOTE for .tex: the AR CI spans the entire parameter space; this is consistent "
            "with F < 1 first stage and underscores the weak-IV caveat."
        )
    else:
        print(
            f"NOTE for .tex: \\beta_{{IV}} = {iv_res['beta']:.3f}, "
            f"AR\\,95\\%\\,CI = [{ar_lo:.3f},\\,{ar_hi:.3f}], "
            f"F = {iv_res['first_stage_f']:.2f}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax = axes[0]
    ax.bar(volc["year"], volc["vssi"], width=0.8,
            color=np.where(volc["source"] == "Sigl_VSSI", "navy", "darkorange"))
    for yr, label in ERUPTION_LABELS.items():
        if yr in volc["year"].values:
            row = volc.loc[volc["year"] == yr].iloc[0]
            ax.annotate(label, (yr, row["vssi"]), textcoords="offset points",
                          xytext=(0, 5), fontsize=8, ha="center")
    ax.set_ylabel("VSSI (Tg S equiv.) / rescaled Sato AOD")
    ax.set_title("A. Spliced volcanic forcing series (Sigl 1700-1900 + Sato 1900-2012)")

    ax = axes[1]
    # Show 2SLS point estimate with both standard CI and AR CI
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.bar([0], [iv_res["beta"]], yerr=[[1.96 * iv_res["se"]]], color="firebrick",
            alpha=0.7, width=0.4, label=r"2SLS $\beta$ ± 1.96 SE")
    if np.isfinite(ar_lo) and np.isfinite(ar_hi):
        ax.errorbar([0.05], [iv_res["beta"]],
                    yerr=[[iv_res["beta"] - ar_lo], [ar_hi - iv_res["beta"]]],
                    fmt="none", color="navy", capsize=6, lw=2,
                    label=f"AR 95% CI [{_lo_str}, {_hi_str}]")
    else:
        ax.text(0.5, 0.5,
                f"AR 95% CI: unbounded\n(F={iv_res['first_stage_f']:.2f} < 1; weak IV)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8})
    ax.set_xticks([0])
    ax.set_xticklabels([r"$\beta_{IV}$"])
    ax.set_ylabel(r"$\beta$ (climate-fertility elasticity)")
    ax.set_title(
        f"B. 2SLS LATE β = {iv_res['beta']:+.4f} (SE {iv_res['se']:.4f})\n"
        f"AR 95% CI: [{_lo_str}, {_hi_str}]  |  First-stage F = {iv_res['first_stage_f']:.2f}"
    )
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Fig 13v2 — Extended volcanic IV + AR identification-robust CI (Phase 10)",
                  fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig13v2_volcanic_iv_extended.pdf"
    png = FIG_DIR / "fig13v2_volcanic_iv_extended.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png, iv_res


if __name__ == "__main__":
    res = make_fig13v2()
    if res is BLOCKED:
        print("BLOCKED: volcanic panel v2 unavailable; Fig 13v2 not built")
    else:
        pdf, png, iv = res
        print(f"wrote {pdf}")
        print(f"beta={iv['beta']:.4f}, se={iv['se']:.4f}, F={iv['first_stage_f']:.2f}")
        print(f"AR CI: [{iv.get('ar_ci_lo', 'n/a')}, {iv.get('ar_ci_hi', 'n/a')}]")
