"""Fig 23 - Climate uncertainty channel: IRF panels for M1/M2/M3.

Three rows:
  Row 1 (M1): IRF of log_cbr on t_growing + inset bar for realized_SD coefficient
  Row 2 (M2): IRF + inset bar for ensstd_t coefficient
  Row 3 (M3): IRF + inset bars for realized_SD (base) and interaction (t_sd_x_above)

Output:
  /Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig23_uncertainty_dl.{pdf,png}
  /Users/jalonso/.../Fertility/long_shadow/figures/fig23_uncertainty_dl.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

FIG_DIR_COMPUTE = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility"
)
FIG_DIR_PAPER = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/Fertility/long_shadow/figures"
)
DATA_IN = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility"
    "/phase10_uncertainty_dl.json"
)

LAGS = 3
MODEL_LABELS = {
    "m1_baseline_with_realized_sd": "M1: Realized within-season SD",
    "m2_baseline_with_ensstd": "M2: ModE-RA ensemble spread (ensstd)",
    "m3_realized_sd_x_threshold": "M3: Realized SD × Hansen threshold",
}
UNCERTAINTY_COEF = {
    "m1_baseline_with_realized_sd": "t_anom_c_within_season_sd",
    "m2_baseline_with_ensstd": "ensstd_t_growing",
    "m3_realized_sd_x_threshold": "t_anom_c_within_season_sd",
}
UNCERTAINTY_LABEL = {
    "m1_baseline_with_realized_sd": r"$\delta_{SD}$  realized SD",
    "m2_baseline_with_ensstd": r"$\delta_{ens}$  ensstd T",
    "m3_realized_sd_x_threshold": r"$\delta_{SD}$  realized SD (base)",
}

PALETTE = {"irf": "steelblue", "cum": "firebrick", "unc": "#E87722", "int": "#7B2D8B"}


def _load() -> dict:
    return json.loads(DATA_IN.read_text())


def _irf_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Split per-lag rows from cumulative row."""
    per_lag = [r for r in records if r["lag"] != "cumulative"]
    cum = [r for r in records if r["lag"] == "cumulative"][0]
    lags = np.array([r["lag"] for r in per_lag], dtype=int)
    betas = np.array([r["beta"] for r in per_lag])
    errs_lo = np.array([r["beta"] - r["ci_low"] for r in per_lag])
    errs_hi = np.array([r["ci_high"] - r["beta"] for r in per_lag])
    return lags, betas, np.vstack([errs_lo, errs_hi]), cum


def _draw_coef_bar(ax, label: str, coef_dict: dict, color: str,
                   y_pos: float = 0, bar_height: float = 0.6):
    """Draw a single horizontal bar + 95% CI for a scalar coefficient."""
    b = coef_dict["beta"]
    lo = coef_dict["ci_low"]
    hi = coef_dict["ci_high"]
    ax.barh(y_pos, b, height=bar_height, color=color, alpha=0.75, label=label)
    ax.errorbar(b, y_pos, xerr=[[b - lo], [hi - b]], fmt="none",
                color="black", capsize=3, lw=1.2)
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_yticks([y_pos])
    ax.set_yticklabels([label], fontsize=8)
    sig = "" if lo <= 0 <= hi else ("*" if abs(b / ((hi - lo) / 3.92)) > 1.645 else "")
    ax.set_xlabel(rf"Coefficient ($\pm$95% CI){sig}", fontsize=8)


def make_fig23():
    d = _load()
    keys = list(MODEL_LABELS.keys())

    fig = plt.figure(figsize=(12, 11))
    gs_outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    for row_i, mkey in enumerate(keys):
        res = d[mkey]
        irf_records = res["irf"]
        ctrl_coefs = res["ctrl_coefs"]

        lags, betas, errs, cum = _irf_arrays(irf_records)

        # Each row: IRF panel (wide) + uncertainty bar panel (narrow)
        # M3 also gets an interaction panel
        n_extra = 2 if mkey == "m3_realized_sd_x_threshold" else 1
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, 2 + (n_extra - 1), subplot_spec=gs_outer[row_i],
            width_ratios=[4] + [1.2] * n_extra, wspace=0.05,
        )

        # --- IRF bar chart ---
        ax_irf = fig.add_subplot(gs_inner[0])
        ax_irf.bar(lags, betas, color=PALETTE["irf"], alpha=0.7,
                   yerr=errs, error_kw={"lw": 1.2, "capsize": 4})
        ax_irf.errorbar(
            [LAGS + 1.3], [cum["beta"]],
            yerr=[[cum["beta"] - cum["ci_low"]], [cum["ci_high"] - cum["beta"]]],
            fmt="o", color=PALETTE["cum"], markersize=7,
            label=f"cum = {cum['beta']:+.3f} (SE {cum['se']:.3f})",
        )
        ax_irf.axhline(0, color="black", lw=0.6, ls="--")
        ax_irf.set_xticks(list(lags) + [LAGS + 1.3])
        ax_irf.set_xticklabels([f"k={k}" for k in lags] + ["cum"])
        ax_irf.set_xlabel("Lag (years)", fontsize=9)
        ax_irf.set_ylabel(r"$\beta_k$, log CBR on T$_{growing}$", fontsize=9)
        ax_irf.set_title(MODEL_LABELS[mkey], fontsize=10, fontweight="bold")
        ax_irf.legend(loc="best", frameon=False, fontsize=8)
        ax_irf.tick_params(labelsize=8)

        # --- Uncertainty coefficient bar ---
        unc_key = UNCERTAINTY_COEF[mkey]
        unc_label = UNCERTAINTY_LABEL[mkey]
        ax_unc = fig.add_subplot(gs_inner[1])
        if unc_key in ctrl_coefs:
            _draw_coef_bar(ax_unc, unc_label, ctrl_coefs[unc_key],
                           color=PALETTE["unc"])
        else:
            ax_unc.text(0.5, 0.5, f"{unc_key}\nnot in output",
                        ha="center", va="center", transform=ax_unc.transAxes,
                        fontsize=7, color="grey")
        ax_unc.tick_params(labelsize=8)

        # --- Interaction bar (M3 only) ---
        if mkey == "m3_realized_sd_x_threshold":
            ax_int = fig.add_subplot(gs_inner[2])
            int_key = "t_sd_x_above"
            if int_key in ctrl_coefs:
                _draw_coef_bar(ax_int, r"$\delta_{int}$  SD × above",
                               ctrl_coefs[int_key], color=PALETTE["int"])
            else:
                ax_int.text(0.5, 0.5, "t_sd_x_above\nnot in output",
                            ha="center", va="center", transform=ax_int.transAxes,
                            fontsize=7, color="grey")
            ax_int.tick_params(labelsize=8)

    fig.suptitle(
        "Fig 23: Climate uncertainty channel — distributed-lag IRFs\n"
        r"(7-country pooled OLS, country + year FE, cluster-robust SE)",
        fontsize=11, y=1.01,
    )

    FIG_DIR_COMPUTE.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PAPER.mkdir(parents=True, exist_ok=True)

    pdf_comp = FIG_DIR_COMPUTE / "fig23_uncertainty_dl.pdf"
    png_comp = FIG_DIR_COMPUTE / "fig23_uncertainty_dl.png"
    pdf_paper = FIG_DIR_PAPER / "fig23_uncertainty_dl.pdf"

    fig.savefig(pdf_comp, bbox_inches="tight")
    fig.savefig(png_comp, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_paper, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {pdf_comp}")
    print(f"Wrote {png_comp}")
    print(f"Wrote {pdf_paper}")
    return pdf_comp, png_comp, pdf_paper


if __name__ == "__main__":
    make_fig23()
