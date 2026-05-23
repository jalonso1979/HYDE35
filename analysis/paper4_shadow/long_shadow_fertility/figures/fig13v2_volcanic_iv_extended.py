"""Fig 13v2 — Volcanic IV first stage + LATE on the extended Sigl+Sato series.

Reuses the Phase 3 iv_2sls estimator with the Phase 7 spliced volcanic panel
(1700-2012). Returns BLOCKED if the splice was unavailable.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ERUPTION_LABELS = {1815: "Tambora", 1883: "Krakatau", 1902: "Sta. Maria",
                    1912: "Katmai", 1963: "Agung", 1982: "El Chichón", 1991: "Pinatubo"}


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

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax = axes[0]
    ax.bar(volc["year"], volc["vssi"], width=0.8,
            color=np.where(volc["source"] == "Sigl_VSSI", "navy", "darkorange"))
    for y, label in ERUPTION_LABELS.items():
        if y in volc["year"].values:
            row = volc.loc[volc["year"] == y].iloc[0]
            ax.annotate(label, (y, row["vssi"]), textcoords="offset points",
                          xytext=(0, 5), fontsize=8, ha="center")
    ax.set_ylabel("VSSI (Tg S equiv.) / rescaled Sato AOD")
    ax.set_title("A. Spliced volcanic forcing series (Sigl 1700-1900 + Sato 1900-2012)")

    ax = axes[1]
    h0 = ["IV LATE β", "First-stage F", "AR p-value"]
    vals = [iv_res.get("beta"),
             iv_res.get("first_stage_f"),
             iv_res.get("ar_pvalue")]
    ax.barh(h0, vals, color=["firebrick", "steelblue", "darkgreen"])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Estimate / statistic")
    ax.set_title("B. IV 2SLS results (T instrumented by spliced volcanic forcing)")

    fig.suptitle("Fig 13v2 — Extended volcanic IV reaching Pinatubo 1991",
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
        print(iv)
