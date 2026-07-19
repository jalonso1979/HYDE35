"""Fig 11 -- Mediation T -> log real wage -> log CBR with effect decomposition."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel import (
    build_real_wage_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.mediation import (
    fit_mediation,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def make_fig11_mediation():
    panel = assemble_panel_multi()
    wages = build_real_wage_panel()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(wages, on=["iso3", "year"], how="left")
    sub = df.dropna(subset=["log_cbr", "t_growing", "log_real_wage"])
    res = fit_mediation(sub, y="log_cbr", x="t_growing", m="log_real_wage", n_boot=200)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    boxes = {"T (growing-season)": (0.05, 0.5),
              "log real wage (M)": (0.45, 0.5),
              "log CBR (Y)": (0.85, 0.5)}
    for label, (x, y) in boxes.items():
        ax.text(x, y, label, ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black"))
    ax.annotate("", xy=(0.4, 0.5), xytext=(0.13, 0.5),
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.265, 0.55, f"$\\varphi$ = {res['phi']:+.4f}", ha="center", fontsize=10, color="darkgreen")
    ax.annotate("", xy=(0.8, 0.5), xytext=(0.53, 0.5),
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.665, 0.55, f"$\\delta$ = {res['delta']:+.4f}", ha="center", fontsize=10, color="darkgreen")
    ax.annotate("", xy=(0.8, 0.4), xytext=(0.13, 0.4),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="firebrick",
                                  connectionstyle="arc3,rad=-0.3"))
    ax.text(0.46, 0.30, f"direct $\\beta$ = {res['direct']:+.4f} (SE {res['direct_se']:.4f})",
            ha="center", fontsize=10, color="firebrick")
    ax.text(0.46, 0.78,
            f"indirect ($\\varphi\\cdot\\delta$) = {res['indirect']:+.4f} (SE {res['indirect_se']:.4f})  |  "
            f"total = {res['total']:+.4f}  |  N={res['n']}",
            ha="center", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Climate-fertility mediation through real wages")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig11_mediation_diagram.pdf"
    png = FIG_DIR / "fig11_mediation_diagram.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig11_mediation()
    print(f"wrote {pdf}; direct={res['direct']:.4f}, indirect={res['indirect']:.4f}, total={res['total']:.4f}")
