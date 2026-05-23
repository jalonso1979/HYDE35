"""Fig 11v2 -- Mediation T -> wage -> fertility with cluster bootstrap SE."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel import (
    build_real_wage_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.cluster_bootstrap import (
    block_bootstrap,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def _fit_step(df: pd.DataFrame, y_col: str, regressors: list[str], unit_col: str = "iso3"):
    dums = pd.get_dummies(df[unit_col], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[regressors].astype(float), dums], axis=1))
    res = sm.OLS(df[y_col].astype(float).to_numpy(), X.to_numpy()).fit()
    return {r: float(res.params[1 + i]) for i, r in enumerate(regressors)}


def _indirect(d: pd.DataFrame) -> float:
    s1 = _fit_step(d, "log_real_wage", ["t_growing"])
    s2 = _fit_step(d, "log_cbr", ["t_growing", "log_real_wage"])
    return s1["t_growing"] * s2["log_real_wage"]


def _direct(d: pd.DataFrame) -> float:
    s2 = _fit_step(d, "log_cbr", ["t_growing", "log_real_wage"])
    return s2["t_growing"]


def make_fig11v2():
    panel = assemble_panel_multi()
    wages = build_real_wage_panel()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(wages, on=["iso3", "year"], how="left")
    sub = df.dropna(subset=["log_cbr", "t_growing", "log_real_wage", "iso3"])

    direct = _direct(sub)
    indirect = _indirect(sub)
    total = direct + indirect

    boot_indirect = block_bootstrap(sub, _indirect, cluster_col="iso3", n_boot=500, seed=0)
    boot_direct = block_bootstrap(sub, _direct, cluster_col="iso3", n_boot=500, seed=0)

    res = {
        "direct": float(direct),
        "indirect": float(indirect),
        "total": float(total),
        "indirect_cluster_se": float(np.std(boot_indirect)) if boot_indirect else float("nan"),
        "direct_cluster_se": float(np.std(boot_direct)) if boot_direct else float("nan"),
        "n_clusters": int(sub["iso3"].nunique()),
        "n": int(len(sub)),
    }

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
    ax.annotate("", xy=(0.8, 0.5), xytext=(0.53, 0.5),
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.8, 0.4), xytext=(0.13, 0.4),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="firebrick",
                                  connectionstyle="arc3,rad=-0.3"))
    ax.text(0.46, 0.30,
            f"direct β = {res['direct']:+.4f} (cluster SE {res['direct_cluster_se']:.4f})",
            ha="center", fontsize=10, color="firebrick")
    ax.text(0.46, 0.78,
            f"indirect (φ·δ) = {res['indirect']:+.4f} "
            f"(cluster SE {res['indirect_cluster_se']:.4f})  |  "
            f"total = {res['total']:+.4f}  |  N={res['n']}, clusters={res['n_clusters']}",
            ha="center", fontsize=10)
    ax.text(0.5, 0.10,
            f"WARNING: only {res['n_clusters']} clusters -- wild cluster bootstrap is preferred (Phase 5)",
            ha="center", fontsize=8, style="italic", color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Climate-fertility mediation through real wages (cluster bootstrap SE)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig11v2_mediation_cluster_bootstrap.pdf"
    png = FIG_DIR / "fig11v2_mediation_cluster_bootstrap.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig11v2()
    print(f"wrote {pdf}; indirect={res['indirect']:.4f} (cluster SE {res['indirect_cluster_se']:.4f})")
