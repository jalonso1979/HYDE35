"""Fig 4 -- country x decade beta heatmap of climate-fertility elasticity."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def _cell_beta(sub: pd.DataFrame) -> float:
    sub = sub.dropna(subset=["log_cbr", "t_growing"])
    if len(sub) < 5:
        return np.nan
    X = sm.add_constant(sub[["t_growing"]])
    res = sm.OLS(sub["log_cbr"].to_numpy(), X.to_numpy()).fit()
    return float(res.params[1])


def make_fig4_heatmap():
    df = assemble_panel_multi()
    df["decade"] = (df["year"] // 10) * 10
    df = df.loc[df["decade"].between(1700, 2000)]
    mat = (df.groupby(["iso3", "decade"]).apply(_cell_beta)
           .unstack("decade").sort_index())

    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(0, len(mat.columns), 2))
    ax.set_xticklabels([int(d) for d in mat.columns[::2]], rotation=45)
    ax.set_xlabel("Decade")
    ax.set_title("Climate-fertility elasticity beta by country x decade")
    fig.colorbar(im, ax=ax, label=r"$\beta$ on $T_{growing}$")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig4_country_decade_heatmap.pdf"
    png = FIG_DIR / "fig4_country_decade_heatmap.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png, mat


if __name__ == "__main__":
    pdf, png, _ = make_fig4_heatmap()
    print(f"wrote {pdf}")
