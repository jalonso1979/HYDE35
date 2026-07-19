"""Fig 1v3 -- 12-panel rolling-window IRF (gap-filled England + 11 other countries)."""
from __future__ import annotations
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.rolling_window import (
    rolling_elasticity,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
PAPER_FIG_DIR = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/"
    "My Drive/Fertility/long_shadow/figures"
)
ORDER = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP", "NOR", "DNK", "FIN", "ISL", "CHE"]


def make_fig1v3(window: int = 40):
    df = assemble_panel_multi()
    estimates: dict[str, pd.DataFrame] = {}
    # 3x4 grid for 12 countries — no spare panels
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    for ax, iso in zip(axes.flat, ORDER):
        sub = df.loc[df["iso3"] == iso]
        est = rolling_elasticity(sub, y="log_cbr", x="t_growing", window=window)
        estimates[iso] = est
        ax.fill_between(est["center_year"], est["ci_low"], est["ci_high"],
                         alpha=0.25, color="steelblue")
        ax.plot(est["center_year"], est["beta"], color="steelblue", lw=1.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        for v in (1600, 1641, 1815, 1883, 1991):
            ax.axvline(v, color="gray", lw=0.4, alpha=0.5)
        ax.set_title(iso)
    fig.suptitle("Rolling-window IRF -- 12 countries (gap-filled England via HMD)", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig1v3_rolling_7country.pdf"
    png = FIG_DIR / "fig1v3_rolling_7country.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    # Copy to paper figures folder
    try:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, PAPER_FIG_DIR / "fig1v3_rolling_7country.pdf")
    except OSError as exc:
        print(f"WARNING: could not copy PDF to paper repo: {exc}")
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, _ = make_fig1v3()
    print(f"wrote {pdf}")
