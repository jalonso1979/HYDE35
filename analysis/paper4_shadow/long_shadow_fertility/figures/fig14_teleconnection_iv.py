"""Fig 14 — Teleconnection (NAO/AMO/ENSO) IV 2SLS. Activates when teleconnection_panel exists."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
TELE = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
             "teleconnection_panel.parquet")


def make_fig14():
    from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
        assemble_panel_multi,
    )
    from analysis.paper4_shadow.long_shadow_fertility.estimators.iv_2sls import (
        fit_iv_2sls,
    )
    panel = assemble_panel_multi()
    tele = pd.read_parquet(TELE)
    df = panel.merge(tele, on="year", how="left")
    CONTROLS = ["war_active", "log_war_fatalities", "pandemic_active",
                 "disaster_count", "log_disaster_deaths", "heat_extreme", "drought",
                 "vol_t_10y", "vol_p_10y"]
    instruments = [c for c in ("nao", "amo", "enso") if c in df.columns]
    df = df.dropna(subset=["log_cbr", "t_growing"] + instruments + CONTROLS)
    res = fit_iv_2sls(df, y="log_cbr", x="t_growing",
                       instruments=instruments, controls=CONTROLS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([0], [res["beta"]], yerr=[1.96 * res["se"]], color="seagreen", alpha=0.7, width=0.5)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks([]); ax.set_title(
        f"Teleconnection IV (NAO/AMO/ENSO) 2SLS β = {res['beta']:+.4f}  |  "
        f"first-stage F = {res['first_stage_f']:.1f}  |  AR p = {res['ar_pvalue']:.3g}")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig14_teleconnection_iv.pdf"
    png = FIG_DIR / "fig14_teleconnection_iv.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, res


if __name__ == "__main__":
    pdf, png, res = make_fig14()
    print(f"wrote {pdf}")
