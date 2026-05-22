"""Fig 3-R — Phase 1 England volcanic event studies with controls in the regression."""
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

ERUPTIONS = [("Huaynaputina", 1600), ("Tambora", 1815), ("Pinatubo", 1991)]
WINDOW_RADIUS = 30
CONTROL_COLS = [
    "war_active", "log_war_fatalities", "pandemic_active",
    "disaster_count", "log_disaster_deaths",
    "heat_extreme", "drought", "vol_t_10y", "vol_p_10y",
]


def _event_study_with_controls(df: pd.DataFrame, y: str, eruption_year: int,
                                  pre: int = 5, post: int = 10,
                                  reference_h: int = -1) -> pd.DataFrame:
    sub = df.loc[df["year"].between(eruption_year - WINDOW_RADIUS, eruption_year + WINDOW_RADIUS)].copy()
    sub = sub.dropna(subset=[y] + CONTROL_COLS).sort_values("year").reset_index(drop=True)
    sub["h"] = sub["year"] - eruption_year
    horizons = [h for h in range(-pre, post + 1) if h != reference_h]
    for h in horizons:
        sub[f"D_h{h}"] = (sub["h"] == h).astype(int)
    sub["trend"] = sub["year"] - sub["year"].mean()
    X_cols = [f"D_h{h}" for h in horizons] + ["trend"] + CONTROL_COLS
    X = sm.add_constant(sub[X_cols])
    res = sm.OLS(sub[y].to_numpy(), X.to_numpy()).fit(cov_type="HC1")
    rows = [{"h": reference_h, "delta": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0}]
    for i, h in enumerate(horizons):
        idx = i + 1
        b = float(res.params[idx])
        s = float(res.bse[idx])
        rows.append({"h": h, "delta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)


def make_fig3r():
    df = assemble_panel_multi()
    eng = df.loc[df["iso3"] == "GBR"].copy()
    for c in CONTROL_COLS:
        if c in eng.columns:
            eng[c] = eng[c].fillna(0)
    est = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (name, yr) in zip(axes, ERUPTIONS):
        res = _event_study_with_controls(eng, y="log_cbr", eruption_year=yr)
        est[name] = res
        ax.fill_between(res["h"], res["ci_low"], res["ci_high"], alpha=0.25, color="firebrick")
        ax.plot(res["h"], res["delta"], marker="o", color="firebrick", lw=1.5)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{name} ({yr}) — with controls")
        ax.set_xlabel("Years from eruption")
    axes[0].set_ylabel("Δ log CBR (h=-1 ref)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig3r_volcanic_with_controls_england.pdf"
    png = FIG_DIR / "fig3r_volcanic_with_controls_england.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, est


if __name__ == "__main__":
    pdf, png, _ = make_fig3r()
    print(f"wrote {pdf}")
