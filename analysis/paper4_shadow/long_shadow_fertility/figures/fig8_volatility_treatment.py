"""Fig 8 - Climate volatility as treatment: beta^L (level) vs beta^V (vol) per country."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ORDER = ["GBR", "FRA", "ITA", "SWE"]


def _level_vol_fit(sub: pd.DataFrame) -> dict:
    sub = sub.dropna(subset=["log_cbr", "t_growing", "vol_t_10y"])
    if len(sub) < 30:
        return {"beta_L": None, "beta_V": None, "se_L": None, "se_V": None, "n": len(sub)}
    X = sm.add_constant(sub[["t_growing", "vol_t_10y"]].astype(float))
    res = sm.OLS(sub["log_cbr"].astype(float).to_numpy(), X.to_numpy()).fit(cov_type="HC1")
    return {"beta_L": float(res.params[1]), "beta_V": float(res.params[2]),
            "se_L": float(res.bse[1]), "se_V": float(res.bse[2]), "n": int(len(sub))}


def make_fig8_vol():
    df = assemble_panel_multi()
    estimates: dict[str, dict] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, iso in zip(axes.flat, ORDER):
        sub = df.loc[df["iso3"] == iso]
        fit = _level_vol_fit(sub)
        estimates[iso] = fit
        if fit["beta_L"] is None:
            ax.set_title(f"{iso} (insufficient data, n={fit['n']})")
            continue
        x_pos = [0, 1]
        y_vals = [fit["beta_L"], fit["beta_V"]]
        yerr = [1.96 * fit["se_L"], 1.96 * fit["se_V"]]
        ax.bar(x_pos, y_vals, yerr=yerr, color=["steelblue", "firebrick"], alpha=0.7,
                tick_label=[r"$\beta^L$ (T level)", r"$\beta^V$ (T vol 10y)"])
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(f"{iso} (n={fit['n']})")
    fig.suptitle("Climate level (beta^L) vs volatility (beta^V) on log CBR - 4 countries", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig8_volatility_treatment.pdf"
    png = FIG_DIR / "fig8_volatility_treatment.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200); plt.close(fig)
    return pdf, png, estimates


if __name__ == "__main__":
    pdf, png, _ = make_fig8_vol()
    print(f"wrote {pdf}")
