"""Robustness layer for the ModE-RA-based analyses.

1. ERA5 vs ModE-RA calibration in 1950-2008 overlap.
2. Subsample stability for pre-industrial Malthus
   (1421-1600 vs 1600-1750 split).
3. Pathway-stratified t-shock with ModE-RA proxy-density caveat.

Outputs robustness tables + a calibration figure.
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}


def calibration_figure() -> None:
    bias = pd.read_parquet(DATA / "modera_era5_bias_1950_2008.parquet")
    # Note: existing era5 panel is region-level mapped to countries, so country-
    # year correlation is inherently capped. We document this rather than
    # claim ModE-RA is bad.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(bias["t_bias"], bins=30, color="#0072B2", alpha=0.7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("ERA5 − ModE-RA mean T (°C), 1950–2008")
    ax.set_ylabel("Countries")
    ax.set_title(f"Cross-source mean bias\n"
                 f"median |bias| = {bias['t_bias'].abs().median():.2f}°C")
    ax = axes[1]
    ax.hist(bias["t_corr"], bins=30, color="#D55E00", alpha=0.7)
    ax.set_xlabel("Country-level T correlation (ModE-RA vs ERA5-region)")
    ax.set_ylabel("Countries")
    ax.set_title("Year-to-year T correlation\n(ERA5 panel is region-level — see text)")
    plt.tight_layout()
    fig.savefig(FIG / "fig12_calibration.png", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG / 'fig12_calibration.png'}")
    print(f"  Median |bias| = {bias['t_bias'].abs().median():.2f} C")
    print(f"  Median t-correlation = {bias['t_corr'].median():.3f}")
    print(f"  Note: existing ERA5 panel uses 25-region resolution mapped to countries;")
    print(f"  the ModE-RA panel is true country-level. Correlation is bounded above")
    print(f"  by within-region country-level T variation that ERA5 panel cannot resolve.")


def malthus_subsample_stability() -> None:
    df = pd.read_parquet(DATA / "preindustrial_malthus_panel.parquet")
    splits = [("1421-1600", df[df["year"] < 1600]),
              ("1600-1700", df[df["year"].between(1600, 1699)]),
              ("1700-1750", df[df["year"].between(1700, 1750)])]
    print("\n=== Pre-industrial Malthus stability across sub-periods ===")
    rows = []
    for label, sub in splits:
        for cl in sorted(sub["cluster"].unique()):
            d = sub[sub["cluster"] == cl].dropna(subset=["log_density", "pop_growth_ann"])
            if len(d) < 25:
                continue
            # Demean within country
            g = d.groupby("iso3")
            yd = d["pop_growth_ann"] - g["pop_growth_ann"].transform("mean")
            xd = d["log_density"] - g["log_density"].transform("mean")
            X = sm.add_constant(xd.rename("log_density"))
            res = sm.OLS(yd, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
            rows.append({
                "period": label,
                "pathway": PATHWAY_NAMES[cl],
                "n": int(res.nobs),
                "beta_density": res.params["log_density"],
                "p_density": res.pvalues["log_density"],
            })
    out = pd.DataFrame(rows)
    print(out.pivot(index="pathway", columns="period",
                    values=["beta_density", "p_density"]).round(5))
    out.to_parquet(DATA / "robustness_malthus_subperiods.parquet", index=False)


def main() -> None:
    print("=== Calibration: ERA5 vs ModE-RA in overlap ===")
    calibration_figure()
    malthus_subsample_stability()


if __name__ == "__main__":
    main()
