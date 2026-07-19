"""Long-shadow regression on additional modern outcomes.

Beyond modern population growth and crop-share change, we test whether
pre-industrial inter-annual temperature volatility predicts:
  - GDP per capita level (2015-2018 mean, from PWT)
  - Log growth of GDP per capita 1995-2018
  - Urbanization rate level (modern HYDE)
  - Urbanization rate change 1950→2020

The long-shadow story is more general than just population: σ_v should
predict any indicator of modern economic development if it traces a
genuine deep determinant. We use the standard deep-determinants battery
of controls (latitude, log area, landlocked, Neolithic distance, Addis
distance).
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"


def load_gdp_panel() -> pd.DataFrame:
    df = pd.read_excel(DATA / "pwt_gdppc.xlsx")
    df = df.rename(columns={"COU": "iso3", "Year": "year", "GDPpc": "gdp_pc"})
    # Modern level: mean 2015-2018
    late = df[df["year"].between(2015, 2018)].groupby("iso3", as_index=False).agg(
        gdp_pc_late=("gdp_pc", "mean"))
    # Earlier baseline: mean 1995-1998
    early = df[df["year"].between(1995, 1998)].groupby("iso3", as_index=False).agg(
        gdp_pc_early=("gdp_pc", "mean"))
    out = late.merge(early, on="iso3", how="outer")
    out["log_gdp_pc_late"] = np.log(out["gdp_pc_late"])
    out["log_gdp_growth_1995_2018"] = np.log(out["gdp_pc_late"]) - np.log(out["gdp_pc_early"])
    return out


def main() -> None:
    # Modern outcomes from HYDE panel
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        pop_e=("pop", "mean"), urban_e=("urban_share", "mean"),
        crop_e=("crop_share", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        pop_l=("pop", "mean"), urban_l=("urban_share", "mean"),
        crop_l=("crop_share", "mean"))
    out = early.merge(late, on="iso3")
    out = out[(out["pop_e"] > 0) & (out["pop_l"] > 0)]
    out["log_pop_growth"] = np.log(out["pop_l"] / out["pop_e"])
    out["d_urban"] = out["urban_l"] - out["urban_e"]
    out["d_crop"] = out["crop_l"] - out["crop_e"]
    out["urban_2020"] = out["urban_l"]

    # GDP per capita (Penn World Tables)
    gdp = load_gdp_panel()
    out = out.merge(gdp, on="iso3", how="left")
    print(f"Sample with GDP: {out['gdp_pc_late'].notna().sum()} of {len(out)} countries")

    # Pre-industrial climate predictor
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)]
    feats = pre.groupby("iso3", as_index=False).agg(
        sigma_v=("t_mean", "std"), t_mean=("t_mean", "mean"))

    # Deep determinants controls
    deep = pd.read_parquet(DATA / "deep_determinants.parquet")
    ext_d = pd.read_parquet(DATA / "deep_determinants_extended.parquet")
    deep = deep.merge(ext_d[["iso3", "log_mig_dist_addis"]], on="iso3", how="left")

    df = out.merge(feats, on="iso3").merge(deep, on="iso3")
    print(f"Final sample: {len(df)} countries")

    outcomes = [
        ("log_pop_growth",     "Log pop growth 1955→2020",          197),
        ("d_urban",            "Δ urban share 1955→2020",           197),
        ("log_gdp_pc_late",    "Log GDP per capita 2015–2018",      183),
        ("log_gdp_growth_1995_2018", "Log GDP/cap growth 1995→2018",  183),
    ]

    print("\n=== Long-shadow on modern outcomes ===")
    print(f"{'outcome':<35} {'spec':<25} {'beta':>8} {'SE':>8} {'p':>10}  R²    N")
    rows = []
    for outcome, label, n_expected in outcomes:
        for spec_label, ctrls in [
            ("baseline",           []),
            ("+ |lat|",            ["abs_lat"]),
            ("+ |lat| + deep det.", ["abs_lat", "log_area", "landlocked",
                                       "log_dist_neolithic", "log_mig_dist_addis"]),
        ]:
            sub = df.dropna(subset=[outcome, "sigma_v", "abs_lat"] + ctrls)
            if len(sub) < 30: continue
            X = sm.add_constant(sub[["sigma_v"] + ctrls])
            y = sub[outcome]
            r = sm.OLS(y, X).fit(cov_type="HC1")
            rows.append({
                "outcome": label, "spec": spec_label,
                "beta_sigma_v": r.params["sigma_v"],
                "se":  r.bse["sigma_v"],
                "p":   r.pvalues["sigma_v"],
                "rsq": r.rsquared, "n":   int(r.nobs),
            })
            print(f"{label[:33]:<35} {spec_label:<25} "
                  f"{r.params['sigma_v']:>+8.3f} {r.bse['sigma_v']:>8.3f} "
                  f"{r.pvalues['sigma_v']:>10.4g}  "
                  f"{r.rsquared:>4.3f} {int(r.nobs):>4}")
    res = pd.DataFrame(rows)
    res.to_parquet(DATA / "long_shadow_extra_outcomes.parquet", index=False)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (outcome, label, _) in zip(axes.flat, outcomes):
        sub = df.dropna(subset=[outcome, "sigma_v"])
        ax.scatter(sub["sigma_v"], sub[outcome], s=10,
                   color="#404040", alpha=0.5, edgecolor="white", linewidth=0.3)
        z = np.polyfit(sub["sigma_v"], sub[outcome], 1)
        xs = np.array([sub["sigma_v"].min(), sub["sigma_v"].max()])
        ax.plot(xs, z[0]*xs + z[1], color="#000000", linewidth=1.2)
        ax.set_xlabel(r"Pre-industrial $\sigma_v^T$ (1421–1750)")
        ax.set_ylabel(label)
        from scipy import stats
        slope, intercept, rval, pval, stderr = stats.linregress(sub["sigma_v"], sub[outcome])
        ax.text(0.04, 0.96, fr"$\hat\beta = {slope:+.3f}$, $p = {pval:.2g}$, $N={len(sub)}$",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(label.replace("→", "→"), loc="left", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG / "fig16_long_shadow_outcomes.pdf")
    fig.savefig(FIG / "fig16_long_shadow_outcomes.png")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig16_long_shadow_outcomes.pdf'}")


if __name__ == "__main__":
    main()
