"""GAEZ-style crop suitability robustness for Stage 1.

We replicate the climate-based logic FAO uses in its Global Agro-Ecological
Zones rainfed-wheat suitability layer, using only variables we can derive
from the ModE-RA + CRU monthly panel. This serves as an alternative measure
to the productive-months index in Section 5.

Rainfed wheat suitability (after FAO GAEZ documentation):
  1. Growing degree days base 5°C in growing season:  1500-3500 ideal
  2. Annual precipitation:                            250-1500 mm ideal
  3. Frost months (mean T < 0°C):                     ≤ 5
  4. Mean T of warmest month:                         18-30°C

Each criterion is mapped to a 0-1 score via piecewise-linear membership;
the suitability index is the geometric mean of the four scores.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def _trapezoid(x: np.ndarray, lo_off: float, lo_on: float, hi_on: float, hi_off: float) -> np.ndarray:
    """0 below lo_off, ramps to 1 by lo_on, 1 between lo_on and hi_on, ramps to 0 by hi_off."""
    out = np.zeros_like(x, dtype=np.float64)
    rising = (x > lo_off) & (x < lo_on)
    out[rising] = (x[rising] - lo_off) / (lo_on - lo_off)
    plateau = (x >= lo_on) & (x <= hi_on)
    out[plateau] = 1.0
    falling = (x > hi_on) & (x < hi_off)
    out[falling] = 1.0 - (x[falling] - hi_on) / (hi_off - hi_on)
    return out


def main() -> None:
    print("Building country-year wheat-suitability index from ModE-RA + CRU climatology...")
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    pre = df[df["year"].between(1421, 1750)].copy()

    # Year-level metrics
    yr = pre.groupby(["iso3", "year"], as_index=False).agg(
        gdd5 = ("t_abs", lambda s: (np.clip(s - 5, 0, None) * 30).sum()),
        p_ann = ("p_abs", "sum"),
        frost_months = ("t_abs", lambda s: (s < 0).sum()),
        warmest_month_T = ("t_abs", "max"),
    )
    feats = yr.groupby("iso3", as_index=False).agg(
        gdd5 = ("gdd5", "mean"),
        p_ann = ("p_ann", "mean"),
        frost_months = ("frost_months", "mean"),
        warmest_month_T = ("warmest_month_T", "mean"),
    )

    # Trapezoid membership functions per FAO GAEZ rainfed wheat
    feats["s_gdd"] = _trapezoid(feats["gdd5"].values, 800, 1500, 3500, 5000)
    feats["s_pre"] = _trapezoid(feats["p_ann"].values, 100, 250, 1500, 2500)
    feats["s_frost"] = 1.0 - np.clip(feats["frost_months"].values / 5.0, 0, 1)
    feats["s_warm"] = _trapezoid(feats["warmest_month_T"].values, 14, 18, 30, 35)

    # Geometric mean of the four scores
    feats["wheat_suit"] = (feats["s_gdd"] * feats["s_pre"] *
                           feats["s_frost"] * feats["s_warm"]) ** 0.25

    feats.to_parquet(DATA / "gaez_style_wheat_suitability.parquet", index=False)
    print(f"\nSaved suitability index for {len(feats)} countries")
    print(f"Median suitability: {feats['wheat_suit'].median():.3f}")
    print(f"Top 10 countries by wheat suitability:")
    print(feats.nlargest(10, 'wheat_suit')[['iso3', 'wheat_suit', 'gdd5', 'p_ann', 'frost_months']].to_string(index=False))

    # Stage 1 ANOVA replication with wheat_suit
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    d = feats.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    big = d[d["cluster"] != 2]
    print(f"\n=== Stage 1 ANOVA: wheat_suit by pathway (N = {len(big)}) ===")
    print(big.groupby("cluster").agg(
        n=("iso3", "count"), wheat_suit=("wheat_suit", "mean")).round(3))
    groups = [big.loc[big["cluster"] == k, "wheat_suit"].values
              for k in sorted(big["cluster"].unique())]
    groups = [g[~np.isnan(g)] for g in groups]
    F, p = stats.f_oneway(*groups)
    print(f"\nANOVA: F = {F:.3f}, p = {p:.4g}")

    # Compare to the productive-months ANOVA
    print(f"\nFor reference (from Section 5.2):")
    print(f"  productive_months ANOVA: F = 6.31, p < 0.001")
    print(f"  wheat_suit            ANOVA: F = {F:.2f}, p = {p:.3g}")

    # Multinomial logit
    big = big.dropna(subset=["wheat_suit", "gdd5"])
    print(f"\n=== MNLogit: pathway ~ wheat_suit (N = {len(big)}) ===")
    X = sm.add_constant(big[["wheat_suit"]])
    y = big["cluster"]
    r = sm.MNLogit(y, X).fit(method="bfgs", maxiter=200, disp=False)
    print(f"Pseudo-R^2 = {r.prsquared:.4f}, LLR p = {r.llr_pvalue:.4g}")


if __name__ == "__main__":
    main()
