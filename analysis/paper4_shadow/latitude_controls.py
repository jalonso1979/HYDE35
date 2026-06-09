"""Long-shadow regressions with latitude controls.

A natural worry: modern population growth is high in tropical countries
(demographic-transition lag) and pre-industrial climate volatility is high
in mid-latitudes (continental seasonality). The cross-country long-shadow
result could be a latitude story masquerading as a climate story.

We address this by controlling for absolute latitude in the long-shadow
regression and reporting whether the climate coefficients survive.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def main() -> None:
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    centroids = ext[["iso3", "centroid_lat", "centroid_lon"]].drop_duplicates()
    centroids["abs_lat"] = centroids["centroid_lat"].abs()

    # Outcome variables (modern change)
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        crop_e=("crop_share", "mean"), pop_e=("pop", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        crop_l=("crop_share", "mean"), pop_l=("pop", "mean"))
    out = early.merge(late, on="iso3")
    out = out[(out["pop_e"] > 0) & (out["pop_l"] > 0)]
    out["log_pop_growth"] = np.log(out["pop_l"] / out["pop_e"])
    out["d_crop_share"] = out["crop_l"] - out["crop_e"]

    # Pre-industrial climate predictors
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)]
    feats = pre.groupby("iso3", as_index=False).agg(
        sigma_s_real=("sigma_s", "mean"),
        sigma_v_real=("t_mean", "std"),
        t_mean=("t_mean", "mean"),
    )

    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)

    df = out.merge(feats, on="iso3").merge(centroids[["iso3", "abs_lat"]], on="iso3")
    df = df.merge(clust[["iso3", "cluster"]], on="iso3", how="left")
    print(f"Sample: {len(df)} countries with all variables")

    # Pathway dummies, joined to df
    pw = pd.get_dummies(df["cluster"], prefix="pw", drop_first=True).astype(float)
    df = pd.concat([df, pw], axis=1)
    pw_cols = list(pw.columns)

    specs = [
        ("σᵥ → log pop growth", "log_pop_growth", "sigma_v_real"),
        ("σₛ → Δ crop share",   "d_crop_share",   "sigma_s_real"),
    ]
    print()
    for name, outcome, predictor in specs:
        sub = df.dropna(subset=[outcome, predictor, "abs_lat"]).copy()
        print(f"=== {name} ===")
        for label, controls in [
            ("baseline",              []),
            ("+ |lat|",               ["abs_lat"]),
            ("+ |lat| + t_mean",      ["abs_lat", "t_mean"]),
            ("+ |lat| + pathway FE",  ["abs_lat"] + pw_cols),
        ]:
            X = sm.add_constant(sub[[predictor] + controls].fillna(0))
            y = sub[outcome]
            r = sm.OLS(y, X).fit(cov_type="HC1")
            b = r.params[predictor]; p = r.pvalues[predictor]; se = r.bse[predictor]
            print(f"  {label:<25s} β = {b:+.4f}  SE = {se:.4f}  p = {p:.4g}  R²={r.rsquared:.3f}  N={int(r.nobs)}")
        print()


if __name__ == "__main__":
    main()
