"""Sub-national Stage 1 robustness using the pop-weighted sub-national ModE-RA panel."""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def build_features(mod_path: Path) -> pd.DataFrame:
    mod = pd.read_parquet(mod_path)
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    pre = df[df["year"].between(1421, 1750)].copy()
    pre["prod"] = ((pre["t_abs"] >= 5) & (pre["t_abs"] <= 30)
                   & (pre["p_abs"] >= 30)).astype(np.int8)
    yr = pre.groupby(["sub_id", "iso3", "year"], as_index=False).agg(
        prod_months=("prod", "sum"))
    return yr.groupby(["sub_id", "iso3"], as_index=False).agg(
        productive_months=("prod_months", "mean"))


def fe_reg(df: pd.DataFrame, outcome: str, predictor: str) -> dict:
    d = df.dropna(subset=[outcome, predictor, "iso3"]).copy()
    counts = d["iso3"].value_counts()
    d = d[d["iso3"].isin(counts[counts >= 2].index)]
    if len(d) < 30: return {}
    g = d.groupby("iso3")
    d[outcome] = d[outcome] - g[outcome].transform("mean")
    d[predictor] = d[predictor] - g[predictor].transform("mean")
    X = sm.add_constant(d[[predictor]])
    y = d[outcome]
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    return {"beta": r.params[predictor], "se": r.bse[predictor],
            "p": r.pvalues[predictor], "n": int(r.nobs)}


def main() -> None:
    hyde = pd.read_parquet(DATA / "subnational_hyde.parquet")
    h = hyde[hyde["year"] == 1750].copy()
    h["cropland_ha"] = h["cropland_ha"].fillna(0)
    h["grazing_ha"] = h["grazing_ha"].fillna(0)
    h["ag_total"] = h["cropland_ha"] + h["grazing_ha"]
    h["crop_share"] = np.where(h["ag_total"]>0, h["cropland_ha"]/h["ag_total"], np.nan)
    h["log_density"] = np.log((h["subpop"] / h["ag_total"].clip(lower=1)).clip(lower=1e-6))

    for label, p in [("area-weighted", DATA / "modera_subnational_monthly.parquet"),
                      ("pop-weighted",  DATA / "modera_subnational_monthly_popw.parquet")]:
        print(f"\n=== {label} ===")
        feats = build_features(p)
        df = feats.merge(h[["sub_id", "iso3", "crop_share", "log_density", "ag_total"]],
                          on=["sub_id", "iso3"], how="inner")
        df = df[df["ag_total"] > 0]
        for outcome in ["crop_share", "log_density"]:
            r = fe_reg(df, outcome, "productive_months")
            if r:
                print(f"  {outcome:>12s} ~ productive_months (country FE): "
                      f"β = {r['beta']:+.4f}  SE = {r['se']:.4f}  p = {r['p']:.4g}  N = {r['n']}")


if __name__ == "__main__":
    main()
