"""Within-country Stage 1 test using sub-national variation.

If productive-months drives storage demand which drives intensive ag,
then WITHIN a country, sub-units with more productive months should
show different cropland share / population density patterns than
sub-units with fewer productive months.

Country fixed effects absorb everything that varies at the country
level (legal system, language, distance to coast, historical contingency
of state formation) and identifies the storage-demand effect off the
within-country climate gradient alone.

Two tests:
  (i)  cropland share at 1750 ~ pre-industrial productive_months + country FE
  (ii) population density at 1750 ~ pre-industrial productive_months + country FE
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"


def _compute_subnational_features() -> pd.DataFrame:
    """Pre-industrial productive months and seasonality per sub-unit."""
    mod = pd.read_parquet(DATA / "modera_subnational_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    # CRU climatology is country-level; broadcast to all sub-units of each country.
    mod = mod.merge(clim, on=["iso3", "month"], how="inner")
    mod["t_abs"] = mod["t_anom_c"] + mod["tmp_c_clim"]
    mod["p_abs"] = (mod["p_anom_mm"] + mod["pre_mm_clim"]).clip(lower=0)
    pre = mod[mod["year"].between(1421, 1750)].copy()
    pre["prod"] = ((pre["t_abs"] >= 5) & (pre["t_abs"] <= 30)
                   & (pre["p_abs"] >= 30)).astype(np.int8)

    yr = pre.groupby(["sub_id", "iso3", "year"], as_index=False).agg(
        prod_months=("prod", "sum"),
        t_max=("t_abs", "max"),
        t_min=("t_abs", "min"),
        p_total=("p_abs", "sum"),
    )
    yr["sigma_s"] = yr["t_max"] - yr["t_min"]
    feats = yr.groupby(["sub_id", "iso3"], as_index=False).agg(
        productive_months=("prod_months", "mean"),
        sigma_s=("sigma_s", "mean"),
        p_annual=("p_total", "mean"),
    )
    return feats


def _hyde_features_at(year: int) -> pd.DataFrame:
    """Per-sub-unit cropland share, density at a given HYDE timestep."""
    hyde = pd.read_parquet(DATA / "subnational_hyde.parquet")
    sub = hyde[hyde["year"] == year].copy()
    sub["cropland_ha"] = sub["cropland_ha"].fillna(0)
    sub["grazing_ha"] = sub["grazing_ha"].fillna(0)
    sub["pop"] = sub["subpop"].fillna(0)
    sub["ag_total_ha"] = sub["cropland_ha"] + sub["grazing_ha"]
    sub["crop_share"] = np.where(sub["ag_total_ha"] > 0,
                                  sub["cropland_ha"] / sub["ag_total_ha"],
                                  np.nan)
    return sub[["sub_id", "iso3", "pop", "cropland_ha", "grazing_ha",
                "ag_total_ha", "crop_share"]]


def fe_regression(df: pd.DataFrame, outcome: str, predictors: list[str],
                  fe_col: str = "iso3", min_per_group: int = 2) -> dict:
    d = df.dropna(subset=[outcome] + predictors + [fe_col]).copy()
    counts = d[fe_col].value_counts()
    d = d[d[fe_col].isin(counts[counts >= min_per_group].index)]
    if len(d) < 30:
        return {}
    g = d.groupby(fe_col)
    for c in predictors + [outcome]:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[predictors])
    y = d[outcome]
    n_groups = d[fe_col].nunique()
    if n_groups > 1:
        r = sm.OLS(y, X).fit(cov_type="cluster",
                              cov_kwds={"groups": d[fe_col].values})
    else:
        r = sm.OLS(y, X).fit(cov_type="HC1")
    return {
        "params": r.params.to_dict(), "pvalues": r.pvalues.to_dict(),
        "se": r.bse.to_dict(), "n": int(r.nobs), "rsq": r.rsquared,
        "n_groups": n_groups,
    }


def main() -> None:
    print("Building sub-national pre-industrial features...", flush=True)
    feats = _compute_subnational_features()
    print(f"  {len(feats):,} sub-units across {feats['iso3'].nunique()} countries")
    feats.to_parquet(DATA / "subnational_features.parquet", index=False)

    for yr in [1750, 1850, 1950]:
        print(f"\n=== HYDE outcomes at year {yr} ===")
        h = _hyde_features_at(yr)
        df = feats.merge(h, on=["sub_id", "iso3"], how="inner")
        df = df[df["ag_total_ha"] > 0]
        print(f"  Sample: {len(df):,} sub-units across {df['iso3'].nunique()} countries")

        for outcome in ["crop_share"]:
            # No FE (between + within variation)
            r0 = fe_regression(df.assign(no_fe=0), outcome,
                               ["productive_months"], fe_col="no_fe")
            # Country FE (within variation only)
            r_fe = fe_regression(df, outcome, ["productive_months"], fe_col="iso3")
            for tag, r in [("no FE", r0), ("country FE", r_fe)]:
                if not r:
                    continue
                b = r["params"]["productive_months"]
                p = r["pvalues"]["productive_months"]
                se = r["se"]["productive_months"]
                n = r["n"]; ng = r.get("n_groups", "all")
                print(f"  {outcome} ~ productive_months ({tag}): "
                      f"β={b:+.6f} (se={se:.6f}) p={p:.3g} "
                      f"N={n} groups={ng} R²={r['rsq']:.3f}")

        # Population density
        df["log_density"] = np.log((df["pop"] / df["ag_total_ha"].clip(lower=1)).clip(lower=1e-6))
        r0 = fe_regression(df.assign(no_fe=0), "log_density",
                           ["productive_months"], fe_col="no_fe")
        r_fe = fe_regression(df, "log_density", ["productive_months"], fe_col="iso3")
        for tag, r in [("no FE", r0), ("country FE", r_fe)]:
            if not r:
                continue
            b = r["params"]["productive_months"]
            p = r["pvalues"]["productive_months"]
            se = r["se"]["productive_months"]
            n = r["n"]; ng = r.get("n_groups", "all")
            print(f"  log_density ~ productive_months ({tag}): "
                  f"β={b:+.6f} (se={se:.6f}) p={p:.3g} "
                  f"N={n} groups={ng} R²={r['rsq']:.3f}")


if __name__ == "__main__":
    main()
