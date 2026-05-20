"""Two appendix tables for the GS-climate long-shadow extension.

Table A1: GS-decomposition sensitivity battery (reshape of Task 4 output)
Table A2: Modern-window GS placebo (recompute mask on 1950-2008)

Spec: docs/superpowers/specs/2026-05-19-growing-season-volatility-design.md §5.4
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


# === Table A1: reshape Task 4 output into a wide-format appendix table ===

def build_table_a1() -> pd.DataFrame:
    sens = pd.read_parquet(DATA / "long_shadow_sensitivity_gs.parquet")

    # Normalize regressor names so all weightings share a common short label
    def _short(r):
        if r == "sigma_v_preind_annual": return "annual"
        if "sigma_v_T_gs_" in r:        return "T_GS"
        if "sigma_v_T_nongs_" in r:     return "T_nonGS"
        if "sigma_v_P_gs_" in r:        return "P_GS"
        return r

    sens["short"] = sens["regressor"].apply(_short)

    # Pivot: index = (weighting, sample), columns = short, values = (beta, p, n)
    rows = []
    for (w, s), grp in sens.groupby(["weighting", "sample"]):
        row = {"weighting": w, "sample": s}
        for _, r in grp.iterrows():
            short = r["short"]
            row[f"{short}_beta"] = r["beta"]
            row[f"{short}_p"]    = r["p"]
            row[f"{short}_n"]    = r["n"]
        row["r2"] = grp["r2"].max()   # nearly constant across single-regressor specs
        rows.append(row)

    tab = pd.DataFrame(rows)
    out = DATA / "long_shadow_appendix_table_gs_decomp.parquet"
    tab.to_parquet(out, index=False)
    print(f"[A1] wrote {out}  shape={tab.shape}")
    print(tab[["weighting", "sample",
               "annual_beta", "T_GS_beta", "T_nonGS_beta", "P_GS_beta"]
              ].round(3).to_string(index=False))
    return tab


# === Table A2: modern-window GS placebo ===

def build_table_a2() -> pd.DataFrame:
    """Recompute GS mask on 1950-2008 climatology and run the headline regression."""
    from analysis.paper4_shadow.build_gs_climate import _absolute_levels

    # Load crop-weighted monthly data (covers 1421-2008)
    mod = pd.read_parquet(DATA / "modera_country_monthly_cropw.parquet")

    # Recover absolute T and P via CRU 1901-1950 climatology
    abs_df = _absolute_levels(mod, entity_col="iso3")

    # Filter to modern window
    abs_modern = abs_df[abs_df["year"].between(1950, 2008)].copy()

    # --- Mask on modern-window climatology (1950-2008 mean per iso3 × month) ---
    clim = abs_modern.groupby(["iso3", "month"], as_index=False).agg(
        t_clim=("t_abs", "mean"),
        p_clim=("p_abs", "mean"))
    clim["in_gs"] = ((clim["t_clim"] >= 5) & (clim["t_clim"] <= 30)
                     & (clim["p_clim"] >= 30))
    mask = clim[["iso3", "month", "in_gs"]]

    # --- GS-restricted annual mean T/P per (iso3, year) for modern window ---
    merged = abs_modern.merge(mask, on=["iso3", "month"])
    gs = merged[merged["in_gs"]]

    gs_yr = gs.groupby(["iso3", "year"], as_index=False).agg(
        t_gs=("t_abs", "mean"),
        p_gs=("p_abs", "mean"))

    cs_modern = gs_yr.groupby("iso3", as_index=False).agg(
        sigma_v_T_gs_modern=("t_gs", "std"),
        sigma_v_P_gs_modern=("p_gs", "std"))

    # --- Build outcome: log(pop_2015-2025 / pop_1950-1960) ---
    modern_panel = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")

    p0 = (modern_panel[modern_panel["year"].between(1950, 1960)]
            .groupby("iso3", as_index=False).agg(p0=("pop", "mean")))
    p1 = (modern_panel[modern_panel["year"].between(2015, 2025)]
            .groupby("iso3", as_index=False).agg(p1=("pop", "mean")))
    outcome = (p0.merge(p1, on="iso3")
                 .query("p0 > 0 and p1 > 0")
                 .copy())
    outcome["log_pop_growth"] = np.log(outcome["p1"] / outcome["p0"])

    # --- Covariates: |lat|, pathway cluster ---
    cov = (modern_panel.dropna(subset=["centroid_lat"])
                       .groupby("iso3", as_index=False)
                       .agg(centroid_lat=("centroid_lat", "first")))
    cov["abs_lat"] = cov["centroid_lat"].abs()

    pathways = pd.read_parquet(DATA / "paper1_clustered_features.parquet")

    # --- Pre-industrial GS (headline) ---
    pre = pd.read_parquet(DATA / "country_seasonality_gs_preindustrial.parquet")

    # --- Assemble cross-section data frame ---
    df = (outcome[["iso3", "log_pop_growth"]]
          .merge(cov[["iso3", "abs_lat"]], on="iso3", how="inner")
          .merge(pathways[["iso3", "cluster"]], on="iso3", how="left")
          .merge(cs_modern, on="iso3", how="left")
          .merge(pre[["iso3", "sigma_v_T_gs_pre1750_cropw",
                       "sigma_v_P_gs_pre1750_cropw"]], on="iso3", how="left"))

    print(f"[A2] cross-section N={len(df)}; "
          f"modern GS available: {df['sigma_v_T_gs_modern'].notna().sum()}, "
          f"pre-ind GS available: {df['sigma_v_T_gs_pre1750_cropw'].notna().sum()}")

    # --- Fit helper (mirrors Task 3 headline: HC1 + |lat| + pathway FE) ---
    def _fit(key: str) -> dict:
        sub = df[[key, "log_pop_growth", "abs_lat", "cluster"]].dropna()
        if len(sub) < 5:
            return {"beta": np.nan, "se": np.nan, "p": np.nan, "r2": np.nan, "n": 0}
        dums = pd.get_dummies(sub["cluster"], prefix="pw", drop_first=True, dtype=float)
        X_cols = [key, "abs_lat"] + list(dums.columns)
        sub = pd.concat([sub.reset_index(drop=True), dums.reset_index(drop=True)], axis=1)
        X = sm.add_constant(sub[X_cols].astype(float))
        r = sm.OLS(sub["log_pop_growth"], X).fit(cov_type="HC1")
        return {"beta":  float(r.params[key]),
                "se":    float(r.bse[key]),
                "p":     float(r.pvalues[key]),
                "r2":    float(r.rsquared),
                "n":     int(r.nobs)}

    rows = [
        {"window": "1421-1750 (headline)",
         "regressor": "sigma_v_T_GS_cropw",
         **_fit("sigma_v_T_gs_pre1750_cropw")},
        {"window": "1421-1750 (headline)",
         "regressor": "sigma_v_P_GS_cropw",
         **_fit("sigma_v_P_gs_pre1750_cropw")},
        {"window": "1950-2008 (modern placebo)",
         "regressor": "sigma_v_T_GS_modern_cropw",
         **_fit("sigma_v_T_gs_modern")},
        {"window": "1950-2008 (modern placebo)",
         "regressor": "sigma_v_P_GS_modern_cropw",
         **_fit("sigma_v_P_gs_modern")},
    ]
    tab = pd.DataFrame(rows)

    out = DATA / "long_shadow_appendix_table_gs_modern_placebo.parquet"
    tab.to_parquet(out, index=False)
    print(f"\n[A2] wrote {out}")
    print(tab.to_string(index=False))
    return tab


def main() -> None:
    print("=== Building appendix tables for GS extension ===\n")
    build_table_a1()
    print()
    build_table_a2()


if __name__ == "__main__":
    main()
