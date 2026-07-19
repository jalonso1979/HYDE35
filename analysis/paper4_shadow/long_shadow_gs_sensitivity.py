"""Decomposition sensitivity battery for the GS / non-GS / annual long-shadow exercise.

For each (weighting × sample variant), report single-regressor β, p, R²
for each of {annual σ_v^T, σ_v^T_GS, σ_v^T_nonGS, σ_v^P_GS}, on the
same outcome (modern log pop growth 1950-1960 → 2015-2025) with the
same covariates (|lat| + pathway FE, HC1 SEs).

Result: a long-format parquet for the §5 appendix decomposition table.

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

WEIGHTINGS = ["area", "pop", "cropw"]


def _build_master() -> pd.DataFrame:
    modern = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    gs = pd.read_parquet(DATA / "country_seasonality_gs_preindustrial.parquet")
    annual = pd.read_parquet(DATA / "country_seasonality_preindustrial.parquet")
    pathways = pd.read_parquet(DATA / "paper1_clustered_features.parquet")

    p0 = modern[modern["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(p0=("pop", "mean"))
    p1 = modern[modern["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(p1=("pop", "mean"))
    out = p0.merge(p1, on="iso3").query("p0 > 0 and p1 > 0").copy()
    out["log_pop_growth"] = np.log(out["p1"] / out["p0"])

    cov = (modern.dropna(subset=["centroid_lat"])
                 .groupby("iso3", as_index=False)
                 .agg(centroid_lat=("centroid_lat", "first")))
    cov["abs_lat"] = cov["centroid_lat"].abs()

    df = (out[["iso3", "log_pop_growth"]]
            .merge(cov[["iso3", "abs_lat"]], on="iso3", how="inner")
            .merge(pathways[["iso3", "cluster"]], on="iso3", how="left")
            .merge(annual[["iso3", "sigma_v_preind"]], on="iso3", how="left"))

    # All weighting-specific GS columns
    keep = ["iso3"]
    for w in WEIGHTINGS:
        keep += [f"n_gs_months_{w}",
                 f"sigma_v_T_gs_pre1750_{w}",
                 f"sigma_v_P_gs_pre1750_{w}",
                 f"sigma_v_T_nongs_pre1750_{w}"]
    df = df.merge(gs[keep], on="iso3", how="left")
    return df


def _fit_single(d: pd.DataFrame, key: str) -> dict:
    sub = d[[key, "log_pop_growth", "abs_lat", "cluster"]].dropna()
    if len(sub) < 5:
        return {"beta": np.nan, "se": np.nan, "p": np.nan,
                "r2": np.nan, "n": int(len(sub))}
    dums = pd.get_dummies(sub["cluster"], prefix="pw", drop_first=True, dtype=float)
    X_cols = [key, "abs_lat"] + list(dums.columns)
    sub = pd.concat([sub, dums], axis=1)
    X = sm.add_constant(sub[X_cols].astype(float))
    r = sm.OLS(sub["log_pop_growth"], X).fit(cov_type="HC1")
    return {"beta": float(r.params[key]), "se": float(r.bse[key]),
            "p": float(r.pvalues[key]), "r2": float(r.rsquared),
            "n": int(r.nobs)}


def run_sensitivities() -> pd.DataFrame:
    print("[gs-sensitivity] loading...", flush=True)
    master = _build_master()
    print(f"[gs-sensitivity] master sample: N={len(master)} countries")

    rows: list[dict] = []
    for w in WEIGHTINGS:
        # Three sample variants per weighting
        variants = {
            "headline_drop_empty":  master[master[f"n_gs_months_{w}"] > 0],
            "include_empty":        master,
            "drop_short_GS_le3":    master[master[f"n_gs_months_{w}"] > 3],
        }
        for variant, sub in variants.items():
            # Annual σ_v^T (same regressor across weightings — for comparison)
            r = _fit_single(sub, "sigma_v_preind")
            rows.append({"weighting": w, "sample": variant,
                          "regressor": "sigma_v_preind_annual", **r})
            # GS T
            r = _fit_single(sub, f"sigma_v_T_gs_pre1750_{w}")
            rows.append({"weighting": w, "sample": variant,
                          "regressor": f"sigma_v_T_gs_{w}", **r})
            # non-GS T placebo
            r = _fit_single(sub, f"sigma_v_T_nongs_pre1750_{w}")
            rows.append({"weighting": w, "sample": variant,
                          "regressor": f"sigma_v_T_nongs_{w}", **r})
            # GS P
            r = _fit_single(sub, f"sigma_v_P_gs_pre1750_{w}")
            rows.append({"weighting": w, "sample": variant,
                          "regressor": f"sigma_v_P_gs_{w}", **r})

    df = pd.DataFrame(rows)
    out = DATA / "long_shadow_sensitivity_gs.parquet"
    df.to_parquet(out, index=False)

    # Pretty-print: pivot for readability
    pivot = df.pivot_table(index=["weighting", "sample"],
                            columns="regressor",
                            values=["beta", "p"],
                            aggfunc="first")
    print(f"\n[gs-sensitivity] β estimates (rows: weighting × sample, cols: regressor):")
    print(pivot["beta"].round(3).to_string())
    print(f"\n[gs-sensitivity] p-values:")
    print(pivot["p"].round(4).to_string())

    # Punchline summary: how often does non-GS dominate GS in magnitude?
    print(f"\n[gs-sensitivity] PUNCHLINE: non-GS dominates GS in |β|?")
    for w in WEIGHTINGS:
        for v in ["headline_drop_empty", "include_empty", "drop_short_GS_le3"]:
            sub = df[(df["weighting"] == w) & (df["sample"] == v)]
            gs_beta = abs(sub[sub["regressor"] == f"sigma_v_T_gs_{w}"]["beta"].iloc[0])
            nongs_beta = abs(sub[sub["regressor"] == f"sigma_v_T_nongs_{w}"]["beta"].iloc[0])
            mark = "YES (nonGS > GS)" if nongs_beta > gs_beta else "NO  (GS > nonGS)"
            print(f"  {w:6s} {v:30s}  |β_GS|={gs_beta:.3f}  |β_nonGS|={nongs_beta:.3f}  → {mark}")
    print(f"\n[gs-sensitivity] wrote {out}")
    return df


def main() -> None:
    run_sensitivities()


if __name__ == "__main__":
    main()
