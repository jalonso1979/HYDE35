"""GS-headline long-shadow cross-section: §5 of paper/long_shadow.tex.

Runs the headline regression of modern (1950-1960 → 2015-2025) log
population growth on pre-industrial 1421-1750 growing-season-restricted
volatility measures, with non-GS placebo as a side-by-side column, and
prints the three pre-registered predictions (PASS/FAIL).

Spec: docs/superpowers/specs/2026-05-19-growing-season-volatility-design.md §5.1

Join note: paper1_clustered_features.parquet has iso3 as alpha codes (AFG, AGO…)
for 154/157 rows; the 3 numeric rows (530=ANT, 736=SDN-old, 891=YUG) are
dissolved entities absent from the modern panel and dropped on the left join.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def _build_outcome(modern: pd.DataFrame) -> pd.DataFrame:
    """One row per iso3 with log_pop_growth from 1950-1960 baseline to 2015-2025."""
    p0 = (modern[modern["year"].between(1950, 1960)]
            .groupby("iso3", as_index=False).agg(p0=("pop", "mean")))
    p1 = (modern[modern["year"].between(2015, 2025)]
            .groupby("iso3", as_index=False).agg(p1=("pop", "mean")))
    out = p0.merge(p1, on="iso3").query("p0 > 0 and p1 > 0").copy()
    out["log_pop_growth"] = np.log(out["p1"] / out["p0"])
    return out[["iso3", "log_pop_growth"]]


def _build_covariates(modern: pd.DataFrame) -> pd.DataFrame:
    """One row per iso3 with country_id and abs_lat for pathway/lat merges."""
    cov = (modern.dropna(subset=["centroid_lat"])
                 .groupby("iso3", as_index=False)
                 .agg(country_id=("country_id", "first"),
                      centroid_lat=("centroid_lat", "first")))
    cov["abs_lat"] = cov["centroid_lat"].abs()
    return cov[["iso3", "country_id", "abs_lat"]]


def _fit(d: pd.DataFrame, key: str, lat: bool, fe: bool) -> dict:
    """OLS log_pop_growth ~ key [+ abs_lat] [+ pathway FE]. HC1 SEs."""
    cols = [key, "log_pop_growth"]
    if lat: cols.append("abs_lat")
    if fe:  cols.append("cluster")
    sub = d[cols].dropna()
    if len(sub) < 5:
        return {"key": key, "lat": lat, "fe": fe, "beta": np.nan,
                "se": np.nan, "p": np.nan, "r2": np.nan, "n": 0}
    rhs = [key]
    if lat: rhs.append("abs_lat")
    if fe:
        dums = pd.get_dummies(sub["cluster"], prefix="pw", drop_first=True, dtype=float)
        sub = pd.concat([sub, dums], axis=1)
        rhs.extend(dums.columns.tolist())
    X = sm.add_constant(sub[rhs].astype(float))
    r = sm.OLS(sub["log_pop_growth"], X).fit(cov_type="HC1")
    return {"key": key, "lat": lat, "fe": fe,
            "beta": float(r.params[key]), "se": float(r.bse[key]),
            "p": float(r.pvalues[key]), "r2": float(r.rsquared),
            "n": int(r.nobs)}


def _fit_multi(d: pd.DataFrame, keys: list[str], lat: bool, fe: bool) -> dict:
    """OLS with multiple regressors; returns dict keyed by regressor name."""
    cols = list(keys) + ["log_pop_growth"]
    if lat: cols.append("abs_lat")
    if fe:  cols.append("cluster")
    sub = d[cols].dropna()
    if len(sub) < 5:
        return {k: {"beta": np.nan, "p": np.nan} for k in keys}
    rhs = list(keys)
    if lat: rhs.append("abs_lat")
    if fe:
        dums = pd.get_dummies(sub["cluster"], prefix="pw", drop_first=True, dtype=float)
        sub = pd.concat([sub, dums], axis=1)
        rhs.extend(dums.columns.tolist())
    X = sm.add_constant(sub[rhs].astype(float))
    r = sm.OLS(sub["log_pop_growth"], X).fit(cov_type="HC1")
    out = {k: {"beta": float(r.params[k]), "se": float(r.bse[k]),
               "p": float(r.pvalues[k])} for k in keys}
    out["_meta"] = {"r2": float(r.rsquared), "n": int(r.nobs)}
    return out


def run_headline_gs() -> dict:
    print("[gs-headline] loading data...", flush=True)
    modern = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    gs = pd.read_parquet(DATA / "country_seasonality_gs_preindustrial.parquet")
    annual = pd.read_parquet(DATA / "country_seasonality_preindustrial.parquet")
    pathways = pd.read_parquet(DATA / "paper1_clustered_features.parquet")

    # pathways iso3 is alpha for 154/157 rows; merge directly on iso3.
    # The 3 numeric rows (dissolved entities) will be NaN cluster and dropped
    # when pathway FE are added — that is correct behaviour.
    pathway_cluster = pathways[["iso3", "cluster"]].copy()

    outcome = _build_outcome(modern)
    cov = _build_covariates(modern)

    df = (outcome.merge(cov, on="iso3", how="inner")
                 .merge(pathway_cluster, on="iso3", how="left")
                 .merge(gs[["iso3", "n_gs_months_cropw",
                             "sigma_v_T_gs_pre1750_cropw",
                             "sigma_v_P_gs_pre1750_cropw",
                             "sigma_v_T_nongs_pre1750_cropw"]],
                        on="iso3", how="left")
                 .merge(annual[["iso3", "sigma_v_preind"]], on="iso3", how="left"))

    # Headline sample: drop empty-GS countries
    headline = df[df["n_gs_months_cropw"] > 0].copy()
    print(f"[gs-headline] headline sample: N={len(headline)} "
          f"(drops {len(df)-len(headline)} empty-GS countries)")

    # 1) Joint GS spec (the headline)
    joint = _fit_multi(headline,
                       ["sigma_v_T_gs_pre1750_cropw",
                        "sigma_v_P_gs_pre1750_cropw",
                        "sigma_v_T_nongs_pre1750_cropw"],
                       lat=True, fe=True)

    # 2) GS T-only (single regressor, same sample)
    gs_t_only = _fit(headline, "sigma_v_T_gs_pre1750_cropw", lat=True, fe=True)

    # 3) Annual σ_v^T comparison (same sample)
    annual_same_sample = _fit(headline, "sigma_v_preind", lat=True, fe=True)

    summary = {
        "joint":  joint,
        "gs_t_only":  gs_t_only,
        "annual_same_sample": annual_same_sample,
        "sample_n": int(len(headline)),
    }

    # Persist
    rows = [
        {"spec": "annual_same_sample", "key": "sigma_v_preind",
         **{k: annual_same_sample[k] for k in ("beta","se","p","r2","n")}},
        {"spec": "gs_t_only", "key": "sigma_v_T_gs_pre1750_cropw",
         **{k: gs_t_only[k] for k in ("beta","se","p","r2","n")}},
        {"spec": "joint", "key": "sigma_v_T_gs_pre1750_cropw",
         **joint["sigma_v_T_gs_pre1750_cropw"],
         "r2": joint["_meta"]["r2"], "n": joint["_meta"]["n"]},
        {"spec": "joint", "key": "sigma_v_P_gs_pre1750_cropw",
         **joint["sigma_v_P_gs_pre1750_cropw"],
         "r2": joint["_meta"]["r2"], "n": joint["_meta"]["n"]},
        {"spec": "joint", "key": "sigma_v_T_nongs_pre1750_cropw",
         **joint["sigma_v_T_nongs_pre1750_cropw"],
         "r2": joint["_meta"]["r2"], "n": joint["_meta"]["n"]},
    ]
    out_pq = DATA / "long_shadow_results_gs.parquet"
    pd.DataFrame(rows).to_parquet(out_pq, index=False)
    print(f"[gs-headline] wrote {out_pq}")
    return summary


def assert_predictions(s: dict) -> None:
    """Three pre-registered predictions from spec §7."""
    print("\n=== Pre-registered prediction checks ===")
    a = s["annual_same_sample"]
    gt = s["gs_t_only"]
    j = s["joint"]
    p_p = j["sigma_v_P_gs_pre1750_cropw"]["p"]
    p_b = j["sigma_v_P_gs_pre1750_cropw"]["beta"]
    pl_p = j["sigma_v_T_nongs_pre1750_cropw"]["p"]
    pl_b = j["sigma_v_T_nongs_pre1750_cropw"]["beta"]
    print(f"  N = {s['sample_n']} countries (empty-GS-dropped, with pathway FE)")
    print(f"P1: |β(σ_v^T_GS)|={abs(gt['beta']):.3f}  vs  "
          f"|β(σ_v^T_annual)|={abs(a['beta']):.3f}  → "
          f"{'PASS' if abs(gt['beta']) > abs(a['beta']) else 'FAIL (channel may be climate-deep, not GS-specific)'}")
    print(f"P2: β(σ_v^P_GS)={p_b:+.3f}, p={p_p:.4f}  → "
          f"{'PASS' if (p_b < 0 and p_p < 0.05) else 'FAIL (P-volatility not significant negative)'}")
    print(f"P3: β(σ_v^T_nonGS)={pl_b:+.3f}, p={pl_p:.4f}  → "
          f"{'PASS' if pl_p > 0.10 else 'FAIL (non-GS placebo significant — channel may not be agronomic)'}")
    print(f"  R² (joint, w/ |lat| + pathway FE) = {j['_meta']['r2']:.3f}")


def main() -> None:
    s = run_headline_gs()
    assert_predictions(s)


if __name__ == "__main__":
    main()
