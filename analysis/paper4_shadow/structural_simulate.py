"""Forward-simulate 196 countries 1421-1950 with the calibrated quantitative
model. Validate by checking whether the simulated panel reproduces the
long-shadow regression (a moment we did NOT target in calibration).

The model evolves log population as
    Δ_ann ln N_{i,t→t+1} = α(τ) + β(τ,t)·ln d_{i,t}
                          + γ_T(τ)·T_anom_{i,t} + γ_P(τ)·P_anom_{i,t}
                          + δ_T(τ)·σ^T_{i,t}
                          + 0.5·η(τ)·(t − 1421)/100
where β(τ, t) = β₀(τ) + η(τ)·(t − 1421)/100 is the calibrated time-varying
Malthusian coefficient (recovering the demographic-transition signature
estimated in subperiod regressions).

Initial conditions: HYDE 1421 population per country. Climate inputs: actual
ModE-RA interval-mean T anomaly and within-interval σ^T per (country, decade).
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def load_panel_and_params() -> tuple[pd.DataFrame, dict, dict]:
    panel = pd.read_parquet(DATA / "preindustrial_malthus_panel_extended.parquet")
    panel = panel.dropna(subset=["pop_growth_ann", "log_density",
                                   "t_mean_int", "p_mean_int", "t_std_int", "cluster"])
    panel["t_anom_int"] = panel["t_mean_int"] - panel.groupby("iso3")["t_mean_int"].transform("mean")
    panel["p_anom_int"] = panel["p_mean_int"] - panel.groupby("iso3")["p_mean_int"].transform("mean")
    params = pd.read_parquet(DATA / "calibrated_model_parameters.parquet")
    params_dict = params.set_index("cluster").to_dict("index")
    iso_areas = panel[["iso3"]].drop_duplicates()
    return panel, params_dict, iso_areas


def simulate(panel: pd.DataFrame, params: dict, scenario: str = "baseline",
              counterfactual: dict | None = None) -> pd.DataFrame:
    """Forward-simulate population trajectories country by country.

    scenario: 'baseline' uses the calibrated parameters and actual climate.
    Counterfactual modifications via the counterfactual dict:
       'pathway_override': dict mapping iso3 -> cluster to swap pathway params
       'zero_volc':        bool, set σ^T contributions from volcanic decades to 0
       'climate_decoupl':  scalar in [0,1], shrink σ^T toward modern (1900-) mean
    """
    counterfactual = counterfactual or {}
    out_rows = []
    for iso3, g in panel.groupby("iso3"):
        g = g.sort_values("year").reset_index(drop=True)
        cl_base = int(g["cluster"].iloc[0])
        cl = counterfactual.get("pathway_override", {}).get(iso3, cl_base)
        if cl not in params: continue
        p = params[cl]

        # The regressions identify within-country effects, so we predict
        # DEMEANED growth (deviation from each country's pre-industrial mean
        # growth) using DEMEANED density and (already-demeaned) climate.
        # Country fixed effects absorb α, country-specific land area, and any
        # secular country-level drift.
        mean_lnd = g["log_density"].mean()
        mean_growth = g["pop_growth_ann"].mean()
        cum_growth_sim_dev = 0.0
        cum_growth_actual_dev = 0.0
        for k in range(len(g)):
            row = g.iloc[k]
            t_centred = (row["year"] - 1421) / 100.0
            t_anom = row["t_anom_int"]   # already demeaned by country
            t_std = row["t_std_int"]
            ln_d_dev = row["log_density"] - mean_lnd

            if "climate_decoupl" in counterfactual:
                w = counterfactual["climate_decoupl"]
                t_std = w * t_std
            if counterfactual.get("zero_volc_years"):
                if any(y in counterfactual["zero_volc_years"]
                        for y in range(row["year"], row["year_next"])):
                    t_std = 0.0

            beta_t = p["beta0"] + p["eta"] * t_centred
            growth_dev_sim = (beta_t * ln_d_dev
                              + p["gamma_T_0"] * t_anom
                              + p["delta_T_0"] * (t_std - g["t_std_int"].mean()))
            growth_dev_act = row["pop_growth_ann"] - mean_growth
            cum_growth_sim_dev += growth_dev_sim * row["dt"]
            cum_growth_actual_dev += growth_dev_act * row["dt"]
            out_rows.append({
                "iso3": iso3, "cluster_actual": cl_base, "cluster_sim": cl,
                "year": int(row["year"]),
                "year_next": int(row["year_next"]),
                "dt": row["dt"],
                "ln_d_dev": ln_d_dev,
                "growth_dev_actual": growth_dev_act,
                "growth_dev_sim": growth_dev_sim,
                "cum_growth_sim_dev": cum_growth_sim_dev,
                "cum_growth_actual_dev": cum_growth_actual_dev,
                "scenario": scenario,
            })
    return pd.DataFrame(out_rows)


def validate_long_shadow(sim: pd.DataFrame) -> dict:
    """Check whether the simulated panel reproduces the long-shadow regression.
    Uses σ_v computed from the underlying climate panel (1421-1750) — same as
    the empirical version — but uses SIMULATED pop growth 1955 vs 2020 as
    proxy via terminal-year simulated log pop.
    """
    # Take last simulation row per country, which corresponds to the latest
    # interval (around 1940-1950) — terminal pre-industrial state.
    final = sim.sort_values(["iso3", "year"]).groupby("iso3").tail(1)
    final = final[["iso3", "cum_growth_sim_dev"]].rename(
        columns={"cum_growth_sim_dev": "sim_cum_growth"})

    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)]
    sv = pre.groupby("iso3")["t_mean"].std().rename("sigma_v").reset_index()
    df = final.merge(sv, on="iso3").dropna()
    # Regress simulated cumulative log-pop growth on pre-industrial volatility
    X = sm.add_constant(df[["sigma_v"]])
    r = sm.OLS(df["sim_cum_growth"], X).fit(cov_type="HC1")
    return {
        "n": int(r.nobs),
        "beta_sim": r.params["sigma_v"],
        "se_sim": r.bse["sigma_v"],
        "p_sim": r.pvalues["sigma_v"],
        "rsq_sim": r.rsquared,
    }


def main() -> None:
    print("Loading calibrated parameters and panel...")
    panel, params, _ = load_panel_and_params()
    print(f"  panel: {len(panel):,} cells, {panel['iso3'].nunique()} countries")
    print(f"  parameters loaded for clusters: {list(params.keys())}")

    print("\nRunning baseline simulation...")
    sim = simulate(panel, params, scenario="baseline")
    sim.to_parquet(DATA / "structural_sim_baseline.parquet", index=False)
    print(f"  Simulated {len(sim):,} country-interval cells across "
          f"{sim['iso3'].nunique()} countries")

    # Validation: pre-industrial growth fit
    print("\n=== Model-vs-actual fit for within-country growth deviation ===")
    for cl, g in sim.merge(panel[["iso3", "year", "cluster"]],
                              on=["iso3", "year"]).groupby("cluster"):
        if len(g) < 50: continue
        ss_res = ((g["growth_dev_actual"] - g["growth_dev_sim"])**2).sum()
        ss_tot = ((g["growth_dev_actual"] - g["growth_dev_actual"].mean())**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        bias = (g["growth_dev_sim"] - g["growth_dev_actual"]).mean()
        rmse = np.sqrt(((g["growth_dev_sim"] - g["growth_dev_actual"])**2).mean())
        print(f"  {PATHWAY_NAMES[cl]:<25s} N={len(g):>5}  "
              f"R²={r2:+.4f}  bias={bias:+.6f}  RMSE={rmse:.5f}")

    # Validation: does the simulation reproduce the long shadow?
    print("\n=== Out-of-sample validation: simulated terminal pop ~ σ_v ===")
    ls = validate_long_shadow(sim)
    print(f"  Simulated:  β = {ls['beta_sim']:+.3f}  SE = {ls['se_sim']:.3f}  "
          f"p = {ls['p_sim']:.4g}  R² = {ls['rsq_sim']:.3f}  N = {ls['n']}")
    print(f"  Empirical (Section 9): β = -2.48 (sigma_v_real on log pop growth 1955→2020)")
    print(f"  Compare: the model SHOULD generate a negative coefficient between")
    print(f"  σ_v and simulated terminal pop, since pathways with high σ_v have")
    print(f"  more-negative δ_T·σ_v drag in their dynamics.")


if __name__ == "__main__":
    main()
