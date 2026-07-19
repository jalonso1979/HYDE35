"""Welfare counterfactuals: translate population effects from the structural
counterfactuals into real-wage / welfare units using the Allen wage Malthus
regression coefficients (Section 8 of the paper).

The Allen Malthus regression (1421-1850, 6 European countries, country FE):

    Δ ln real_wage_{it} = β · ln pop_{i,t-5} + γ_T · T_anom_{it} + γ_P · P_anom_{it} + α_i

with β = -0.29 (p=0.08), γ_T = +0.046 (p<10⁻³), γ_P ≈ 0.

For each structural counterfactual we compute:

  Δ_wage_per_cap = β · Δ ln pop_cumulative + γ_T · Δ T_anom_cumulative
  Δ_welfare_total = Δ ln pop + Δ ln wage  (utilitarian aggregate, log form)

Per-capita wage change tells us about MALTHUSIAN welfare; total welfare is
the utilitarian sum of head-count and per-head welfare. The Allen panel is
European only, so this exercise is meaningfully informative for the
intensive pathway (which includes France, Germany, Netherlands), the early
extensifiers (Britain), and partly for the crop-dominant pathway (Spain).
We are extrapolating heroically when applying these coefficients to the
pastoral pathway (China, Brazil, etc.).
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

# Allen Malthus coefficients from Section 8 (Spec 3, joint regression)
BETA_LOG_POP = -0.285      # density coefficient: doubling pop → -28.5% wage
GAMMA_T = 0.0321           # temperature anomaly: +1°C → +3.2% wage
GAMMA_P = 0.00013          # precipitation: essentially zero
# Standard errors for uncertainty:
BETA_SE = 0.164
GAMMA_T_SE = 0.0095

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}

# Allen panel coverage by pathway (which countries are IN the Allen sample)
ALLEN_COUNTRIES = {"BEL", "ESP", "FRA", "GBR", "ITA", "NLD"}


def compute_cf1_welfare() -> pd.DataFrame:
    """CF1: pastoral → intensive reassignment. Translate cumulative pop lift
    into real-wage effect via β·Δ ln pop. There is no T shift in CF1, only
    Δ ln pop, so γ_T·ΔT = 0."""
    base = pd.read_parquet(DATA / "structural_sim_baseline.parquet")
    cf = pd.read_parquet(DATA / "structural_sim_cf_intensive.parquet")
    bf = (base[base["year"] < 1750].sort_values(["iso3", "year"])
           .groupby("iso3").tail(1)[["iso3", "cum_growth_sim_dev"]]
           .rename(columns={"cum_growth_sim_dev": "cum_base"}))
    cff = (cf[cf["year"] < 1750].sort_values(["iso3", "year"])
            .groupby("iso3").tail(1)[["iso3", "cum_growth_sim_dev"]]
            .rename(columns={"cum_growth_sim_dev": "cum_cf"}))
    cmp = bf.merge(cff, on="iso3")
    panel = pd.read_parquet(DATA / "preindustrial_malthus_panel_extended.parquet")
    cmp = cmp.merge(panel[["iso3", "cluster"]].drop_duplicates(), on="iso3")
    pastoral = cmp[cmp["cluster"] == 1].copy()
    pastoral["dlnN"] = pastoral["cum_cf"] - pastoral["cum_base"]
    pastoral["d_wage_pc"] = BETA_LOG_POP * pastoral["dlnN"]
    pastoral["d_welfare_total"] = pastoral["dlnN"] + pastoral["d_wage_pc"]
    pastoral["in_allen_sample"] = pastoral["iso3"].isin(ALLEN_COUNTRIES)
    return pastoral


def compute_cf2_welfare() -> pd.DataFrame:
    """CF2: no 1500-1900 volcanism. Two channels:
       (i) population effect from σ_T elimination: β · Δ ln pop
       (ii) direct climate effect from removing eruption-year cooling:
             eruption years drop T by ~0.3°C cumulatively over the period.
       We approximate the direct cooling burden as 11 eruptions × 3-year
       impact × -0.3°C per year = -10°C-years cumulative T-anom shift.
       Counterfactual: T_anom_cumulative goes UP by ~10°C-years.
       Per-year direct wage gain = γ_T × 0.3°C × (11×3/N_years).
    """
    base = pd.read_parquet(DATA / "structural_sim_baseline.parquet")
    cf = pd.read_parquet(DATA / "structural_sim_cf_no_volc.parquet")
    bf = (base[base["year"].between(1500, 1900)]
           .sort_values(["iso3", "year"]).groupby("iso3").tail(1)
           [["iso3", "cum_growth_sim_dev"]]
           .rename(columns={"cum_growth_sim_dev": "cum_base"}))
    cff = (cf[cf["year"].between(1500, 1900)]
            .sort_values(["iso3", "year"]).groupby("iso3").tail(1)
            [["iso3", "cum_growth_sim_dev"]]
            .rename(columns={"cum_growth_sim_dev": "cum_cf"}))
    cmp = bf.merge(cff, on="iso3")
    panel = pd.read_parquet(DATA / "preindustrial_malthus_panel_extended.parquet")
    cmp = cmp.merge(panel[["iso3", "cluster"]].drop_duplicates(), on="iso3")
    cmp["dlnN"] = cmp["cum_cf"] - cmp["cum_base"]
    # Approximate direct climate channel: 11 eruptions × 3-year post-window
    # × ~0.3°C cooling each = a cumulative T_anom shift of about +10°C·years
    # spread over a 400-year window (1500-1900). Per-year average T anomaly
    # uplift = 10 / 400 ≈ 0.025°C. We use the mean Allen γ_T coefficient.
    # The effect on the AVERAGE wage over 400 years is γ_T × 0.025
    # ≈ 0.046 × 0.025 = 0.001 log-units per year averaged — small.
    AVG_T_UPLIFT = 0.025  # °C, averaged across 400 years
    cmp["d_wage_pc"] = (BETA_LOG_POP * cmp["dlnN"]
                        + GAMMA_T * AVG_T_UPLIFT)
    cmp["d_welfare_total"] = cmp["dlnN"] + cmp["d_wage_pc"]
    cmp["in_allen_sample"] = cmp["iso3"].isin(ALLEN_COUNTRIES)
    return cmp


def compute_cf3_welfare(weight: float = 0.0) -> pd.DataFrame:
    """CF3: climate decoupling. At weight=0 σ_T contributions are zero;
    at weight=1 baseline. Translate cumulative pop deviation into wages."""
    cf = pd.read_parquet(DATA / "structural_sim_cf_decoupling.parquet")
    cf = cf[cf["weight"] == weight]
    base_w = pd.read_parquet(DATA / "structural_sim_cf_decoupling.parquet")
    base_w = base_w[base_w["weight"] == 1.0].rename(
        columns={"cum_growth_sim_dev": "cum_base"})
    cmp = cf[["iso3", "cluster", "cum_growth_sim_dev"]].merge(
        base_w[["iso3", "cum_base"]], on="iso3")
    cmp["dlnN"] = cmp["cum_growth_sim_dev"] - cmp["cum_base"]
    cmp["d_wage_pc"] = BETA_LOG_POP * cmp["dlnN"]
    cmp["d_welfare_total"] = cmp["dlnN"] + cmp["d_wage_pc"]
    cmp["in_allen_sample"] = cmp["iso3"].isin(ALLEN_COUNTRIES)
    return cmp


def summarise(df: pd.DataFrame, label: str) -> None:
    if "cluster" in df:
        for cl, g in df.groupby("cluster"):
            if cl not in PATHWAY_NAMES: continue
            n = len(g)
            n_allen = g["in_allen_sample"].sum() if "in_allen_sample" in g else 0
            med_dlnN = g["dlnN"].median()
            med_dwage = g["d_wage_pc"].median()
            med_dwelf = g["d_welfare_total"].median()
            mark = "[Allen sample present]" if n_allen > 0 else "[extrapolated]"
            print(f"    {PATHWAY_NAMES[cl]:<25s} N={n:>3} {mark}  "
                  f"Δln N = {med_dlnN:+.3f} ({(np.exp(med_dlnN)-1)*100:+.1f}%)  "
                  f"Δ wage/cap = {med_dwage:+.3f} ({(np.exp(med_dwage)-1)*100:+.1f}%)  "
                  f"Δ welfare = {med_dwelf:+.3f}")
    else:
        med_dlnN = df["dlnN"].median()
        med_dwage = df["d_wage_pc"].median()
        med_dwelf = df["d_welfare_total"].median()
        print(f"    N={len(df)}  Δln N = {med_dlnN:+.3f} "
              f"Δ wage/cap = {med_dwage:+.3f}  Δ welfare = {med_dwelf:+.3f}")


def main() -> None:
    print("=== Welfare counterfactuals via Allen wage Malthus coefficients ===")
    print(f"Coefficients: β_log_pop = {BETA_LOG_POP}, γ_T = {GAMMA_T}")
    print(f"Allen sample: {sorted(ALLEN_COUNTRIES)}")
    print()

    print("CF1 — Pastoral reassigned to intensive parameters")
    cf1 = compute_cf1_welfare()
    summarise(cf1, "CF1")

    print("\nCF2 — No 1500-1900 volcanism (eruption-year σ_T set to zero)")
    cf2 = compute_cf2_welfare()
    summarise(cf2, "CF2")

    print("\nCF3 — Full climate decoupling (σ_T → 0)")
    cf3 = compute_cf3_welfare(weight=0.0)
    summarise(cf3, "CF3")

    cf1.to_parquet(DATA / "welfare_cf1.parquet", index=False)
    cf2.to_parquet(DATA / "welfare_cf2.parquet", index=False)
    cf3.to_parquet(DATA / "welfare_cf3.parquet", index=False)

    # Total/utilitarian welfare summary table
    print("\n=== Summary: per-pathway median welfare effects ===")
    print(f"{'CF':<5} {'Pathway':<25} {'Δ ln N':>10} {'Δ wage/cap':>14} "
          f"{'Δ welfare':>12}")
    for cf, label in [(cf1, "CF1"), (cf2, "CF2"), (cf3, "CF3")]:
        for cl in sorted(set(cf["cluster"]) & set(PATHWAY_NAMES.keys())):
            g = cf[cf["cluster"] == cl]
            if len(g) == 0: continue
            print(f"{label:<5} {PATHWAY_NAMES[cl]:<25} "
                  f"{g['dlnN'].median():>+10.3f} "
                  f"{(np.exp(g['d_wage_pc'].median())-1)*100:>+13.1f}% "
                  f"{g['d_welfare_total'].median():>+12.3f}")


if __name__ == "__main__":
    main()
