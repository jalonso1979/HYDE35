"""Colonial-origin partialling robustness for the functional-allele bundle.

The §3.1 interpretation of DARC (and HBB) is that the variant's negative
loading on log GDPpc captures the Acemoglu-Johnson-Robinson reversal-of-
fortune institutional channel — DARC is a West African ancestry marker
and West Africa faced colonial-extractive institutions. This script tests
the interpretation directly by adding a colonial-origin control to the
headline OLS spec and reporting how each allele coefficient changes.

Two colonial-origin indicators:
  - AJR_COLONIAL: 60 countries from Acemoglu-Johnson-Robinson (2001)
    Table 1, the canonical "extractive-colonial-origins" set.
  - EUR_COLONIZED: broader binary — was the country a European colony or
    protectorate at any point 1500-1980 (built here from a hand-coded
    list; includes North America, Australia, NZ, settler colonies as
    well as the AJR extractive set).

Predictions:
  - If DARC's negative GDPpc coefficient runs through colonial-extractive
    institutions: should attenuate substantially under EUR_COLONIZED.
  - If through ancestry-marker collinearity with general colonialism:
    should attenuate under both AJR_COLONIAL and EUR_COLONIZED.
  - If through a non-colonial channel (direct disease, other ancestry):
    should be stable.
  - FADS1/2 and AMY1 are global agricultural-adaptation markers; their
    coefficients should be relatively stable.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper5_horserace.subsamples import AJR_COLONIAL
from analysis.paper5_horserace.shapley import shapley_r2_decomposition

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

# Broader binary: was the country a European colony / protectorate / dominion
# at any point 1500-1980? Includes both extractive (AJR) and settler colonies.
# Excludes never-colonized core states (CHN, JPN, KOR, THA, IRN, TUR, etc.)
# and European powers themselves.
EUR_COLONIZED = sorted(set(AJR_COLONIAL) | {
    "AGO", "AUS", "BWA", "BDI", "CAN", "CIV", "CPV", "CUB",
    "DJI", "ERI", "GMB", "GNB", "GNQ", "IRQ", "JOR", "KHM",
    "KWT", "LAO", "LBN", "LBR", "LBY", "LKA", "MMR", "MRT",
    "MUS", "MYS", "NAM", "NPL", "NZL", "OMN", "PNG", "PRI",
    "PSE", "QAT", "SGP", "SOM", "SSD", "SWZ", "SYR", "TCD",
    "TLS", "TTO", "USA", "ARE", "YEM",
})

ALLELE_COLS = ["fa_lct", "fa_adh1b", "fa_amy1", "fa_edar",
                "fa_darc", "fa_slc24a5", "fa_hbb", "fa_fads"]
NON_FA_SUBSTRATES = ["t_mean_pre1750", "p_mean_pre1750",
                      "sigma_v_T_pre1750", "sigma_v_P_pre1750",
                      "neolithic_frac", "ancestral_yield_log",
                      "pandemic_intensity_norm"]
CONTROLS = ["abs_lat", "log_area", "landlocked", "ruggedness_proxy",
             "log_dist_neolithic"]


def _add_colonial_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add ajr_colonial and eur_colonized binaries to the panel."""
    df = df.copy()
    df["ajr_colonial"] = df["iso3"].isin(AJR_COLONIAL).astype(int)
    df["eur_colonized"] = df["iso3"].isin(EUR_COLONIZED).astype(int)
    return df


def _fit(df: pd.DataFrame, outcome: str, regressors: list[str],
         label: str) -> dict:
    pathway_cols = sorted(c for c in df.columns if c.startswith("pathway_"))[1:]
    keep = [outcome] + regressors + CONTROLS + pathway_cols
    sub = df.dropna(subset=keep).copy()
    X = sm.add_constant(sub[regressors + CONTROLS + pathway_cols].astype(float))
    res = sm.OLS(sub[outcome], X).fit(cov_type="HC3")
    return {"label": label, "n": int(res.nobs), "r2": float(res.rsquared),
            "model": res}


def run() -> pd.DataFrame:
    panel = pd.read_parquet(DATA / "deep_determinants_horserace.parquet")
    panel = _add_colonial_flags(panel)
    print(f"Panel: N={len(panel)} countries; AJR-colonial={panel['ajr_colonial'].sum()}; "
          f"EUR-colonized={panel['eur_colonized'].sum()}")

    outcome = "log_gdppc_2015"
    print(f"\n=== Headline OLS on {outcome} with/without colonial controls ===\n")

    base_regs = NON_FA_SUBSTRATES + ALLELE_COLS

    specs = {
        "baseline":         (base_regs, "no colonial control"),
        "ajr_colonial":     (base_regs + ["ajr_colonial"],
                             "+ AJR-colonial binary"),
        "eur_colonized":    (base_regs + ["eur_colonized"],
                             "+ broader European-colonized binary"),
        "both":             (base_regs + ["ajr_colonial", "eur_colonized"],
                             "+ both"),
    }

    fits = {name: _fit(panel, outcome, regs, lbl) for name, (regs, lbl) in specs.items()}

    rows = []
    for allele in ALLELE_COLS:
        r = {"allele": allele}
        for name, fit in fits.items():
            p = fit["model"].params.get(allele, np.nan)
            pv = fit["model"].pvalues.get(allele, np.nan)
            r[f"{name}_beta"] = float(p)
            r[f"{name}_p"] = float(pv)
        # % attenuation under EUR_COLONIZED
        b0 = r["baseline_beta"]
        b1 = r["eur_colonized_beta"]
        r["pct_attenuation"] = 100 * (1 - abs(b1) / abs(b0)) if abs(b0) > 1e-12 else np.nan
        rows.append(r)
    coef_df = pd.DataFrame(rows)

    print(f"{'allele':<10s}  {'base β':>10s}  {'base p':>8s}  "
          f"{'+AJR β':>10s}  {'p':>8s}  "
          f"{'+EUR β':>10s}  {'p':>8s}  {'%atten':>8s}")
    print("-" * 90)
    for _, row in coef_df.iterrows():
        print(f"{row['allele']:<10s}  {row['baseline_beta']:>+10.3f}  "
              f"{row['baseline_p']:>8.4f}  "
              f"{row['ajr_colonial_beta']:>+10.3f}  {row['ajr_colonial_p']:>8.4f}  "
              f"{row['eur_colonized_beta']:>+10.3f}  {row['eur_colonized_p']:>8.4f}  "
              f"{row['pct_attenuation']:>7.1f}%")

    # The colonial-control coefficients themselves
    print(f"\n=== Coefficient on colonial indicators ===")
    for name, fit in fits.items():
        m = fit["model"]
        for c in ("ajr_colonial", "eur_colonized"):
            if c in m.params:
                print(f"  {name:<20s}  {c}: β={m.params[c]:+.3f}, "
                      f"p={m.pvalues[c]:.4f}")

    # Bundle Shapley R² with/without colonial control
    print(f"\n=== Functional-bundle Shapley R² on {outcome} with/without colonial ===")
    from analysis.paper5_horserace.exercise1_shapley import (
        CLIMATE_BUNDLE, FUNCTIONAL_BUNDLE, SUBSTRATES, SUBSTRATE_KEYS,
    )

    for ctrl_name, extras in [("baseline", []),
                                ("+ AJR_COLONIAL", ["ajr_colonial"]),
                                ("+ EUR_COLONIZED", ["eur_colonized"])]:
        sub = panel.dropna(subset=[outcome] + list(CLIMATE_BUNDLE) +
                            list(FUNCTIONAL_BUNDLE) +
                            ["neolithic_frac", "ancestral_yield_log",
                             "pandemic_intensity_norm"] +
                            CONTROLS + extras)
        result = shapley_r2_decomposition(
            sub, outcome, substrates=SUBSTRATES,
            controls=CONTROLS + extras + [c for c in sub.columns
                                            if c.startswith("pathway_")][1:],
        )
        # The result is a dict keyed by substrate (or '+'.join for coalition)
        fa_key = "+".join(FUNCTIONAL_BUNDLE)
        cl_key = "+".join(CLIMATE_BUNDLE)
        print(f"\n  {ctrl_name} (N={len(sub)}):")
        print(f"    functional-allele bundle Shapley R²: {result.get(fa_key, np.nan):.4f}")
        print(f"    climate bundle Shapley R²:           {result.get(cl_key, np.nan):.4f}")
        for s in ("neolithic_frac", "ancestral_yield_log", "pandemic_intensity_norm"):
            print(f"    {s:<32s} Shapley R²: {result.get(s, np.nan):.4f}")

    out = DATA / "deep_determinants" / "exercise_colonial_partial_results.parquet"
    coef_df.to_parquet(out, index=False)
    print(f"\nWrote {out}")
    return coef_df


if __name__ == "__main__":
    run()
