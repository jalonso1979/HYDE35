"""Volcanic dust-veil IV for Roman pandemics — robustness exercise
addressing FINDINGS.md §5.2 option (c).

Idea: pre-modern epidemics were partly triggered by climate-driven famine
and immune compromise (Harper 2017; Newfield 2018; Buentgen 2016 on 536
CE).  If volcanic-cooling shocks raise pandemic probability with a lag,
we can instrument pandemic timing with LAGGED volcanic forcing.

We control for contemporaneous and short-lead volcanic forcing in the
second stage (to absorb the direct climate-on-activity channel) and use
medium-lag volcanic (t-5, t-7, t-10) as plausibly exogenous variation
in pandemic risk.

Exclusion restriction caveat: lagged volcanic could affect lead-Z via
multi-year climate persistence, deforestation/agricultural rebound, or
political-economy responses.  We report results as a robustness exercise,
not a clean point identification.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "roman_jevons"


def main() -> None:
    df = pd.read_csv(DATA / "roman_v2_panel.csv").sort_values("year_ce").reset_index(drop=True)
    # Build volcanic lags
    for lag in (1, 2, 3, 5, 7, 10):
        df[f"volc_l{lag}"] = df["volcanic"].shift(lag)
    # Drop rows lost to lag construction
    work = df.dropna(subset=["lead_z", "pandemic_v2_intensity",
                              "volcanic", "volc_l1", "volc_l2", "volc_l3",
                              "volc_l5", "volc_l7", "volc_l10"]).copy()
    print(f"N (working sample after lag construction): {len(work)}")

    # ── First-stage: pandemic_intensity on volcanic lags ──────────────
    print("\n=== First stage: pandemic_v2_intensity on volcanic and lags ===")
    fs_X = work[["volcanic", "volc_l1", "volc_l2", "volc_l3",
                  "volc_l5", "volc_l7", "volc_l10",
                  "war_v2_intensity"]]
    fs_X = sm.add_constant(fs_X)
    fs = sm.OLS(work["pandemic_v2_intensity"], fs_X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    print(fs.summary().tables[1])
    # First-stage F on lag-5,7,10 (the instruments)
    inst_names = ["volc_l5", "volc_l7", "volc_l10"]
    f_test = fs.f_test([f"{n} = 0" for n in inst_names])
    print(f"\nFirst-stage F (instruments only, lags 5/7/10): "
          f"F = {float(f_test.fvalue):.3f}, p = {float(f_test.pvalue):.4g}")
    print(f"(Stock-Yogo rule of thumb: F > 10 suggests strong instruments)")

    # ── 2SLS using volc_l5, volc_l7, volc_l10 as instruments ─────────
    # endog: pandemic_v2_intensity
    # exog (controls):  volcanic, volc_l1, volc_l2, volc_l3, war
    # instruments: volc_l5, volc_l7, volc_l10
    endog = work[["pandemic_v2_intensity"]]
    exog  = sm.add_constant(work[["volcanic", "volc_l1", "volc_l2", "volc_l3",
                                    "war_v2_intensity"]])
    instr = work[inst_names]
    y     = work["lead_z"]
    iv = IV2SLS(y, exog, endog, instr).fit(cov_type="kernel", kernel="bartlett",
                                              bandwidth=5)
    print("\n=== 2SLS: lead_z on pandemic_intensity (IV: lagged volcanic) ===")
    print(iv.summary)

    # ── OLS comparison ────────────────────────────────────────────────
    ols_X = sm.add_constant(work[["pandemic_v2_intensity",
                                    "volcanic", "volc_l1", "volc_l2", "volc_l3",
                                    "war_v2_intensity"]])
    ols = sm.OLS(y, ols_X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    print("\n=== OLS comparison (same controls, no IV) ===")
    print(ols.summary().tables[1])

    # ── Save summary table ────────────────────────────────────────────
    summary = pd.DataFrame({
        "spec":   ["OLS", "2SLS"],
        "beta":   [float(ols.params["pandemic_v2_intensity"]),
                    float(iv.params["pandemic_v2_intensity"])],
        "se":     [float(ols.bse["pandemic_v2_intensity"]),
                    float(iv.std_errors["pandemic_v2_intensity"])],
        "p":      [float(ols.pvalues["pandemic_v2_intensity"]),
                    float(iv.pvalues["pandemic_v2_intensity"])],
        "N":      [int(ols.nobs), int(iv.nobs)],
        "fs_F":   [np.nan, float(f_test.fvalue)],
    })
    summary.to_csv(OUT / "iv_pandemic_volcanic.csv", index=False)
    print(f"\nSaved {OUT}/iv_pandemic_volcanic.csv")
    print("\nHeadline: OLS β = "
          f"{summary.beta[0]:+.4f} (SE {summary.se[0]:.4f}); "
          f"2SLS β = {summary.beta[1]:+.4f} (SE {summary.se[1]:.4f}); "
          f"first-stage F = {summary.fs_F[1]:.2f}.")


if __name__ == "__main__":
    main()
