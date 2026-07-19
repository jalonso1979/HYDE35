"""Volcanic dust-veil IV — decade-level companion to `03_volcanic_iv_pandemics.py`.

The annual specification finds a near-zero first-stage F.  Test whether the
volcanic→pandemic link emerges at coarser temporal scale: aggregate to
decades, instrument decade-pandemic-intensity with cumulative volcanic
forcing in the *prior* 1-3 decades.

This is a Buentgen-2016-style "536 CE dust-veil" hypothesis at scale.
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
    df["decade"] = (df["year_ce"] // 10) * 10
    d = (df.groupby("decade")
           .agg(lead_z=("lead_z", "mean"),
                 pandemic=("pandemic_v2_intensity", "sum"),
                 volcanic=("volcanic", "sum"),
                 war=("war_v2_intensity", "sum"))
           .reset_index())
    # lag the volcanic series (in decades)
    d["volc_l1"] = d["volcanic"].shift(1)
    d["volc_l2"] = d["volcanic"].shift(2)
    d["volc_l3"] = d["volcanic"].shift(3)
    d = d.dropna()
    print(f"N (decades after lag): {len(d)}")

    # First stage
    fs_X = sm.add_constant(d[["volcanic", "volc_l1", "volc_l2", "volc_l3", "war"]])
    fs = sm.OLS(d["pandemic"], fs_X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    print("\n=== Decade-level first stage ===")
    print(fs.summary().tables[1])
    inst_names = ["volc_l1", "volc_l2", "volc_l3"]
    f_test = fs.f_test([f"{n} = 0" for n in inst_names])
    print(f"\nFirst-stage F on instruments (decade-lags 1/2/3): "
          f"F = {float(f_test.fvalue):.3f}, p = {float(f_test.pvalue):.4g}")

    # 2SLS
    exog = sm.add_constant(d[["volcanic", "war"]])
    endog = d[["pandemic"]]
    instr = d[inst_names]
    iv = IV2SLS(d["lead_z"], exog, endog, instr).fit(cov_type="kernel",
                                                       kernel="bartlett",
                                                       bandwidth=2)
    print("\n=== Decade-level 2SLS: lead_z on pandemic (IV: lag-1/2/3 volcanic decades) ===")
    print(iv.summary)

    # OLS comparison
    ols = sm.OLS(d["lead_z"], sm.add_constant(d[["pandemic", "volcanic", "war"]])
                 ).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    print("\n=== Decade-level OLS for comparison ===")
    print(ols.summary().tables[1])

    summary = pd.DataFrame({
        "spec":  ["OLS", "2SLS"],
        "beta":  [float(ols.params["pandemic"]),
                   float(iv.params["pandemic"])],
        "se":    [float(ols.bse["pandemic"]),
                   float(iv.std_errors["pandemic"])],
        "p":     [float(ols.pvalues["pandemic"]),
                   float(iv.pvalues["pandemic"])],
        "N":     [int(ols.nobs), int(iv.nobs)],
        "fs_F":  [np.nan, float(f_test.fvalue)],
    })
    summary.to_csv(OUT / "iv_pandemic_volcanic_decade.csv", index=False)
    print(f"\nSaved {OUT}/iv_pandemic_volcanic_decade.csv")
    print(f"\nHeadline: OLS β = {summary.beta[0]:+.4f} (SE {summary.se[0]:.4f}); "
          f"2SLS β = {summary.beta[1]:+.4f} (SE {summary.se[1]:.4f}); "
          f"first-stage F = {summary.fs_F[1]:.2f}.")


if __name__ == "__main__":
    main()
