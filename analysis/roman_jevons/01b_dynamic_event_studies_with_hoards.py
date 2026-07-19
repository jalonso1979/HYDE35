"""Dynamic event-studies on the Roman v2 panel WITH CHRE coin-hoards covariate.

Companion to `01_dynamic_event_studies.py`.  Goal: test whether the Antonine
and Cyprian parallel-trends failures close once we control for empire-wide
coin-hoarding pressure (CHRE; 18,310 dated hoards, 12-region 25-yr panel),
which is a leading indicator of monetary/military crisis and may be co-
moving with the lead-Z mining proxy in ways the baseline spec misses.

Two coin-hoards covariates are constructed and tested:
    * `hoards_empire_z` — empire-wide pooled log1p hoards count, z-scored
      across the 86 BCE–802 CE window
    * `hoards_italy_z`  — Italy-region z-score (within-region z),
      a more peninsular crisis index that may better track the Western
      mining/smelting catchment of the Greenland lead-Z record.

For each treatment (volcanic, pandemic pooled, three pandemic families)
we run two augmented specs: (a) +hoards_empire_z and (b) +hoards_italy_z,
forward-filling the 25-yr bin values to annual frequency.  We report
parallel-trends F-statistics side-by-side with the no-covariate baseline
in `FINDINGS.md`.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "paper4_shadow"))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
SHARED = ROOT / "analysis" / "shared" / "pandemics_v3" / "data"
OUT = ROOT / "analysis" / "roman_jevons"

LEADS_LAGS = list(range(-5, 11))


def build_hoards_annual() -> pd.DataFrame:
    """Forward-fill the 25-yr CHRE bins to annual frequency, return a df
    with columns year_ce, hoards_empire_z, hoards_italy_z."""
    h = pd.read_csv(SHARED / "coin_hoards_region_25yr.csv")
    # Empire-wide pooled
    emp = h.groupby("year_bin_start").agg(n=("n_hoards", "sum")).reset_index()
    emp["log1p_n"] = np.log1p(emp["n"])
    emp["hoards_empire_z"] = (emp["log1p_n"] - emp["log1p_n"].mean()) / emp["log1p_n"].std()
    # Italy region
    italy = h[h["region"] == "Italy"][["year_bin_start", "z_log_hoards_within_region"]]
    italy = italy.rename(columns={"z_log_hoards_within_region": "hoards_italy_z"})
    bins = emp[["year_bin_start", "hoards_empire_z"]].merge(italy,
                                                            on="year_bin_start",
                                                            how="left")
    bins = bins.sort_values("year_bin_start").reset_index(drop=True)
    # Forward-fill to annual — each year is assigned its 25-yr bin's value
    # (bin labelled by the bin-start year, so years [b, b+24] inherit bin b).
    rows = []
    for _, r in bins.iterrows():
        b = int(r["year_bin_start"])
        for y in range(b, b + 25):
            rows.append({"year_ce": y,
                          "hoards_empire_z": r["hoards_empire_z"],
                          "hoards_italy_z":  r["hoards_italy_z"]})
    return pd.DataFrame(rows)


def dynamic_irf(df: pd.DataFrame, outcome: str, treatment: str,
                  controls: list[str] | None = None,
                  leads_lags: list[int] = LEADS_LAGS) -> dict:
    controls = controls or []
    p = df.copy().sort_values("year_ce")
    cols = []
    for h in leads_lags:
        col = f"{treatment}_h{h:+d}"
        p[col] = p[treatment].shift(-h)
        cols.append(col)
    p = p.dropna(subset=[outcome] + cols + controls)
    if len(p) < 30:
        return {}
    X = sm.add_constant(p[cols + controls])
    y = p[outcome]
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    leads = [c for h, c in zip(leads_lags, cols) if h < 0]
    diag: dict = {"N": int(r.nobs)}
    if leads:
        ft = r.f_test([f"{c} = 0" for c in leads])
        diag["lead_F"] = float(ft.fvalue)
        diag["lead_p"] = float(ft.pvalue)
        diag["n_leads"] = len(leads)
    # also keep individual coefs for table later
    coefs = {}
    for h_off, c in zip(leads_lags, cols):
        coefs[h_off] = (float(r.params[c]),
                         float(r.bse[c]),
                         float(r.pvalues[c]))
    diag["coefs"] = coefs
    return diag


def verdict(p: float | None) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    return "PASS" if p > 0.10 else "FAIL"


def main() -> None:
    df = pd.read_csv(DATA / "roman_v2_panel.csv").sort_values("year_ce").reset_index(drop=True)
    hoards = build_hoards_annual()
    df = df.merge(hoards, on="year_ce", how="left")
    print(f"Panel: N = {len(df)} years, "
          f"{int(df.year_ce.min())} to {int(df.year_ce.max())} CE")
    print(f"hoards_empire_z coverage: {df.hoards_empire_z.notna().sum()}")
    print(f"hoards_italy_z  coverage: {df.hoards_italy_z.notna().sum()}")
    print()

    treatments = [
        ("volcanic",                       "Volcanic forcing"),
        ("pandemic_v2_intensity",          "Pandemic intensity (pooled)"),
        ("pandemic_v2_family_antonine",    "Antonine (~165–180 CE)"),
        ("pandemic_v2_family_cyprian",     "Cyprian (~249–262 CE)"),
        ("pandemic_v2_family_justinianic", "Justinianic (~541 CE)"),
    ]

    rows = []
    for tr, label in treatments:
        print(f"=== {label} ({tr}) ===")
        base = dynamic_irf(df, "lead_z", tr)
        emp = dynamic_irf(df, "lead_z", tr, controls=["hoards_empire_z"])
        ity = dynamic_irf(df, "lead_z", tr, controls=["hoards_italy_z"])
        both = dynamic_irf(df, "lead_z", tr,
                            controls=["hoards_empire_z", "hoards_italy_z"])
        for tag, d in [("baseline (no ctrl)", base),
                       ("+ hoards_empire_z",  emp),
                       ("+ hoards_italy_z",   ity),
                       ("+ both",             both)]:
            F = d.get("lead_F", float("nan"))
            p = d.get("lead_p", float("nan"))
            N = d.get("N", 0)
            tag_padded = tag.ljust(20)
            print(f"  {tag_padded}  F = {F:7.3f}  p = {p:.4g}  "
                  f"{verdict(p)}  (N={N})")
            rows.append({
                "treatment": tr,
                "spec":      tag,
                "lead_F":    F,
                "lead_p":    p,
                "verdict":   verdict(p),
                "N":         N,
            })
        # also note h=+5 / h=+10 effect coefficients on the +empire-z spec
        for h in [0, 5, 10]:
            if h in emp.get("coefs", {}):
                b, se, pv = emp["coefs"][h]
                print(f"    [empire-z spec] h={h:+d}:  beta = {b:+.5f}, "
                      f"SE = {se:.5f}, p = {pv:.3g}")
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "irf_parallel_trends_with_hoards.csv", index=False)
    print(f"\nSaved summary table to {OUT}/irf_parallel_trends_with_hoards.csv")


if __name__ == "__main__":
    main()
