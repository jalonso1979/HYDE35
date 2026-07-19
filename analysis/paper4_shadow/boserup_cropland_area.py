"""Boserupian regression with alternative cropland measures.

The headline cropland-share specification is sensitive to HYDE-construction
artifacts in 24 small-island and tropical-Africa countries whose cropland
share collapses in colonial transitions of 1840, 1850, and 1890.  The
denominator (cropland + grazing) is small in these countries, and small
absolute changes in either component create large share changes.

We test the headline Boserupian intensification finding against four
alternative outcome measures:

  (i)   Δ_ann log cropland area in km²  (the absolute extensive margin)
  (ii)  Δ_ann log grazing  area in km²  (the substitution-mirror outcome:
        if Boserup is converting grazing to cropland we should see this
        DECREASE in lockstep with the increase in (i))
  (iii) Δ_ann log cropland per capita    (Boserup's land-labour ratio,
        mechanically conflated with the demographic response)
  (iv)  Headline cropland share          (existing measure, for comparison)

For each outcome we run the same joint-VAR specification: country FE,
country-clustered SEs, joint-VAR controls (lagged log levels of pop,
cropland share, urban share + contemporaneous T, P, σ_T + VSSI).

Output:
    analysis/data/boserup_cropland_area.parquet
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

CONTROLS = ["log_pop", "log_crop_share", "log_urban_share",
            "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]

AMERICAS_IN_CDL = ["HTI", "JAM", "SLV"]
SMALL_OR_OUTLIER = ['FJI','REU','NCL','WSM','TON','JAM','MUS','VUT','DMA','GLP',
                    'LCA','KNA','MTQ','VCT','HTI','SLV','LSO','SLE','CIV','GNB',
                    'GNQ','CMR','BRN','SGP']


def _build_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the four outcome variables to the panel."""
    df = panel.sort_values(["iso3", "year"]).copy()
    df["crop_km2"] = df["crop_km2"].clip(lower=1e-6)   # guard log(0)
    df["past_km2"] = df["past_km2"].clip(lower=1e-6)
    df["pop"]      = df["pop"].clip(lower=1e-6)
    df["log_crop_km2"]    = np.log(df["crop_km2"])
    df["log_past_km2"]    = np.log(df["past_km2"])
    df["log_crop_pc"]     = np.log(df["crop_km2"] / df["pop"] * 100)  # hectares per person
    # Annualised log changes across HYDE intervals
    grp = df.groupby("iso3")
    for level_col, growth_col in [
        ("log_crop_km2", "g_crop_km2_ann"),
        ("log_past_km2", "g_past_km2_ann"),
        ("log_crop_pc",  "g_crop_pc_ann"),
    ]:
        next_level = grp[level_col].shift(-1)
        next_year  = grp["year"].shift(-1)
        dt = next_year - df["year"]
        df[growth_col] = (next_level - df[level_col]) / dt
    return df


def _run(d: pd.DataFrame, lhs: str) -> dict | None:
    d = d.dropna(subset=[lhs] + CONTROLS + ["iso3"]).copy()
    if len(d) < 20 or d["iso3"].nunique() < 2:
        return None
    g = d.groupby("iso3")
    for c in [lhs] + CONTROLS:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[CONTROLS])
    res = sm.OLS(d[lhs], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"]}
    )
    return {
        "beta": float(res.params.get("vssi_int", np.nan)),
        "se":   float(res.bse.get("vssi_int",   np.nan)),
        "p":    float(res.pvalues.get("vssi_int", np.nan)),
        "n":    int(res.nobs),
        "n_iso": int(d["iso3"].nunique()),
    }


def main() -> None:
    raw = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    panel = _build_outcomes(raw)
    cdl   = panel[panel["cluster"] == 0].copy()
    print(f"Crop-dominant late panel: {len(cdl):,} rows, "
          f"{cdl['iso3'].nunique()} countries\n")

    # Sanity check: distribution of the new outcomes
    print("=== Distribution of new outcomes in crop-dominant late ===")
    for col in ["g_crop_km2_ann", "g_past_km2_ann", "g_crop_pc_ann",
                 "g_crop_ann"]:
        s = cdl[col].dropna()
        print(f"  {col:18s}: N={len(s):4d}  mean={s.mean():+.5f}  "
              f"std={s.std():+.5f}  min={s.min():+.5f}  max={s.max():+.5f}")
    print()

    outcomes = [
        ("g_crop_km2_ann", "Δ_ann log cropland area (km²)"),
        ("g_past_km2_ann", "Δ_ann log grazing area (km²)"),
        ("g_crop_pc_ann",  "Δ_ann log cropland per capita"),
        ("g_crop_ann",     "Δ_ann log cropland share (headline)"),
    ]
    samples = [
        ("Full 1500-1900",            cdl["year"].notna()),
        ("Post-1700",                 cdl["year"] >= 1700),
        ("Post-1750",                 cdl["year"] >= 1750),
        ("1800-1900",                 cdl["year"] >= 1800),
        ("1800-1900 drop 24 outliers", (cdl["year"] >= 1800) & (~cdl["iso3"].isin(SMALL_OR_OUTLIER))),
        ("1800-1900 drop 3 Americas", (cdl["year"] >= 1800) & (~cdl["iso3"].isin(AMERICAS_IN_CDL))),
    ]
    rows = []
    for outcome_col, outcome_label in outcomes:
        print(f"\n=== Outcome: {outcome_label}  ({outcome_col}) ===")
        for sample_label, mask in samples:
            sub = cdl[mask]
            r = _run(sub, outcome_col)
            if r is None:
                print(f"  {sample_label:30s}: skip")
                continue
            sig = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
            print(f"  {sample_label:30s}: β = {r['beta']:+.5g}  "
                  f"(SE {r['se']:.5g}) p = {r['p']:.3g}  N = {r['n']}  "
                  f"n_iso = {r['n_iso']}  {sig}")
            rows.append({"outcome": outcome_col, "outcome_label": outcome_label,
                         "sample": sample_label, **r})
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "boserup_cropland_area.parquet", index=False)
    print(f"\nSaved {DATA/'boserup_cropland_area.parquet'}")


if __name__ == "__main__":
    main()
