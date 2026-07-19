"""KK10 cross-validation of the area-orthogonality finding.

The HYDE-based area regression of Section 4.5 finds that the cropland-area
response to volcanic forcing collapses to zero once Δlog P is added as a
control.  HYDE's cropland back-projection uses population as an input
variable, so the collapse could be a mechanical consequence of that
methodology (the control captures the input HYDE used to construct the
outcome) rather than a substantive "land follows people" finding.

KK10 (Kaplan et al. 2011) is an independent reconstruction whose
anthropogenic-fraction series is methodologically population-independent.
Re-running the orthogonality test on KK10 distinguishes the two
interpretations:

  - If the KK10 baseline coefficient is significant and survives the
    pop-control, the area response on KK10 is NOT mediated by population.
    The HYDE orthogonality is then a HYDE-back-projection artefact and
    the substantive "land follows people" claim does not hold.

  - If the KK10 area coefficient is null in baseline (or collapses under
    pop-control as on HYDE), the orthogonality finding generalises across
    reconstructions — the "land follows people" claim is robust.

KK10 ends at 1850, so the 1800-1900 window becomes 1800-1850 (5 decades).
Also, KK10 reports only combined anthropogenic land (cropland + pasture),
not separate cropland and grazing.  The comparison outcome on HYDE is
therefore log(crop_km2 + past_km2), to align like-with-like.

Output: analysis/data/kk10_orthogonality.parquet
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

CDL_OUTLIERS = ['FJI','REU','NCL','WSM','TON','JAM','MUS','VUT','DMA','GLP',
                'LCA','KNA','MTQ','VCT','HTI','SLV','LSO','SLE','CIV','GNB',
                'GNQ','CMR','BRN','SGP']


def _build_combined_panel() -> pd.DataFrame:
    """Merge the joint-VAR panel with KK10 country totals; compute
    annualised log-change of (HYDE crop+past) and (KK10 anthro) so the
    two outcomes are like-with-like."""
    hyde = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    kk10 = pd.read_parquet(DATA / "kk10_country_panel.parquet")
    print(f"HYDE panel: {len(hyde):,} rows, {hyde['iso3'].nunique()} countries")
    print(f"KK10 panel: {len(kk10):,} rows, {kk10['iso3'].nunique()} countries")

    # HYDE: build log(crop+past) and its annualised growth
    hyde = hyde.sort_values(["iso3", "year"]).copy()
    hyde["hyde_anthro_km2"] = hyde["crop_km2"].clip(lower=1e-6) + \
                              hyde["past_km2"].clip(lower=1e-6)
    hyde["log_hyde_anthro"] = np.log(hyde["hyde_anthro_km2"])
    g = hyde.groupby("iso3")
    nxt = g["log_hyde_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - hyde["year"]
    hyde["g_hyde_anthro_ann"] = (nxt - hyde["log_hyde_anthro"]) / dt

    # KK10: same logic
    kk10 = kk10.sort_values(["iso3", "year"]).copy()
    kk10["log_kk10_anthro"] = np.log(kk10["kk10_anthro_km2"].clip(lower=1e-6))
    g = kk10.groupby("iso3")
    nxt = g["log_kk10_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - kk10["year"]
    kk10["g_kk10_anthro_ann"] = (nxt - kk10["log_kk10_anthro"]) / dt

    # Merge on (iso3, year)
    merged = hyde.merge(
        kk10[["iso3", "year", "kk10_anthro_km2",
              "log_kk10_anthro", "g_kk10_anthro_ann"]],
        on=["iso3", "year"], how="left"
    )
    print(f"Merged: {len(merged):,} rows, "
          f"{merged.dropna(subset=['g_kk10_anthro_ann'])['iso3'].nunique()} "
          f"countries with KK10 growth observed")
    return merged


def _run(d: pd.DataFrame, lhs: str, controls: list[str]) -> dict | None:
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    if len(d) < 20 or d["iso3"].nunique() < 2:
        return None
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
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
    panel = _build_combined_panel()
    cdl = panel[panel["cluster"] == 0].copy()
    cdl_core = cdl[~cdl["iso3"].isin(CDL_OUTLIERS)]
    print(f"\nCrop-dominant late: {cdl['iso3'].nunique()} countries (full), "
          f"{cdl_core['iso3'].nunique()} (21-country core)\n")

    # Test both outcomes on the same samples; report baseline and +pop-growth control
    CONTROLS_WITH_POP = CONTROLS + ["g_pop_ann"]

    rows = []
    samples = [
        ("Crop-dominant late (45-country full)",       cdl),
        ("Crop-dominant late (21-country core)",       cdl_core),
    ]
    windows = [
        ("Full 1500-1850",      None),
        ("Post-1700",           1700),
        ("Post-1750",           1750),
        ("1800-1850",           1800),
    ]
    outcomes = [
        ("g_hyde_anthro_ann", "HYDE  Δlog (crop+past) km²"),
        ("g_kk10_anthro_ann", "KK10  Δlog anthro km²"),
    ]
    print("="*90)
    print(f"{'sample':40s}  {'window':14s}  {'outcome':25s}  base β (p)        +pop β (p)")
    print("-"*90)
    for sample_label, s in samples:
        for win_label, ymin in windows:
            s2 = s if ymin is None else s[s["year"] >= ymin]
            for outcome, olabel in outcomes:
                r0 = _run(s2, outcome, CONTROLS)
                r1 = _run(s2, outcome, CONTROLS_WITH_POP)
                if r0 is None or r1 is None:
                    continue
                rows.append({"sample": sample_label, "window": win_label,
                             "outcome": outcome,
                             "spec": "baseline", **r0})
                rows.append({"sample": sample_label, "window": win_label,
                             "outcome": outcome,
                             "spec": "+pop_control",
                             "beta": r1["beta"], "se": r1["se"], "p": r1["p"],
                             "n": r1["n"], "n_iso": r1["n_iso"]})
                sig0 = "***" if r0["p"]<0.01 else "**" if r0["p"]<0.05 else "*" if r0["p"]<0.10 else ""
                sig1 = "***" if r1["p"]<0.01 else "**" if r1["p"]<0.05 else "*" if r1["p"]<0.10 else ""
                print(f"{sample_label[:38]:38s}  {win_label:14s}  {olabel:25s}  "
                      f"{r0['beta']:+.3g} ({r0['p']:.3g}) {sig0:3s}   "
                      f"{r1['beta']:+.3g} ({r1['p']:.3g}) {sig1:3s}")
        print()

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "kk10_orthogonality.parquet", index=False)
    print(f"\nSaved {DATA/'kk10_orthogonality.parquet'}")


if __name__ == "__main__":
    main()
