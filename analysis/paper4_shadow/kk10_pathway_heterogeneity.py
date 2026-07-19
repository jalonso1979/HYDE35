"""KK10 anthropogenic-area orthogonality across all four pathways.

We already showed that on the crop-dominant late 21-country core post-1700
sample, the KK10 anthropogenic-area response to volcanic forcing is
+1.07e-5 per Tg with p=0.005 and survives the Δlog P control.  The
remaining question is whether the same orthogonal-Boserupian signal
appears in the other three pathways or whether it is concentrated in
crop-dominant late.

For each pathway we run:
  - baseline KK10 area regression (joint-VAR controls)
  - + Δlog P control (the orthogonality test)
  - the HYDE comparator (Δlog crop+past area, same sample)

Post-1700 window throughout (where the KK10 signal was identified).

Output: analysis/data/kk10_pathway_heterogeneity.parquet
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
CONTROLS_WITH_POP = CONTROLS + ["g_pop_ann"]

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer",  3: "High-density intensive",
                 4: "Early extensifiers"}

CDL_OUTLIERS = ['FJI','REU','NCL','WSM','TON','JAM','MUS','VUT','DMA','GLP',
                'LCA','KNA','MTQ','VCT','HTI','SLV','LSO','SLE','CIV','GNB',
                'GNQ','CMR','BRN','SGP']


def _build_combined_panel() -> pd.DataFrame:
    hyde = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    kk10 = pd.read_parquet(DATA / "kk10_country_panel.parquet")

    hyde = hyde.sort_values(["iso3", "year"]).copy()
    hyde["hyde_anthro_km2"] = hyde["crop_km2"].clip(lower=1e-6) + \
                              hyde["past_km2"].clip(lower=1e-6)
    hyde["log_hyde_anthro"] = np.log(hyde["hyde_anthro_km2"])
    g = hyde.groupby("iso3")
    nxt = g["log_hyde_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - hyde["year"]
    hyde["g_hyde_anthro_ann"] = (nxt - hyde["log_hyde_anthro"]) / dt

    kk10 = kk10.sort_values(["iso3", "year"]).copy()
    kk10["log_kk10_anthro"] = np.log(kk10["kk10_anthro_km2"].clip(lower=1e-6))
    g = kk10.groupby("iso3")
    nxt = g["log_kk10_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - kk10["year"]
    kk10["g_kk10_anthro_ann"] = (nxt - kk10["log_kk10_anthro"]) / dt

    return hyde.merge(
        kk10[["iso3", "year", "kk10_anthro_km2",
              "log_kk10_anthro", "g_kk10_anthro_ann"]],
        on=["iso3", "year"], how="left"
    )


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
    panel = panel[panel["year"] >= 1700].copy()
    print(f"Post-1700 panel: {len(panel):,} rows, "
          f"{panel['iso3'].nunique()} countries\n")

    pathways = [
        (0, "Crop-dominant late", False),   # full
        (0, "Crop-dominant late", True),    # 21-country core
        (1, "Pastoral/mixed late", False),
        (3, "High-density intensive", False),
        (4, "Early extensifiers", False),
    ]
    rows = []
    print("="*120)
    fmt_h = f"{'Pathway':30s}  {'Sample':12s}  {'Outcome':16s}  {'Baseline β (SE, p)':30s}  {'+Δlog P (β, p)':28s}"
    print(fmt_h)
    print("-"*120)

    for cl, pname, drop_outliers in pathways:
        sub = panel[panel["cluster"] == cl].copy()
        if drop_outliers and cl == 0:
            sub = sub[~sub["iso3"].isin(CDL_OUTLIERS)]
            sample_lbl = "21-c core"
        else:
            sample_lbl = "full"

        for outcome, olabel in [("g_hyde_anthro_ann", "HYDE crop+past"),
                                  ("g_kk10_anthro_ann", "KK10 anthro")]:
            r0 = _run(sub, outcome, CONTROLS)
            r1 = _run(sub, outcome, CONTROLS_WITH_POP)
            if r0 is None or r1 is None:
                continue
            sig0 = ("***" if r0["p"]<0.01 else "**" if r0["p"]<0.05 else
                    "*"   if r0["p"]<0.10 else "  ")
            sig1 = ("***" if r1["p"]<0.01 else "**" if r1["p"]<0.05 else
                    "*"   if r1["p"]<0.10 else "  ")
            survives = (r0["p"] < 0.05 and r1["p"] < 0.05
                        and np.sign(r0["beta"]) == np.sign(r1["beta"]))
            print(f"{pname:30s}  {sample_lbl:12s}  {olabel:16s}  "
                  f"{r0['beta']:+.3g} ({r0['se']:.2g}, p={r0['p']:.3g}) {sig0:3s}   "
                  f"{r1['beta']:+.3g} (p={r1['p']:.3g}) {sig1:3s}  "
                  f"{'SURVIVES' if survives else ''}")
            rows.append({
                "cluster": cl, "pathway": pname, "sample": sample_lbl,
                "outcome": outcome, "spec": "baseline", **r0,
            })
            rows.append({
                "cluster": cl, "pathway": pname, "sample": sample_lbl,
                "outcome": outcome, "spec": "+pop_control",
                "beta": r1["beta"], "se": r1["se"], "p": r1["p"],
                "n": r1["n"], "n_iso": r1["n_iso"],
            })
        print()

    pd.DataFrame(rows).to_parquet(DATA / "kk10_pathway_heterogeneity.parquet",
                                    index=False)
    print(f"\nSaved {DATA/'kk10_pathway_heterogeneity.parquet'}")


if __name__ == "__main__":
    main()
