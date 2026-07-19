"""Three diagnostics for the agricultural-area response to volcanic forcing.

(1) Pathway heterogeneity: do all four pathways show the same area
    contraction, or is the response concentrated in crop-dominant late?
(2) Time-window stability: does the area contraction hold across
    1500-1900 full, post-1700, post-1750, 1800-1900? The 21-country core
    showed significant contraction post-1700 but zero at 1800-1900.
(3) Demographic-margin orthogonality: does the area coefficient survive
    when we add Δlog P as a control? If the area response is just a
    mechanical consequence of population (fewer people → less cultivation),
    controlling for pop should null it.

For each pathway and window we report:
  - cropland-area coefficient and SE
  - grazing-area coefficient and SE
  - cropland-area coefficient with Δlog P added as control (test 3)

Country sub-samples: full pathway membership; substantive 21-country
core for crop-dominant late (drops 24 small-island/tropical-Africa
HYDE-artifact cases).

Output: analysis/data/area_margin_diagnostic.parquet
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

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer",  3: "High-density intensive",
                 4: "Early extensifiers"}

# HYDE small-island/tropical-Africa colonial-transition outliers in the
# crop-dominant late cluster.  Applied only to cluster 0.
CDL_OUTLIERS = ['FJI','REU','NCL','WSM','TON','JAM','MUS','VUT','DMA','GLP',
                'LCA','KNA','MTQ','VCT','HTI','SLV','LSO','SLE','CIV','GNB',
                'GNQ','CMR','BRN','SGP']


def _build_panel() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    p = p.sort_values(["iso3", "year"]).copy()
    p["crop_km2"] = p["crop_km2"].clip(lower=1e-6)
    p["past_km2"] = p["past_km2"].clip(lower=1e-6)
    p["log_crop_km2"] = np.log(p["crop_km2"])
    p["log_past_km2"] = np.log(p["past_km2"])
    g = p.groupby("iso3")
    for level, growth in [("log_crop_km2", "g_crop_km2_ann"),
                            ("log_past_km2", "g_past_km2_ann")]:
        nxt = g[level].shift(-1)
        dt = g["year"].shift(-1) - p["year"]
        p[growth] = (nxt - p[level]) / dt
    return p


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
    panel = _build_panel()
    print(f"Panel: {len(panel):,} rows, {panel['iso3'].nunique()} countries\n")

    windows = [
        ("Full 1500-1900",  panel["year"].notna()),
        ("Post-1700",       panel["year"] >= 1700),
        ("Post-1750",       panel["year"] >= 1750),
        ("1800-1900",       panel["year"] >= 1800),
    ]
    outcomes = [
        ("g_crop_km2_ann", "Δlog cropland area"),
        ("g_past_km2_ann", "Δlog grazing area"),
    ]
    rows = []

    # ===== (1) Pathway heterogeneity x (2) time window =====
    print("=" * 80)
    print("(1)+(2) PATHWAY × WINDOW: cropland-area and grazing-area response")
    print("=" * 80)

    for cl in [0, 1, 3, 4]:  # skip singleton irrigation pioneer
        pname = PATHWAY_NAMES[cl]
        print(f"\n--- {pname} (cluster {cl}) ---")
        sub = panel[panel["cluster"] == cl].copy()
        # For crop-dominant late, also report 21-country core
        sample_specs = [(f"full {pname}", sub)]
        if cl == 0:
            sample_specs.append(
                (f"core 21-country {pname}", sub[~sub["iso3"].isin(CDL_OUTLIERS)])
            )

        for sample_label, s in sample_specs:
            print(f"  Sample: {sample_label} (n_iso = {s['iso3'].nunique()})")
            for win_label, mask in windows:
                s2 = s[mask.reindex(s.index, fill_value=False)
                       if hasattr(mask, "reindex") else
                       s.index.isin(panel[mask].index)]
                # Simpler: just use boolean from the same row-aligned panel
                s2 = s[s["year"].isin(panel[mask]["year"].unique())].copy()
                # Actually we want to filter by year directly
                if win_label == "Full 1500-1900":
                    s2 = s
                elif win_label == "Post-1700":
                    s2 = s[s["year"] >= 1700]
                elif win_label == "Post-1750":
                    s2 = s[s["year"] >= 1750]
                elif win_label == "1800-1900":
                    s2 = s[s["year"] >= 1800]

                for outcome, olabel in outcomes:
                    r = _run(s2, outcome, CONTROLS)
                    if r is None:
                        continue
                    sig = ("***" if r["p"] < 0.01 else
                           "**"  if r["p"] < 0.05 else
                           "*"   if r["p"] < 0.10 else "")
                    print(f"    {win_label:18s} {olabel:25s}: "
                          f"β = {r['beta']:+.4g}  SE = {r['se']:.4g}  "
                          f"p = {r['p']:.3g}  N = {r['n']}  {sig}")
                    rows.append({"diagnostic": "pathway_window",
                                 "cluster": cl, "pathway": pname,
                                 "sample": sample_label, "window": win_label,
                                 "outcome": outcome, **r})

    # ===== (3) Demographic-margin orthogonality =====
    print("\n" + "=" * 80)
    print("(3) DEMOGRAPHIC-MARGIN ORTHOGONALITY: does Δlog cropland area")
    print("    survive controlling for Δlog P?")
    print("=" * 80)

    # Add g_pop_ann to the control set
    CONTROLS_WITH_POPGROWTH = CONTROLS + ["g_pop_ann"]

    for cl, pname in [(0, "Crop-dominant late"), (1, "Pastoral/mixed late"),
                        (3, "High-density intensive"), (4, "Early extensifiers")]:
        print(f"\n--- {pname} ---")
        sub = panel[panel["cluster"] == cl].copy()
        sample_specs = [(f"full {pname}", sub)]
        if cl == 0:
            sample_specs.append(
                (f"core 21-country", sub[~sub["iso3"].isin(CDL_OUTLIERS)])
            )

        for sample_label, s in sample_specs:
            for win_label, ys in [("Full 1500-1900", 0),
                                    ("Post-1700", 1700),
                                    ("1800-1900", 1800)]:
                s2 = s[s["year"] >= ys].copy() if ys > 0 else s.copy()
                # baseline (no pop-growth control)
                r0 = _run(s2, "g_crop_km2_ann", CONTROLS)
                # with pop-growth control
                r1 = _run(s2, "g_crop_km2_ann", CONTROLS_WITH_POPGROWTH)
                if r0 is None or r1 is None:
                    continue
                shrink = ((r1["beta"] - r0["beta"]) / r0["beta"] * 100
                          if r0["beta"] != 0 else np.nan)
                print(f"  {sample_label:25s} {win_label:18s}: "
                      f"baseline β = {r0['beta']:+.4g} (p={r0['p']:.3g}); "
                      f"+pop control β = {r1['beta']:+.4g} (p={r1['p']:.3g})")
                rows.append({"diagnostic": "pop_control_baseline",
                             "cluster": cl, "pathway": pname,
                             "sample": sample_label, "window": win_label,
                             "outcome": "g_crop_km2_ann", **r0})
                rows.append({"diagnostic": "pop_control_with_pop",
                             "cluster": cl, "pathway": pname,
                             "sample": sample_label, "window": win_label,
                             "outcome": "g_crop_km2_ann", **r1})

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "area_margin_diagnostic.parquet", index=False)
    print(f"\nSaved {DATA/'area_margin_diagnostic.parquet'}")


if __name__ == "__main__":
    main()
