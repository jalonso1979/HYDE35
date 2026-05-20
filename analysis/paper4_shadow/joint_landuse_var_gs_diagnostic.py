"""Diagnostic: does GS-mean climate matter for the joint land-use VAR?

Compares the existing annual-climate joint VAR against the GS-mean-climate
analogue, equation by equation. Decides whether Task 7 (substitute GS-mean
climate in joint VAR battery) is worth doing.

Decision rule (mirrors Task 6 diagnostic):
- GS HELPS: any climate coefficient changes by >50% AND flips significance,
  OR the VSSI coefficient changes by >20% in any equation.
- GS DOESN'T HELP: coefficients within 30% across all three equations and
  no significance flips on any climate or VSSI term.
- MIXED: in between.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

from analysis.paper4_shadow.joint_landuse_var import (
    _build_panel, _run_one, PATHWAY_NAMES,
)


def _attach_gs_climate(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace t_bar/p_bar/t_sd with GS-mean analogues; recompute deviations."""
    gs = pd.read_parquet(DATA / "country_climate_gs_1421_2025.parquet")
    gs = gs[["iso3", "year", "t_gs_mean_cropw", "p_gs_mean_cropw"]].dropna()

    rows = []
    for i, r in panel.iterrows():
        sub = gs[(gs["iso3"] == r["iso3"]) &
                 (gs["year"] >= r["year"]) &
                 (gs["year"] < r["next_year"])]
        if len(sub) < 3:
            rows.append({"t_bar_gs": np.nan, "p_bar_gs": np.nan, "t_sd_gs": np.nan})
        else:
            rows.append({"t_bar_gs": sub["t_gs_mean_cropw"].mean(),
                          "p_bar_gs": sub["p_gs_mean_cropw"].mean(),
                          "t_sd_gs":  sub["t_gs_mean_cropw"].std()})

    p2 = pd.concat([panel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    for c in ["t_bar_gs", "p_bar_gs", "t_sd_gs"]:
        g = p2.groupby("iso3")[c]
        p2[c + "_dev"] = p2[c] - g.transform("mean")
    return p2


def _sig_label(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _rel_change(b_ann: float, b_gs: float) -> str:
    if np.isnan(b_ann) or b_ann == 0:
        return "n/a"
    return f"{(b_gs - b_ann) / abs(b_ann) * 100:+.0f}%"


def main() -> None:
    print("Building annual-climate panel (existing _build_panel)...", flush=True)
    panel = _build_panel()
    print(f"  N={len(panel)} cells, {panel['iso3'].nunique()} countries")

    print("Attaching GS-mean climate columns...", flush=True)
    panel = _attach_gs_climate(panel)
    print(f"  GS-non-null cells: {panel['t_bar_gs_dev'].notna().sum()}")

    # Define the two spec variants (per-equation lagged level as state variable)
    annual_pop  = ["log_pop",         "t_bar_dev",    "p_bar_dev",    "t_sd_dev",    "vssi_int"]
    gs_pop      = ["log_pop",         "t_bar_gs_dev", "p_bar_gs_dev", "t_sd_gs_dev", "vssi_int"]
    annual_crop = ["log_crop_share",  "t_bar_dev",    "p_bar_dev",    "t_sd_dev",    "vssi_int"]
    gs_crop     = ["log_crop_share",  "t_bar_gs_dev", "p_bar_gs_dev", "t_sd_gs_dev", "vssi_int"]
    annual_urb  = ["log_urban_share", "t_bar_dev",    "p_bar_dev",    "t_sd_dev",    "vssi_int"]
    gs_urb      = ["log_urban_share", "t_bar_gs_dev", "p_bar_gs_dev", "t_sd_gs_dev", "vssi_int"]

    # Sample-comparability slice: only rows where all GS climate cols are non-null
    gs_sample = panel.dropna(subset=["t_bar_gs_dev"]).copy()
    print(f"  GS-restricted sample: {len(gs_sample)} cells "
          f"(drops {len(panel) - len(gs_sample)} from annual full)")

    def _three_eq(d: pd.DataFrame, ctrl_pop, ctrl_crop, ctrl_urb, label: str):
        rp = _run_one(d, "g_pop_ann",  ctrl_pop)
        rc = _run_one(d, "g_crop_ann", ctrl_crop)
        ru = _run_one(d, "g_urb_ann",  ctrl_urb)
        return {"label": label, "pop": rp, "crop": rc, "urb": ru}

    res = {
        "annual_full":   _three_eq(panel,     annual_pop, annual_crop, annual_urb, "annual_full"),
        "annual_gssamp": _three_eq(gs_sample, annual_pop, annual_crop, annual_urb, "annual_gssamp"),
        "gs":            _three_eq(gs_sample, gs_pop,     gs_crop,     gs_urb,     "gs"),
    }

    eq_label = {"pop": "g_pop_ann", "crop": "g_crop_ann", "urb": "g_urb_ann"}
    lagged_lev = {"pop": "log_pop", "crop": "log_crop_share", "urb": "log_urban_share"}

    print("\n" + "=" * 90)
    print("POOLED 3-equation joint VAR — annual vs GS")
    print("=" * 90)

    vssi_changes = []      # collect |pct change| in VSSI across all equations
    sig_flips = []         # collect any significance flips on climate or VSSI terms

    for eq in ("pop", "crop", "urb"):
        lev = lagged_lev[eq]
        print(f"\n--- Equation: {eq_label[eq]} ---")
        hdr = (f"{'regressor':<22s}  {'annual β':>12s} {'p':>6s} {'sig':>3s}  "
               f"{'ann|GSsamp β':>13s} {'p':>6s} {'sig':>3s}  "
               f"{'GS β':>12s} {'p':>6s} {'sig':>3s}  {'Δ(ann→GS)':>10s}")
        print(hdr)
        print("-" * len(hdr))

        reg_pairs = [
            (lev,          lev),
            ("t_bar_dev",  "t_bar_gs_dev"),
            ("p_bar_dev",  "p_bar_gs_dev"),
            ("t_sd_dev",   "t_sd_gs_dev"),
            ("vssi_int",   "vssi_int"),
        ]
        for reg_ann, reg_gs in reg_pairs:
            ann_b = res["annual_full"][eq]["params"].get(reg_ann, np.nan)
            ann_p = res["annual_full"][eq]["p"].get(reg_ann, np.nan)
            asb_b = res["annual_gssamp"][eq]["params"].get(reg_ann, np.nan)
            asb_p = res["annual_gssamp"][eq]["p"].get(reg_ann, np.nan)
            gs_b  = res["gs"][eq]["params"].get(reg_gs, np.nan)
            gs_p  = res["gs"][eq]["p"].get(reg_gs, np.nan)

            ann_sig = _sig_label(ann_p)
            asb_sig = _sig_label(asb_p)
            gs_sig  = _sig_label(gs_p)
            pct     = _rel_change(ann_b, gs_b)

            print(f"  {reg_ann:<20s}  {ann_b:>+12.4g} {ann_p:>6.4f} {ann_sig:>3s}  "
                  f"{asb_b:>+13.4g} {asb_p:>6.4f} {asb_sig:>3s}  "
                  f"{gs_b:>+12.4g} {gs_p:>6.4f} {gs_sig:>3s}  {pct:>10s}")

            # Accumulate diagnostics
            if reg_ann == "vssi_int" and not np.isnan(ann_b) and ann_b != 0:
                pct_num = abs(gs_b - ann_b) / abs(ann_b) * 100
                vssi_changes.append((eq, pct_num))
            if reg_ann in ("t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"):
                if ann_sig != gs_sig and not (ann_sig == "" and gs_sig == ""):
                    sig_flips.append((eq, reg_ann, ann_sig, gs_sig))

        print(f"  {'N':<20s}  {res['annual_full'][eq]['n']:>12d} {'':>6s} {'':>3s}  "
              f"{res['annual_gssamp'][eq]['n']:>13d} {'':>6s} {'':>3s}  "
              f"{res['gs'][eq]['n']:>12d}")
        print(f"  {'R²':<20s}  {res['annual_full'][eq]['r2']:>12.4f} {'':>6s} {'':>3s}  "
              f"{res['annual_gssamp'][eq]['r2']:>13.4f} {'':>6s} {'':>3s}  "
              f"{res['gs'][eq]['r2']:>12.4f}")

    # Pathway-stratified VSSI on g_pop_ann
    print("\n" + "=" * 90)
    print("PATHWAY-STRATIFIED VSSI on g_pop_ann — annual vs GS")
    print("=" * 90)
    print(f"{'pathway':<28s}  {'N_ann':>6s}  "
          f"{'VSSI β_ann':>12s} {'p':>6s} {'sig':>3s}  "
          f"{'VSSI β_GS':>12s} {'p':>6s} {'sig':>3s}  {'Δ':>8s}")

    pathway_vssi_changes = []
    for cl in sorted(panel["cluster"].unique()):
        ann_sub = panel[panel["cluster"] == cl]
        gs_sub  = gs_sample[gs_sample["cluster"] == cl]
        if len(ann_sub) < 25 or len(gs_sub) < 25:
            continue
        ra = _run_one(ann_sub, "g_pop_ann", annual_pop)
        rg = _run_one(gs_sub,  "g_pop_ann", gs_pop)
        if not ra or not rg:
            continue
        vb_ann = ra["params"].get("vssi_int", np.nan)
        vp_ann = ra["p"].get("vssi_int", np.nan)
        vb_gs  = rg["params"].get("vssi_int", np.nan)
        vp_gs  = rg["p"].get("vssi_int", np.nan)
        pct    = _rel_change(vb_ann, vb_gs)
        if not np.isnan(vb_ann) and vb_ann != 0:
            pathway_vssi_changes.append((PATHWAY_NAMES[cl],
                                         abs(vb_gs - vb_ann) / abs(vb_ann) * 100))
        print(f"  {PATHWAY_NAMES[cl]:<28s}  {ra['n']:>6d}  "
              f"{vb_ann:>+12.4g} {vp_ann:>6.4f} {_sig_label(vp_ann):>3s}  "
              f"{vb_gs:>+12.4g} {vp_gs:>6.4f} {_sig_label(vp_gs):>3s}  {pct:>8s}")

    # -------------------------------------------------------------------------
    # Decision summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DECISION SUMMARY")
    print("=" * 90)

    max_vssi_pooled   = max((v for _, v in vssi_changes), default=0.0)
    max_vssi_pathway  = max((v for _, v in pathway_vssi_changes), default=0.0)
    max_vssi          = max(max_vssi_pooled, max_vssi_pathway)

    print(f"\nVSSI stability (pooled equations):")
    for eq, pct in vssi_changes:
        print(f"  {eq}: {pct:+.1f}% change from annual→GS")
    if pathway_vssi_changes:
        print(f"VSSI stability (pathway-stratified g_pop_ann):")
        for pname, pct in pathway_vssi_changes:
            print(f"  {pname}: {pct:.1f}% change")

    print(f"\nMax VSSI |pct change| across all equations/pathways: {max_vssi:.1f}%")

    if sig_flips:
        print(f"\nSignificance flips detected ({len(sig_flips)}):")
        for eq, reg, s_ann, s_gs in sig_flips:
            print(f"  [{eq}] {reg}: annual={s_ann!r} → GS={s_gs!r}")
    else:
        print("\nNo significance flips on any climate or VSSI term.")

    # Apply decision rule
    if max_vssi > 20 or sig_flips:
        verdict = "GS HELPS"
        reason = (
            f"VSSI moves {max_vssi:.1f}% (threshold 20%) and/or "
            f"{len(sig_flips)} significance flip(s) detected."
        )
    elif max_vssi <= 30 and not sig_flips:
        verdict = "GS DOESN'T HELP"
        reason = (
            f"VSSI moves at most {max_vssi:.1f}% (< 30%) across all equations, "
            "no significance flips. Annual climate is sufficient."
        )
    else:
        verdict = "MIXED"
        reason = (
            f"VSSI moves {max_vssi:.1f}% (between 20-30%) with no significance flips; "
            "borderline case."
        )

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict}")
    print(f"REASON:  {reason}")
    print(f"RECOMMENDATION: {'Proceed with Task 7 (GS substitution in joint VAR).' if verdict == 'GS HELPS' else 'Skip Task 7. Keep annual climate in joint VAR.'}")
    print(f"{'=' * 60}")

    vssi_all_stable = max_vssi <= 20
    print(f"\nVSSI IDENTIFICATION STATUS: {'STABLE' if vssi_all_stable else 'SENSITIVE'} "
          f"(max shift {max_vssi:.1f}%)")
    if vssi_all_stable:
        print("  -> Volcanic-shock identification (§4.1) is robust to climate-control specification.")
    else:
        print("  -> WARNING: VSSI coefficient sensitive to annual vs GS climate control choice.")


if __name__ == "__main__":
    main()
