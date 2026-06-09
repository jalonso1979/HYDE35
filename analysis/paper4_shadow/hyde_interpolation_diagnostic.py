"""HYDE-interpolation robustness diagnostic for the joint-VAR demographic slope.

Concern
-------
HYDE 3.5 reports benchmark population on 100-yr steps pre-1700 and decadal
steps post-1700, but *within* a long inter-benchmark span it geometrically
interpolates population between the two bracketing benchmark years. On a log
scale, geometric interpolation produces a *constant* annualised log-growth
rate across every interval inside the span. So a run of consecutive
within-country decades carrying an identical annualised growth value to high
precision is a signature of pure interpolation: the demographic LHS is inert
(mechanically pinned by two distant benchmarks) while the volcanic VSSI RHS
still varies decade to decade. Those cells can only add noise/attenuation to
the VSSI demographic slope, never genuine demographic response.

This script
-----------
1. Loads analysis/data/joint_landuse_var_panel.parquet.
2. Flags each (country, interval) cell as an interpolation artifact when its
   annualised pop-growth equals the *previous* within-country interval's value
   to ~6 dp (the geometric-interpolation tell). Reports the interpolated share
   overall and for the median crop-dominant-late (cluster 0) country, plus a
   few example country series so the flag can be eyeballed.
3. Re-estimates the crop-dominant-late VSSI *demographic* slope on
   non-interpolated decades only, using the EXACT 7-control within-country FE
   OLS with iso3-clustered SEs from joint_landuse_var.py._run_one.

Run:
    python3 -m analysis.paper4_shadow.hyde_interpolation_diagnostic
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

# Headline spec, copied verbatim from joint_landuse_var.py
CONTROLS = ["log_pop", "log_crop_share", "log_urban_share",
            "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]
CROP_DOMINANT_LATE = 0          # cluster id for "Crop-dominant late"
HEADLINE_BETA = -9.65e-5        # reported full-sample VSSI demographic slope
ROUND_DP = 6                    # precision for the "identical growth" test


def _run_one(d: pd.DataFrame, lhs: str, controls: list[str]) -> dict:
    """Within-country FE OLS with country-clustered SEs.

    Byte-for-byte the estimator in joint_landuse_var.py._run_one.
    """
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}


def flag_interpolated(panel: pd.DataFrame, dp: int = ROUND_DP) -> pd.DataFrame:
    """Add `interp` flag: True when this interval's annualised pop growth
    repeats the prior within-country interval's value to `dp` decimals."""
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    gr = panel["g_pop_ann"].round(dp)
    prev = panel.groupby("iso3")["g_pop_ann"].shift(1).round(dp)
    same_country = panel["iso3"].eq(panel.groupby("iso3")["iso3"].shift(1))
    panel["interp"] = same_country & gr.eq(prev) & gr.notna() & prev.notna()
    return panel


def main() -> None:
    print("=== HYDE-interpolation diagnostic: joint-VAR demographic slope ===\n")
    panel = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    panel = flag_interpolated(panel)

    # ---- 1. Interpolated share -------------------------------------------
    overall = panel["interp"].mean()
    cd = panel[panel["cluster"] == CROP_DOMINANT_LATE].copy()
    cd_overall = cd["interp"].mean()
    per_country = cd.groupby("iso3")["interp"].mean()
    cd_median = per_country.median()

    print(f"Overall interpolated share (all pathways): "
          f"{overall:.1%}  ({panel['interp'].sum()}/{len(panel)} cells)")
    print(f"Crop-dominant-late pooled interpolated share: "
          f"{cd_overall:.1%}  ({cd['interp'].sum()}/{len(cd)} cells)")
    print(f"Crop-dominant-late MEDIAN-country interpolated share: "
          f"{cd_median:.1%}")
    print(f"  (per-country share: min={per_country.min():.1%}, "
          f"q25={per_country.quantile(.25):.1%}, "
          f"median={cd_median:.1%}, "
          f"q75={per_country.quantile(.75):.1%}, "
          f"max={per_country.max():.1%}, n_countries={per_country.size})")

    # ---- Sanity: show a few crop-dominant country series ------------------
    print("\n--- Flag sanity check: example crop-dominant-late series ---")
    examples = (per_country.sort_values(ascending=False).index[:1].tolist()
                + per_country.sort_values().index[:1].tolist())
    # plus a mid-share country
    mid = (per_country - cd_median).abs().sort_values().index[0]
    if mid not in examples:
        examples.append(mid)
    for iso in examples:
        s = cd[cd["iso3"] == iso].sort_values("year")
        print(f"\n  {iso}  (interp share {per_country[iso]:.0%}):")
        view = s[["year", "next_year", "dt", "pop", "g_pop_ann", "interp"]].copy()
        view["g_pop_ann"] = view["g_pop_ann"].round(6)
        print(view.to_string(index=False))

    # ---- 2-3. Re-estimate on non-interpolated decades --------------------
    print("\n\n=== Crop-dominant-late VSSI demographic slope ===")

    full = _run_one(cd, "g_pop_ann", CONTROLS)
    noninterp_df = cd[~cd["interp"]].copy()
    noninterp = _run_one(noninterp_df, "g_pop_ann", CONTROLS)

    def fmt(tag, r):
        b = r["params"].get("vssi_int", np.nan)
        se = r["bse"].get("vssi_int", np.nan)
        p = r["p"].get("vssi_int", np.nan)
        print(f"  {tag:<22} β(vssi_int) = {b:+.3e}  "
              f"SE = {se:.3e}  p = {p:.3g}  N = {r['n']}  R² = {r['r2']:.4f}")
        return b, p, r["n"]

    print(f"  {'reported headline':<22} β(vssi_int) = {HEADLINE_BETA:+.3e}"
          f"  (paper.tex value)")
    b_full, p_full, n_full = fmt("full sample (repro)", full)
    b_ni, p_ni, n_ni = fmt("non-interpolated only", noninterp)

    # ---- Verdict ---------------------------------------------------------
    sign_ok = (np.sign(b_ni) == np.sign(b_full))
    sig_full = p_full < 0.05
    sig_ni = p_ni < 0.05
    ratio = b_ni / b_full if b_full != 0 else np.nan
    pct_change = (b_ni - b_full) / abs(b_full) * 100

    print("\n=== VERDICT ===")
    print(f"  interpolated: overall {overall:.1%}, "
          f"crop-dominant median country {cd_median:.1%}")
    print(f"  full:            β={b_full:+.3e}  p={p_full:.3g}  N={n_full}")
    print(f"  non-interpolated:β={b_ni:+.3e}  p={p_ni:.3g}  N={n_ni}")
    print(f"  sign preserved: {sign_ok};  "
          f"sig(full)@5%={sig_full};  sig(ni)@5%={sig_ni}")
    print(f"  magnitude ratio (ni/full) = {ratio:.2f}  "
          f"({pct_change:+.0f}% change)")
    surv = "SURVIVES" if (sign_ok and sig_ni) else \
           ("SIGN-ROBUST-BUT-LOSES-SIG" if sign_ok else "FLIPS")
    direc = "amplifies" if abs(b_ni) > abs(b_full) else "attenuates"
    print(f"  ONE-LINE: headline {surv}; dropping the {overall:.0%} "
          f"interpolation artifacts {direc} the slope to {b_ni:+.2e} "
          f"({abs(pct_change):.0f}% {'larger' if direc=='amplifies' else 'smaller'} "
          f"in magnitude), p={p_ni:.2g} on N={n_ni}.")


if __name__ == "__main__":
    main()
