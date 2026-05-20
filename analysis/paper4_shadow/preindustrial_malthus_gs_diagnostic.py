"""Diagnostic: does GS-mean climate matter for the preindustrial Malthus regression?

Runs side-by-side comparison of the existing annual-climate Malthus spec
against the GS-mean-climate analogue. Reports whether GS substitution
meaningfully changes the coefficient signs/magnitudes/significance.

If GS shifts any of {β_T, β_P, β_Tstd, β_Pstd} by >50% in magnitude AND
flips significance status on any, GS is worth running across Tasks 6-9.
Otherwise skip.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

from analysis.paper4_shadow.preindustrial_malthus import (
    _load_hyde_intervals, _attach_pathway, _fe_regression, PATHWAY_NAMES,
    START_YEAR, END_YEAR,
)


def _attach_annual_interval_climate(intervals: pd.DataFrame) -> pd.DataFrame:
    """As in preindustrial_malthus._attach_interval_climate — annual full-year T/P."""
    annual = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    annual = annual[(annual["year"] >= START_YEAR) & (annual["year"] <= END_YEAR)]
    annual = annual[["iso3", "year", "t_c", "p_mm"]]
    out = []
    for _, row in intervals.iterrows():
        sub = annual[(annual["iso3"] == row["iso3"])
                     & (annual["year"] >= row["year"])
                     & (annual["year"] < row["year_next"])]
        if len(sub) < 3:
            continue
        out.append({"iso3": row["iso3"],
                    "year": int(row["year"]),
                    "year_next": int(row["year_next"]),
                    "log_density": row["log_density"],
                    "pop_growth_ann": row["pop_growth_ann"],
                    "t_mean_int": sub["t_c"].mean(),
                    "t_std_int": sub["t_c"].std(),
                    "p_mean_int": sub["p_mm"].mean(),
                    "p_std_int": sub["p_mm"].std()})
    return pd.DataFrame(out)


def _attach_gs_interval_climate(intervals: pd.DataFrame) -> pd.DataFrame:
    """GS-mean analogue — uses t_gs_mean_cropw, p_gs_mean_cropw."""
    gs = pd.read_parquet(DATA / "country_climate_gs_1421_2025.parquet")
    gs = gs[(gs["year"] >= START_YEAR) & (gs["year"] <= END_YEAR)]
    gs = gs[["iso3", "year", "t_gs_mean_cropw", "p_gs_mean_cropw"]].dropna()
    out = []
    for _, row in intervals.iterrows():
        sub = gs[(gs["iso3"] == row["iso3"])
                 & (gs["year"] >= row["year"])
                 & (gs["year"] < row["year_next"])]
        if len(sub) < 3:
            continue
        out.append({"iso3": row["iso3"],
                    "year": int(row["year"]),
                    "year_next": int(row["year_next"]),
                    "log_density": row["log_density"],
                    "pop_growth_ann": row["pop_growth_ann"],
                    "t_mean_int": sub["t_gs_mean_cropw"].mean(),
                    "t_std_int": sub["t_gs_mean_cropw"].std(),
                    "p_mean_int": sub["p_gs_mean_cropw"].mean(),
                    "p_std_int": sub["p_gs_mean_cropw"].std()})
    return pd.DataFrame(out)


def _add_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t_anom_int"] = df["t_mean_int"] - df.groupby("iso3")["t_mean_int"].transform("mean")
    df["p_anom_int"] = df["p_mean_int"] - df.groupby("iso3")["p_mean_int"].transform("mean")
    return df


def main() -> None:
    intervals = _load_hyde_intervals()
    print(f"Intervals: {len(intervals)} country-pairs, {intervals['iso3'].nunique()} countries")

    annual_df = _attach_annual_interval_climate(intervals)
    annual_df = _attach_pathway(annual_df)
    annual_df = _add_anomalies(annual_df)
    gs_df = _attach_gs_interval_climate(intervals)
    gs_df = _attach_pathway(gs_df)
    gs_df = _add_anomalies(gs_df)

    print(f"Annual sample: {len(annual_df)} rows, {annual_df['iso3'].nunique()} countries")
    print(f"GS sample:     {len(gs_df)} rows, {gs_df['iso3'].nunique()} countries")

    # ---- Sample comparability: run annual spec on GS-restricted sample ----
    gs_isos = set(gs_df["iso3"].unique())
    gs_years = set(zip(gs_df["iso3"], gs_df["year"]))
    annual_on_gs = annual_df[annual_df.apply(
        lambda r: (r["iso3"], r["year"]) in gs_years, axis=1
    )].copy()
    print(f"Annual-on-GS-sample: {len(annual_on_gs)} rows, {annual_on_gs['iso3'].nunique()} countries")
    n_dropped_countries = annual_df["iso3"].nunique() - annual_on_gs["iso3"].nunique()
    n_dropped_rows = len(annual_df) - len(annual_on_gs)
    print(f"  => GS restriction drops {n_dropped_countries} countries, {n_dropped_rows} rows")
    if n_dropped_rows > 0:
        print("  NOTE: sample-difference effect will be isolated below.")

    regs = ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"]

    print("\n=== Pooled FE Malthus (existing spec) — comparison ===")
    print(f"{'regressor':<15s}  {'annual β':>12s}  {'annual p':>10s}  {'ann_GS-samp β':>14s}  {'ann_GS-samp p':>14s}  {'GS β':>12s}  {'GS p':>10s}  Δβ%(ann→GS)  Δβ%(samp→GS)")
    a = _fe_regression(annual_df, regs)
    a_gs = _fe_regression(annual_on_gs, regs)
    g = _fe_regression(gs_df, regs)
    print(f"{'(N rows)':<15s}  {a['n']:>12d}  {'':>10s}  {a_gs['n']:>14d}  {'':>14s}  {g['n']:>12d}")
    print(f"{'(R²)':<15s}  {a['rsquared']:>12.4f}  {'':>10s}  {a_gs['rsquared']:>14.4f}  {'':>14s}  {g['rsquared']:>12.4f}")
    print()
    sig_flips = []
    large_shifts = []
    for v in regs:
        ab = a["params"][v];    ap = a["pvalues"][v]
        ab_gs = a_gs["params"][v]; ap_gs = a_gs["pvalues"][v]
        gb = g["params"][v];    gp = g["pvalues"][v]
        pct_ann_gs = 100 * (gb - ab) / ab if abs(ab) > 1e-12 else float("nan")
        pct_samp_gs = 100 * (gb - ab_gs) / ab_gs if abs(ab_gs) > 1e-12 else float("nan")
        print(f"  {v:<13s}  {ab:>+12.5g}  {ap:>10.4f}  {ab_gs:>+14.5g}  {ap_gs:>14.4f}  {gb:>+12.5g}  {gp:>10.4f}  {pct_ann_gs:>+10.1f}%  {pct_samp_gs:>+10.1f}%")
        # Track significance flips (annual → GS on same-sample comparison)
        a_sig = ap_gs < 0.05
        g_sig = gp < 0.05
        if a_sig != g_sig and v != "log_density":
            sig_flips.append(v)
        if abs(pct_samp_gs) > 50 and v != "log_density":
            large_shifts.append(v)

    print("\n=== Pathway-stratified — for each pathway, does GS shift anything? ===")
    pathway_flips = []
    pathway_large = []
    for cluster_id in sorted(annual_df["cluster"].unique()):
        a_sub = annual_df[annual_df["cluster"] == cluster_id]
        a_gs_sub = annual_on_gs[annual_on_gs["cluster"] == cluster_id] if len(annual_on_gs) > 0 else pd.DataFrame()
        g_sub = gs_df[gs_df["cluster"] == cluster_id]
        if len(a_sub) < 25 or len(g_sub) < 25:
            continue
        ra = _fe_regression(a_sub, regs)
        ra_gs = _fe_regression(a_gs_sub, regs) if len(a_gs_sub) >= 25 else {}
        rg = _fe_regression(g_sub, regs)
        if not ra or not rg:
            continue
        pname = PATHWAY_NAMES[cluster_id]
        n_ann_gs = ra_gs.get('n', 'n/a') if ra_gs else 'n/a'
        print(f"\n  -- {pname} (N_annual={ra['n']}, N_ann_GS-samp={n_ann_gs}, N_gs={rg['n']}) --")
        print(f"    {'regressor':<15s}  {'annual β':>12s}  {'ann_GS-samp β':>14s}  {'GS β':>12s}  Δβ%(samp→GS)  GSp  sig_flip?")
        for v in regs:
            ab = ra["params"][v]
            ab_gs_v = ra_gs["params"][v] if ra_gs else float("nan")
            gb = rg["params"][v]; gp = rg["pvalues"][v]
            ap_gs_v = ra_gs["pvalues"][v] if ra_gs else float("nan")
            pct = 100 * (gb - ab_gs_v) / ab_gs_v if abs(ab_gs_v) > 1e-12 else float("nan")
            g_sig = gp < 0.05
            a_sig_v = ap_gs_v < 0.05 if ra_gs else False
            flip = "FLIP" if (g_sig != a_sig_v) and v != "log_density" else ""
            print(f"    {v:<13s}  {ab:>+12.5g}  {ab_gs_v:>+14.5g}  {gb:>+12.5g}  {pct:>+10.1f}%  {gp:.4f}  {flip}")
            if flip:
                pathway_flips.append(f"{pname}:{v}")
            if abs(pct) > 50 and v != "log_density":
                pathway_large.append(f"{pname}:{v}")

    # ---- Decision summary ----
    print("\n" + "=" * 70)
    print("DECISION SUMMARY")
    print("=" * 70)
    print(f"Pooled: large shifts (>50%, sample-controlled): {large_shifts or 'none'}")
    print(f"Pooled: significance flips (sample-controlled): {sig_flips or 'none'}")
    print(f"Pathway: large shifts: {pathway_large or 'none'}")
    print(f"Pathway: significance flips: {pathway_flips or 'none'}")
    n_flips = len(sig_flips) + len(pathway_flips)
    n_large = len(large_shifts) + len(pathway_large)
    if n_flips > 0 and n_large > 0:
        verdict = "GS HELPS — at least one coefficient shifts >50% AND flips significance."
    elif n_large > 0:
        verdict = "MIXED — large magnitude shifts but no significance flips."
    else:
        verdict = "GS DOES NOT HELP — all coefficients within 50% of annual-on-GS-sample, no sig flips."
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
