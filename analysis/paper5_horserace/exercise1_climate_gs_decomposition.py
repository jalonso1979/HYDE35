"""Exercise 1c: Growing-season decomposition of the climate bundle.

The long_shadow §5 companion paper shows that pre-industrial σ_v^T's
predictive content is climate-deep rather than growing-season-specific:
non-GS volatility dominates GS volatility by 1.8x–17x in all cross-section
cells. This exercise tests whether the same holds for the horserace panel
across all 6 outcomes.

Two analyses:
  (A) Bundle-level comparison: replicate the headline 5-substrate Shapley
      but with the climate bundle extended from 4 to 5 elements by splitting
      σ_v^T into σ_v^T_GS and σ_v^T_nonGS. Expected: bundle-level Shapley
      R² is nearly identical (the bundle absorbs both seasons).

  (B) Within-bundle sub-decomposition: for each outcome, decompose the 5-
      element GS bundle's contribution across its 5 primitives, conditioning
      on the 4 non-climate substrates + geography controls. Documents which
      primitive (σ_v^T_GS vs σ_v^T_nonGS) carries the bundle's contribution.

Inputs:
  analysis/data/deep_determinants_horserace_gs.parquet
  (built by build_gs_climate_extension.py)

Outputs:
  analysis/data/deep_determinants/exercise1_climate_gs_shapley_results.parquet
    long-form: 6 outcomes × 5 substrates = 30 rows (bundle-level Shapley A)
  analysis/data/deep_determinants/exercise1_climate_gs_subshapley_results.parquet
    long-form: 6 outcomes × 5 GS bundle primitives = 30 rows (within-bundle B)
  analysis/figures/paper5_horserace/tab_gs_bundle_comparison.tex
  analysis/figures/paper5_horserace/tab_gs_subshapley.tex
"""
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.shapley import shapley_r2_decomposition

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL_GS = ROOT / "analysis/data/deep_determinants_horserace_gs.parquet"
OUT_BUNDLE = ROOT / "analysis/data/deep_determinants/exercise1_climate_gs_shapley_results.parquet"
OUT_SUB = ROOT / "analysis/data/deep_determinants/exercise1_climate_gs_subshapley_results.parquet"

# ── GS-decomposed climate bundle (5 primitives, split σ_v^T into GS + nonGS) ──
CLIMATE_BUNDLE_GS = (
    "t_mean_pre1750",
    "p_mean_pre1750",
    "sigma_v_T_gs_pre1750_cropw",
    "sigma_v_T_nongs_pre1750_cropw",
    "sigma_v_P_gs_pre1750_cropw",
)

FUNCTIONAL_BUNDLE = (
    "fa_lct", "fa_adh1b", "fa_amy1", "fa_edar",
    "fa_darc", "fa_slc24a5", "fa_hbb", "fa_fads",
)

# Full 5-substrate spec with GS climate bundle
SUBSTRATES_GS = [
    CLIMATE_BUNDLE_GS,
    FUNCTIONAL_BUNDLE,
    "neolithic_frac",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]

SUBSTRATE_KEYS = [
    "climate_bundle_gs",
    "functional_alleles",
    "neolithic_frac",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]

OUTCOMES = [
    "log_popd_1500",
    "log_popd_2025",
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]

CONTROLS = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]

# Non-climate substrates (used as additional controls for within-bundle sub-decomposition)
NON_CLIMATE_SUBSTRATES = [
    "fa_lct", "fa_adh1b", "fa_amy1", "fa_edar",
    "fa_darc", "fa_slc24a5", "fa_hbb", "fa_fads",
    "neolithic_frac",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]

# Baseline 4-element climate bundle (original Shapley exercise)
CLIMATE_BUNDLE_4 = (
    "t_mean_pre1750",
    "p_mean_pre1750",
    "sigma_v_T_pre1750",
    "sigma_v_P_pre1750",
)

# Baseline 4-element results on the *full* sample (162 obs), pre-computed
BASELINE_CLIMATE_SHAPLEY_FULL = {
    "log_popd_1500":            0.0875,
    "log_popd_2025":            0.0101,
    "log_pop_growth_1950_2025": 0.0766,
    "urban_change_1950_2025":   0.0759,
    "log_gdppc_2015":           0.0102,
    "dt_timing_year":           0.0269,
}


def main() -> None:
    df = pd.read_parquet(PANEL_GS)
    print(f"Loaded GS-extended panel: {len(df)} rows, {df.shape[1]} columns")

    # ── Sample used for GS analysis (restrict to countries with GS data) ─────
    # Countries with year-round growing season (n_gs_months=12) have no nonGS
    # months; sigma_v_T_nongs is undefined for them. The restricted sample
    # excludes these 55 tropical/island countries.
    gs_required = list(CLIMATE_BUNDLE_GS) + list(FUNCTIONAL_BUNDLE) + [
        "neolithic_frac", "ancestral_yield_log", "pandemic_intensity_norm"
    ] + CONTROLS
    sample_sizes = {}
    for outcome in OUTCOMES:
        sub = df.dropna(subset=[outcome] + gs_required)
        sample_sizes[outcome] = len(sub)
    print("\nGS sample sizes (after dropping countries with all-year growing season):")
    for o, n in sample_sizes.items():
        print(f"  {o}: {n} obs")

    # ── (A1) 4-element baseline on the GS-restricted sample ──────────────────
    print("\n" + "=" * 70)
    print("(A1) 4-element baseline Shapley on GS-restricted sample")
    print("=" * 70)

    SUBSTRATES_4 = [
        CLIMATE_BUNDLE_4,
        FUNCTIONAL_BUNDLE,
        "neolithic_frac",
        "ancestral_yield_log",
        "pandemic_intensity_norm",
    ]

    baseline_restricted = {}
    for outcome in OUTCOMES:
        result4 = shapley_r2_decomposition(
            df,
            y_col=outcome,
            substrates=SUBSTRATES_4,
            controls=CONTROLS,
        )
        raw4_key = "+".join(CLIMATE_BUNDLE_4)
        baseline_restricted[outcome] = result4["shapley"][raw4_key]
        print(f"  {outcome}: climate Shapley R² (4-elem, restricted) = {baseline_restricted[outcome]:.4f}  n={result4['n_obs']}")

    # ── (A2) Bundle-level Shapley with 5-element GS bundle ───────────────────
    print("\n" + "=" * 70)
    print("(A2) Bundle-level Shapley: 5-element GS bundle (same sample)")
    print("=" * 70)

    bundle_rows = []
    bundle_climate_gs = {}  # outcome -> GS bundle Shapley R²

    for outcome in OUTCOMES:
        print(f"\n  Outcome: {outcome}")
        result = shapley_r2_decomposition(
            df,
            y_col=outcome,
            substrates=SUBSTRATES_GS,
            controls=CONTROLS,
        )
        # Map raw keys to canonical substrate keys
        raw_climate_key = "+".join(CLIMATE_BUNDLE_GS)
        raw_func_key = "+".join(FUNCTIONAL_BUNDLE)
        key_map = {
            raw_climate_key: "climate_bundle_gs",
            raw_func_key: "functional_alleles",
            "neolithic_frac": "neolithic_frac",
            "ancestral_yield_log": "ancestral_yield_log",
            "pandemic_intensity_norm": "pandemic_intensity_norm",
        }
        for raw_key, phi in result["shapley"].items():
            canonical = key_map[raw_key]
            bundle_rows.append({
                "outcome": outcome,
                "substrate": canonical,
                "shapley_r2": phi,
                "baseline_r2": result["baseline_r2"],
                "full_model_r2": result["full_model_r2"],
                "n_obs": result["n_obs"],
            })
            if canonical == "climate_bundle_gs":
                bundle_climate_gs[outcome] = phi

    out_bundle_df = pd.DataFrame(bundle_rows)
    out_bundle_df.to_parquet(OUT_BUNDLE, index=False)
    print(f"\nWrote {OUT_BUNDLE} ({len(out_bundle_df)} rows)")

    # Print comparison table
    print("\nBundle-level climate Shapley R² comparison:")
    print(f"  {'Outcome':<30} {'4-elem (full,162)':>18} {'4-elem (restr.)':>16} {'5-elem GS':>12}")
    print("  " + "-" * 80)
    for outcome in OUTCOMES:
        base_full = BASELINE_CLIMATE_SHAPLEY_FULL[outcome]
        base_restr = baseline_restricted[outcome]
        gs_val = bundle_climate_gs[outcome]
        print(f"  {outcome:<30} {base_full:>18.4f} {base_restr:>16.4f} {gs_val:>12.4f}")

    # ── (B) Within-bundle sub-decomposition ──────────────────────────────────
    print("\n" + "=" * 70)
    print("(B) Within-bundle GS sub-decomposition (σ_v^T_GS vs σ_v^T_nonGS)")
    print("    Controls: geography + all non-climate substrates")
    print("=" * 70)

    sub_rows = []

    for outcome in OUTCOMES:
        print(f"\n  Outcome: {outcome}")
        result = shapley_r2_decomposition(
            df,
            y_col=outcome,
            substrates=list(CLIMATE_BUNDLE_GS),
            controls=CONTROLS + NON_CLIMATE_SUBSTRATES,
        )
        for prim in CLIMATE_BUNDLE_GS:
            phi = result["shapley"][prim]
            sub_rows.append({
                "outcome": outcome,
                "climate_var": prim,
                "shapley_r2": phi,
                "baseline_r2": result["baseline_r2"],
                "full_model_r2": result["full_model_r2"],
                "n_obs": result["n_obs"],
            })
            print(f"    {prim:<40} {phi:.4f}")

    out_sub_df = pd.DataFrame(sub_rows)
    out_sub_df.to_parquet(OUT_SUB, index=False)
    print(f"\nWrote {OUT_SUB} ({len(out_sub_df)} rows)")

    # Print pivot table
    pivot = out_sub_df.pivot(index="climate_var", columns="outcome", values="shapley_r2")
    pivot = pivot.reindex(index=list(CLIMATE_BUNDLE_GS), columns=OUTCOMES)
    print("\nWithin-bundle GS Shapley R² (conditional on non-climate substrates + controls):")
    print(pivot.round(4).to_string())

    # Check long_shadow finding: nonGS dominates GS?
    print("\nσ_v^T_nonGS / σ_v^T_GS ratio (>1 means nonGS dominates):")
    for outcome in OUTCOMES:
        gs_val = pivot.loc["sigma_v_T_gs_pre1750_cropw", outcome]
        nongs_val = pivot.loc["sigma_v_T_nongs_pre1750_cropw", outcome]
        if abs(gs_val) > 1e-6:
            ratio = nongs_val / gs_val
            print(f"  {outcome:<35}: nonGS={nongs_val:.4f}, GS={gs_val:.4f}, ratio={ratio:.2f}")
        else:
            print(f"  {outcome:<35}: nonGS={nongs_val:.4f}, GS={gs_val:.4f}, ratio=N/A (GS≈0)")

    # Emit LaTeX tables
    _emit_bundle_comparison_table(
        bundle_climate_gs,
        baseline_restricted,
        ROOT / "analysis/figures/paper5_horserace/tab_gs_bundle_comparison.tex",
    )
    _emit_subshapley_table(
        out_sub_df,
        ROOT / "analysis/figures/paper5_horserace/tab_gs_subshapley.tex",
    )


def _emit_bundle_comparison_table(bundle_climate_gs: dict, baseline_restricted: dict, out: Path) -> None:
    """Three-column table: 4-elem (restricted sample) vs 5-elem GS bundle Shapley R²."""
    outcome_labels = {
        "log_popd_1500":            r"$\ln D_{1500}$",
        "log_popd_2025":            r"$\ln D_{2025}$",
        "log_pop_growth_1950_2025": r"$\Delta\ln P_{1950\text{--}2025}$",
        "urban_change_1950_2025":   r"$\Delta\text{Urb}_{1950\text{--}2025}$",
        "log_gdppc_2015":           r"$\ln\text{GDPpc}_{2015}$",
        "dt_timing_year":           r"DT timing (year)",
    }
    with open(out, "w") as f:
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Outcome & 4-prim.~(full, $N$=162) & 4-prim.~(GS sample) & 5-prim.~GS bundle\\\\\n")
        f.write("\\midrule\n")
        for outcome in OUTCOMES:
            base_full = BASELINE_CLIMATE_SHAPLEY_FULL[outcome]
            base_restr = baseline_restricted[outcome]
            gs_val = bundle_climate_gs[outcome]
            label = outcome_labels.get(outcome, outcome)
            f.write(f"{label} & {base_full:.4f} & {base_restr:.4f} & {gs_val:.4f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {out}")


def _emit_subshapley_table(out_df: pd.DataFrame, out: Path) -> None:
    """Within-bundle GS sub-decomposition table (5 primitives × 6 outcomes)."""
    pivot = out_df.pivot(index="climate_var", columns="outcome", values="shapley_r2")
    pivot = pivot.reindex(index=list(CLIMATE_BUNDLE_GS), columns=OUTCOMES)

    label_map = {
        "t_mean_pre1750":                r"Mean T ($\bar T$)",
        "p_mean_pre1750":                r"Mean P ($\bar P$)",
        "sigma_v_T_gs_pre1750_cropw":    r"T volatility, GS ($\sigma_v^{T,\mathrm{GS}}$)",
        "sigma_v_T_nongs_pre1750_cropw": r"T volatility, non-GS ($\sigma_v^{T,\overline{\mathrm{GS}}}$)",
        "sigma_v_P_gs_pre1750_cropw":    r"P volatility, GS ($\sigma_v^{P,\mathrm{GS}}$)",
    }
    outcome_labels = {
        "log_popd_1500":            r"$\ln D_{1500}$",
        "log_popd_2025":            r"$\ln D_{2025}$",
        "log_pop_growth_1950_2025": r"$\Delta\ln P$",
        "urban_change_1950_2025":   r"$\Delta\text{Urb}$",
        "log_gdppc_2015":           r"$\ln\text{GDPpc}$",
        "dt_timing_year":           r"DT yr",
    }
    with open(out, "w") as f:
        col_spec = "l" + "r" * len(OUTCOMES)
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        f.write(" & " + " & ".join(outcome_labels.get(o, o) for o in OUTCOMES) + " \\\\\n")
        f.write("\\midrule\n")
        for prim in CLIMATE_BUNDLE_GS:
            row = " & ".join(f"{pivot.loc[prim, o]:.4f}" for o in OUTCOMES)
            f.write(label_map.get(prim, prim) + " & " + row + " \\\\\n")
        f.write("\\midrule\n")
        totals = " & ".join(f"{pivot[o].sum():.4f}" for o in OUTCOMES)
        f.write("Bundle total & " + totals + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
