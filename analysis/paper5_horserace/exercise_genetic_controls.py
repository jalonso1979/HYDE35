"""Exercise: genetic ancestry controls (R1b-M269 and Lazaridis components).

Implements Options A and B from the §6.6 robustness exercise:

Option A — full panel (~185 countries): add R1b-M269 frequency as additional
  control. Test whether the 3 FWER-surviving cells survive.

Option B — Lazaridis sub-sample (~70 countries): add Anatolian Neolithic %,
  Yamnaya %, and WHG % as controls. Test FWER survival on the smaller panel.
  Also test whether H_pred coefficient on log GDPpc shrinks when Lazaridis
  components are conditioned in (i.e., whether the AG diversity signal traces
  specifically through farmer/steppe ancestry packages).

Outputs
-------
  analysis/data/deep_determinants/r1b_m269_frequency.parquet   (built separately)
  analysis/data/deep_determinants/lazaridis_ancestry.parquet   (built separately)
  analysis/data/deep_determinants_horserace.parquet            (updated in-place)
  analysis/data/deep_determinants/exercise_genetic_controls.parquet

Source references for §6.6:
  Myres et al. (2011) EJHG 19:95–101
  Balaresque et al. (2010) PLoS Biology 8:e1000285
  Underhill et al. (2014) EJHG 23:124–131
  Lazaridis et al. (2014) Nature 513:409–413
  Lazaridis et al. (2022) Science 377:eabm4247
  Haak et al. (2015) Nature 522:207–211
  Allentoft et al. (2015) Nature 522:167–172
  Narasimhan et al. (2019) Science 365:eaat7487
"""

from __future__ import annotations
import time
import warnings
warnings.simplefilter("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper5_horserace.shapley import shapley_r2_decomposition

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
DEEP_DET = DATA / "deep_determinants"
PANEL_PATH = DATA / "deep_determinants_horserace.parquet"

SUBSTRATES = [
    "sigma_v_T_pre1750",
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]
OUTCOMES = [
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]
CONTROLS_BASE = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]

# ============================================================================
# Step 1: Merge genetic controls into master panel
# ============================================================================

def merge_genetic_controls() -> pd.DataFrame:
    """Merge R1b-M269 and Lazaridis ancestry into master panel."""
    print("Merging genetic controls into master panel ...")
    df = pd.read_parquet(PANEL_PATH)

    r1b = pd.read_parquet(DEEP_DET / "r1b_m269_frequency.parquet")[
        ["iso3", "r1b_m269_pct"]]
    laz = pd.read_parquet(DEEP_DET / "lazaridis_ancestry.parquet")[
        ["iso3", "anatolian_neolithic_pct", "yamnaya_pct", "whg_pct"]]

    # Drop if already present (allow re-run)
    for col in ["r1b_m269_pct", "anatolian_neolithic_pct",
                "yamnaya_pct", "whg_pct"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(r1b, on="iso3", how="left")
    df = df.merge(laz, on="iso3", how="left")

    n_r1b = df["r1b_m269_pct"].notna().sum()
    n_laz = df[["anatolian_neolithic_pct", "yamnaya_pct", "whg_pct"]].notna().all(axis=1).sum()
    print(f"  Panel: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Countries with R1b-M269: {n_r1b}")
    print(f"  Countries with Lazaridis full triple: {n_laz}")

    df.to_parquet(PANEL_PATH, index=False)
    print(f"  Saved -> {PANEL_PATH}")
    return df


# ============================================================================
# Helpers
# ============================================================================

def _pathway_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("pathway_") and c != "pathway_0")


def _ols_result(df: pd.DataFrame, y_col: str, substrate: str,
                pathway_dummies: list[str], controls: list[str]) -> dict:
    """Return dict with t-stat, coef, se, n for substrate in OLS."""
    needed = [y_col, substrate] + pathway_dummies + controls
    d = df.dropna(subset=needed)
    if len(d) < 15:
        return {"t": np.nan, "coef": np.nan, "se": np.nan, "n": len(d)}
    X = sm.add_constant(d[[substrate] + controls + pathway_dummies])
    try:
        res = sm.OLS(d[y_col], X).fit()
        return {
            "t": float(res.tvalues[substrate]),
            "coef": float(res.params[substrate]),
            "se": float(res.bse[substrate]),
            "n": int(res.nobs),
        }
    except Exception:
        return {"t": np.nan, "coef": np.nan, "se": np.nan, "n": len(d)}


def _westfall_young(df: pd.DataFrame, controls: list[str],
                    n_perm: int = 1000, seed: int = 46) -> pd.DataFrame:
    """Westfall-Young FWER across 16 substrate×outcome cells."""
    pw_cols = _pathway_cols(df)
    cells = [(out, sub) for out in OUTCOMES for sub in SUBSTRATES]

    # Observed |t|
    obs_t = {}
    for out, sub in cells:
        r = _ols_result(df, out, sub, pw_cols, controls)
        obs_t[(out, sub)] = abs(r["t"]) if not np.isnan(r["t"]) else 0.0

    # Pre-filter data per cell
    cell_dfs = {}
    for out, sub in cells:
        needed = [out, sub] + pw_cols + controls
        cell_dfs[(out, sub)] = df.dropna(subset=needed).copy()

    # Permutation null
    rng = np.random.default_rng(seed)
    max_t_perm = np.zeros(n_perm)
    for p in range(n_perm):
        perm_t_vals = []
        for out, sub in cells:
            d = cell_dfs[(out, sub)].copy()
            d[out] = rng.permutation(d[out].values)
            r = _ols_result(d, out, sub, _pathway_cols(d), controls)
            perm_t_vals.append(abs(r["t"]) if not np.isnan(r["t"]) else 0.0)
        max_t_perm[p] = max(perm_t_vals)
        if (p + 1) % 200 == 0:
            print(f"    WY: {p+1}/{n_perm} permutations ...")

    rows = []
    for out, sub in cells:
        t_obs = obs_t[(out, sub)]
        p_adj = float(np.mean(max_t_perm >= t_obs))
        result = _ols_result(df, out, sub, pw_cols, controls)
        rows.append({
            "outcome": out, "substrate": sub,
            "t_obs": t_obs, "p_adj_wy": p_adj,
            "coef": result["coef"], "se": result["se"],
            "n_obs": result["n"],
        })
    return pd.DataFrame(rows)


# ============================================================================
# Option A: Full panel + R1b-M269
# ============================================================================

def run_option_a(df: pd.DataFrame, n_perm: int = 1000) -> pd.DataFrame:
    """Full panel (~185 countries) with R1b-M269 as additional control."""
    print("\n" + "=" * 60)
    print("OPTION A: Full panel + R1b-M269 control")
    print("=" * 60)
    controls_r1b = CONTROLS_BASE + ["r1b_m269_pct"]

    t0 = time.time()
    wy = _westfall_young(df, controls=controls_r1b, n_perm=n_perm)
    wy["spec"] = "option_a_r1b"
    print(f"  Option A FWER done in {time.time()-t0:.0f}s")

    print("\n  Results (top cells by |t|):")
    print(f"  {'outcome':30s} {'substrate':25s} {'|t|':>7} {'p_WY':>7} {'n':>5}")
    for _, r in wy.sort_values("t_obs", ascending=False).iterrows():
        star = " ***" if r["p_adj_wy"] < 0.05 else ""
        print(f"  {r['outcome']:30s} {r['substrate']:25s} {r['t_obs']:7.3f}"
              f" {r['p_adj_wy']:7.3f} {r['n_obs']:5.0f}{star}")
    return wy


# ============================================================================
# Option B: Lazaridis sub-sample + ENF + Yamnaya + WHG
# ============================================================================

def run_option_b(df: pd.DataFrame, n_perm: int = 1000) -> pd.DataFrame:
    """Lazaridis sub-sample with Anatolian Neolithic, Yamnaya, WHG controls."""
    print("\n" + "=" * 60)
    print("OPTION B: Lazaridis sub-sample (~70 countries) + ENF + Yamnaya + WHG")
    print("=" * 60)
    laz_cols = ["anatolian_neolithic_pct", "yamnaya_pct", "whg_pct"]
    controls_laz = CONTROLS_BASE + laz_cols

    # Sub-sample: non-missing on all Lazaridis columns
    df_laz = df.dropna(subset=laz_cols).copy()
    print(f"  Lazaridis sub-sample N={len(df_laz)} countries")

    t0 = time.time()
    wy = _westfall_young(df_laz, controls=controls_laz, n_perm=n_perm)
    wy["spec"] = "option_b_lazaridis"
    print(f"  Option B FWER done in {time.time()-t0:.0f}s")

    print("\n  Results (top cells by |t|):")
    print(f"  {'outcome':30s} {'substrate':25s} {'|t|':>7} {'p_WY':>7} {'n':>5}")
    for _, r in wy.sort_values("t_obs", ascending=False).iterrows():
        star = " ***" if r["p_adj_wy"] < 0.05 else ""
        print(f"  {r['outcome']:30s} {r['substrate']:25s} {r['t_obs']:7.3f}"
              f" {r['p_adj_wy']:7.3f} {r['n_obs']:5.0f}{star}")
    return wy


# ============================================================================
# Substantive test: does H_pred absorb into Lazaridis components?
# ============================================================================

def run_absorption_test(df: pd.DataFrame) -> pd.DataFrame:
    """Test coefficient shrinkage of H_pred on log GDPpc when Lazaridis
    components are added.

    Interpretation:
    - If coef_H shrinks substantially when ENF/Yamnaya/WHG added:
      AG signal traces through specific ancestry packages.
    - If coef_H is stable: H_pred captures something orthogonal to the
      canonical Lazaridis decomposition (e.g., total within-population
      diversity or out-of-Africa bottleneck effects not captured by
      the three-component frame).
    """
    print("\n" + "=" * 60)
    print("ABSORPTION TEST: H_pred_pwadj coefficient on log_gdppc_2015")
    print("=" * 60)

    laz_cols = ["anatolian_neolithic_pct", "yamnaya_pct", "whg_pct"]
    df_laz = df.dropna(subset=laz_cols + ["H_pred_pwadj", "log_gdppc_2015"]
                       + CONTROLS_BASE).copy()
    pw_cols = _pathway_cols(df_laz)

    substrate = "H_pred_pwadj"
    outcome = "log_gdppc_2015"

    rows = []
    specs = [
        ("baseline_full",    CONTROLS_BASE,                          df),
        ("baseline_laz_ss",  CONTROLS_BASE,                          df_laz),
        ("+r1b_laz_ss",      CONTROLS_BASE + ["r1b_m269_pct"],      df_laz),
        ("+enf_yam_whg",     CONTROLS_BASE + laz_cols,              df_laz),
        ("+all_genetic",     CONTROLS_BASE + ["r1b_m269_pct"]
                             + laz_cols,                             df_laz),
    ]
    for label, controls, data in specs:
        r = _ols_result(data.dropna(subset=[outcome, substrate] + controls + pw_cols),
                        outcome, substrate, pw_cols, controls)
        rows.append({"spec": label, "coef": r["coef"], "se": r["se"],
                     "t": r["t"], "n": r["n"]})
        print(f"  {label:22s}: coef={r['coef']:+.4f}  se={r['se']:.4f}"
              f"  |t|={abs(r['t']):.3f}  n={r['n']}")

    # Also test for pop growth
    print(f"\n  H_pred_pwadj on log_pop_growth_1950_2025:")
    outcome2 = "log_pop_growth_1950_2025"
    for label, controls, data in specs:
        r = _ols_result(data.dropna(subset=[outcome2, substrate] + controls + pw_cols),
                        outcome2, substrate, pw_cols, controls)
        print(f"  {label:22s}: coef={r['coef']:+.4f}  se={r['se']:.4f}"
              f"  |t|={abs(r['t']):.3f}  n={r['n']}")
        rows.append({"spec": label + "_popgrowth",
                     "coef": r["coef"], "se": r["se"], "t": r["t"], "n": r["n"]})

    return pd.DataFrame(rows)


# ============================================================================
# Summary table: 3 FWER survivors across specs
# ============================================================================

def print_summary_table(wy_base: pd.DataFrame, wy_a: pd.DataFrame,
                        wy_b: pd.DataFrame) -> None:
    """Print the 3 FWER survivors' z-statistics across three specifications."""
    THREE_CELLS = [
        ("log_pop_growth_1950_2025", "ancestral_yield_log"),
        ("log_gdppc_2015",           "H_pred_pwadj"),
        ("log_pop_growth_1950_2025", "H_pred_pwadj"),
    ]
    LABELS = [
        "ancestral_yield x pop_growth",
        "H_pred x log_gdppc",
        "H_pred x pop_growth",
    ]

    print("\n" + "=" * 80)
    print("THREE FWER-SURVIVING CELLS: z-stats and p_WY across specifications")
    print("=" * 80)
    header = (f"{'Cell':30s} {'|t| base':>10} {'p_WY base':>10}"
              f" {'|t| +R1b':>10} {'p_WY':>8}"
              f" {'|t| +Laz':>10} {'p_WY':>8} {'n_Laz':>6}")
    print(header)
    print("-" * len(header))

    for (out, sub), label in zip(THREE_CELLS, LABELS):
        def _get(wy, o, s):
            row = wy[(wy["outcome"] == o) & (wy["substrate"] == s)]
            if len(row) == 0:
                return np.nan, np.nan, np.nan
            r = row.iloc[0]
            return r["t_obs"], r["p_adj_wy"], r.get("n_obs", np.nan)

        t0, p0, n0 = _get(wy_base, out, sub)
        ta, pa, na = _get(wy_a, out, sub)
        tb, pb, nb = _get(wy_b, out, sub)

        surv0 = "***" if p0 < 0.05 else "   "
        surva = "***" if pa < 0.05 else "   "
        survb = "***" if (not np.isnan(pb) and pb < 0.05) else "   "

        print(f"{label:30s} {t0:10.3f} {p0:10.3f}{surv0}"
              f" {ta:10.3f} {pa:8.3f}{surva}"
              f" {tb:10.3f}{'' if np.isnan(pb) else f' {pb:8.3f}'}{survb}"
              f" {int(nb) if not np.isnan(nb) else 'N/A':>6}")

    print("\n*** = FWER p_WY < 0.05")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Exercise: genetic ancestry controls (R1b-M269 + Lazaridis)")
    print("=" * 70)

    # Merge genetic data into panel
    df = merge_genetic_controls()

    # Load baseline FWER results
    wy_base = pd.read_parquet(DEEP_DET / "robustness_battery.parquet")
    wy_base = wy_base[wy_base["check"] == "wy_correction"].copy()
    wy_base = wy_base.rename(columns={"t_obs": "t_obs", "p_adj_wy": "p_adj_wy"})

    # Option A: full panel + R1b
    wy_a = run_option_a(df, n_perm=1000)

    # Option B: Lazaridis sub-sample
    wy_b = run_option_b(df, n_perm=1000)

    # Absorption test
    absorption = run_absorption_test(df)

    # Summary table
    print_summary_table(wy_base, wy_a, wy_b)

    # Save all results
    out_cols = ["outcome", "substrate", "t_obs", "p_adj_wy", "coef", "se", "n_obs", "spec"]
    results = pd.concat([
        wy_a[out_cols],
        wy_b[out_cols],
    ], ignore_index=True)
    out_path = DEEP_DET / "exercise_genetic_controls.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nSaved results -> {out_path}")

    # Print absorption test summary
    print("\nABSORPTION TEST (H_pred on log_gdppc — coefficient evolution):")
    print(absorption[absorption["spec"].str.contains("popgrowth") == False]
          [["spec", "coef", "se", "t", "n"]].to_string(index=False))

    print("\nDone.")
    return results, wy_base, wy_a, wy_b, absorption


if __name__ == "__main__":
    main()
