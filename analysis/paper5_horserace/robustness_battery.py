# analysis/paper5_horserace/robustness_battery.py
"""Robustness battery: continent FE, leave-one-out, Westfall-Young,
two climate-window placebos, pre-1900 outcome heterogeneity.

Six checks, all consolidated here.

Output: analysis/data/deep_determinants/robustness_battery.parquet
        (long-form: rows tagged with check_name)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper5_horserace.mediation import mediation_share_with_ci
from analysis.paper5_horserace.shapley import shapley_r2_decomposition
from analysis.paper5_horserace.subsamples import add_continent_dummies

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
CLIMATE_RAW = ROOT / "analysis/data/country_climate_1421_2025.parquet"
OUT = ROOT / "analysis/data/deep_determinants/robustness_battery.parquet"

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
CONTROLS = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]

# Pathway dummies: pathway_0 is the reference (omitted); use pathway_1 through pathway_4
PATHWAY_COLS_DEFAULT = ["pathway_1", "pathway_2", "pathway_3", "pathway_4"]


def _pathway_cols(df: pd.DataFrame) -> list[str]:
    """Identify the non-reference pathway dummies available in df."""
    return sorted(c for c in df.columns if c.startswith("pathway_") and c != "pathway_0")


def _sigma_v_T_window(start: int, end: int) -> pd.DataFrame:
    """Compute country-level std of annual temperature over arbitrary window."""
    df = pd.read_parquet(CLIMATE_RAW)
    sub = df[(df["year"] >= start) & (df["year"] <= end)]
    out = sub.groupby("iso3")["t_c"].std().rename("sigma_v_T_window").reset_index()
    return out


def _ols_tstat(
    df: pd.DataFrame,
    y_col: str,
    substrate: str,
    pathway_dummies: list[str],
    controls: list[str],
) -> float:
    """Return the t-statistic on `substrate` in the mediation regression (with pathways)."""
    needed = [y_col, substrate] + pathway_dummies + controls
    d = df.dropna(subset=needed)
    if len(d) < 10:
        return np.nan
    X = sm.add_constant(d[[substrate] + controls + pathway_dummies])
    try:
        res = sm.OLS(d[y_col], X).fit()
        return float(res.tvalues[substrate])
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Check 1: Continent FE
# ---------------------------------------------------------------------------
def check_continent_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run Exercise 2 mediation with 6 continent dummies added to controls.

    Drops one continent dummy (continent_OC) as the reference to avoid
    perfect collinearity with the constant.
    """
    t0 = time.time()
    df = add_continent_dummies(df)
    continent_cols = [c for c in df.columns if c.startswith("continent_")]
    # Drop one reference category to avoid perfect collinearity
    continent_cols = [c for c in continent_cols if c != "continent_OC"]
    pw_cols = _pathway_cols(df)

    rows = []
    total = len(OUTCOMES) * len(SUBSTRATES)
    done = 0
    for outcome in OUTCOMES:
        for substrate in SUBSTRATES:
            done += 1
            seed = abs(hash(("continent_fe", outcome, substrate))) % (2**31)
            r = mediation_share_with_ci(
                df,
                y_col=outcome,
                substrate=substrate,
                pathway_dummies=pw_cols,
                controls=CONTROLS + continent_cols,
                n_boot=500,
                seed=seed,
            )
            rows.append({
                "check": "continent_fe",
                "outcome": outcome,
                "substrate": substrate,
                **r,
            })
            print(
                f"  [{done:02d}/{total}] continent_fe | {outcome[:22]:22s} ~ {substrate[:22]:22s}"
                f"  ms={r['mediation_share']:+.3f}  CI=[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}]"
            )
    print(f"  check_continent_fe done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 2: Leave-one-out
# ---------------------------------------------------------------------------
def check_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """For each ISO3, drop it and re-compute mediation share (point estimate only).

    Returns median + IQR across the N drops for each (outcome, substrate) cell.
    """
    t0 = time.time()
    pw_cols = _pathway_cols(df)
    iso_list = sorted(df["iso3"].dropna().unique())
    n_iso = len(iso_list)

    # Collect per-drop point estimates
    cell_shares: dict[tuple, list[float]] = {
        (out, sub): [] for out in OUTCOMES for sub in SUBSTRATES
    }

    for i, drop_iso in enumerate(iso_list):
        sub_df = df[df["iso3"] != drop_iso]
        if (i + 1) % 25 == 0:
            print(f"  LOO: drop {i+1}/{n_iso} ({drop_iso}) ...")
        for outcome in OUTCOMES:
            for substrate in SUBSTRATES:
                r = mediation_share_with_ci(
                    sub_df,
                    y_col=outcome,
                    substrate=substrate,
                    pathway_dummies=pw_cols,
                    controls=CONTROLS,
                    n_boot=1,  # minimum; only point estimate needed in inner loop
                    seed=0,
                )
                cell_shares[(outcome, substrate)].append(r["mediation_share"])

    # Summarise across drops
    rows = []
    for outcome in OUTCOMES:
        for substrate in SUBSTRATES:
            arr = np.array(cell_shares[(outcome, substrate)])
            arr = arr[~np.isnan(arr)]
            rows.append({
                "check": "leave_one_out",
                "outcome": outcome,
                "substrate": substrate,
                "mediation_share": float(np.nanmedian(arr)),  # median across drops
                "loo_median": float(np.nanmedian(arr)),
                "loo_q25": float(np.nanquantile(arr, 0.25)),
                "loo_q75": float(np.nanquantile(arr, 0.75)),
                "loo_iqr": float(np.nanquantile(arr, 0.75) - np.nanquantile(arr, 0.25)),
                "n_obs": n_iso,
            })
    print(f"  check_leave_one_out done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 3: Westfall-Young multi-testing correction
# ---------------------------------------------------------------------------
def check_wy_correction(df: pd.DataFrame, n_perm: int = 1000) -> pd.DataFrame:
    """Westfall-Young FWER-corrected p-values for the 16-cell mediation t-stat matrix.

    Algorithm:
      1. Compute observed |t-stat| for substrate in the mediation regression (with pathways)
         for all 16 cells.
      2. For each permutation, shuffle the outcome vector within the regression frame,
         recompute all 16 |t-stats|, track the maximum.
      3. Adjusted p-value for cell k = P(max_permutation |t| >= |t_obs_k|).
    """
    t0 = time.time()
    pw_cols = _pathway_cols(df)

    # Step 1: observed |t-stats|
    cells = [(out, sub) for out in OUTCOMES for sub in SUBSTRATES]
    obs_t = {}
    for outcome, substrate in cells:
        obs_t[(outcome, substrate)] = abs(
            _ols_tstat(df, outcome, substrate, pw_cols, CONTROLS)
        )

    # Step 2: permutation null distribution
    rng = np.random.default_rng(46)
    max_t_perm = np.zeros(n_perm)

    # Pre-filter data per cell (common missing mask)
    cell_dfs: dict[tuple, pd.DataFrame] = {}
    for outcome, substrate in cells:
        needed = [outcome, substrate] + pw_cols + CONTROLS
        cell_dfs[(outcome, substrate)] = df.dropna(subset=needed).copy()

    print(f"  WY: running {n_perm} permutations ...")
    for p in range(n_perm):
        perm_t_vals = []
        for outcome, substrate in cells:
            d = cell_dfs[(outcome, substrate)].copy()
            d[outcome] = rng.permutation(d[outcome].values)
            t = abs(_ols_tstat(d, outcome, substrate, pw_cols, CONTROLS))
            perm_t_vals.append(t if not np.isnan(t) else 0.0)
        max_t_perm[p] = max(perm_t_vals)
        if (p + 1) % 200 == 0:
            print(f"  WY: {p+1}/{n_perm} permutations done ...")

    # Step 3: adjusted p-values
    rows = []
    for outcome, substrate in cells:
        t_obs = obs_t[(outcome, substrate)]
        p_adj = float(np.mean(max_t_perm >= t_obs))
        rows.append({
            "check": "wy_correction",
            "outcome": outcome,
            "substrate": substrate,
            "t_obs": t_obs,
            "p_adj_wy": p_adj,
            # Store t_obs as mediation_share proxy for the figure
            "mediation_share": t_obs,
            "n_obs": len(cell_dfs[(outcome, substrate)]),
        })
        print(
            f"  WY: {outcome[:22]:22s} ~ {substrate[:22]:22s}"
            f"  |t|={t_obs:.3f}  p_wy={p_adj:.3f}"
        )
    print(f"  check_wy_correction done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 4+5: Climate-window placebos (Exercise 1 Shapley)
# ---------------------------------------------------------------------------
def check_climate_placebo(
    df: pd.DataFrame, start: int, end: int, tag: str
) -> pd.DataFrame:
    """Re-run Exercise 1 Shapley with sigma_v_T computed over an alternative window.

    The baseline sigma_v_T_pre1750 column is replaced by the alternative.
    """
    t0 = time.time()
    sigma_alt = _sigma_v_T_window(start, end)

    # How many countries have data in this window?
    n_alt = sigma_alt["sigma_v_T_window"].notna().sum()
    print(f"  {tag}: sigma_v_T computed over {start}-{end}, {n_alt} countries have data")

    df2 = df.drop(columns=["sigma_v_T_pre1750"]).merge(sigma_alt, on="iso3", how="left")
    df2 = df2.rename(columns={"sigma_v_T_window": "sigma_v_T_pre1750"})

    rows = []
    for outcome in OUTCOMES:
        result = shapley_r2_decomposition(
            df2,
            y_col=outcome,
            substrates=SUBSTRATES,
            controls=CONTROLS,
        )
        for s in SUBSTRATES:
            rows.append({
                "check": tag,
                "outcome": outcome,
                "substrate": s,
                "shapley_r2": result["shapley"][s],
                "mediation_share": result["shapley"][s],  # alias for figure
                "full_model_r2": result["full_model_r2"],
                "baseline_r2": result["baseline_r2"],
                "n_obs": result["n_obs"],
            })
        print(
            f"  {tag}: {outcome[:22]:22s}  "
            f"shapley_sigma={result['shapley']['sigma_v_T_pre1750']:.4f}"
        )
    print(f"  {tag} done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 6: Outcome-period heterogeneity (pre-1900 only)
# ---------------------------------------------------------------------------
def check_pre1900_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run Shapley + mediation using only dt_timing_year on countries where
    dt_timing_year < 1950 (early-transition cohort, n~44).

    The dt_timing_year outcome spans 1809-2023; the pre-1950 sub-sample captures
    European / settled-offshoots early-transition countries.  Only 6 countries
    have dt_timing_year < 1900 so we use the broader <1950 threshold.

    This tests whether the Shapley pattern is stable when restricting to
    outcomes that are plausibly determined before modern-era interventions.
    """
    t0 = time.time()
    outcome = "dt_timing_year"
    pre1900_df = df[df[outcome] < 1950].copy()
    n = pre1900_df[outcome].notna().sum()
    print(f"  pre1900_outcome: {n} countries with dt_timing_year < 1950 (early-transition cohort)")

    pw_cols = _pathway_cols(df)

    # Shapley
    shapley_rows = []
    result = shapley_r2_decomposition(
        pre1900_df,
        y_col=outcome,
        substrates=SUBSTRATES,
        controls=CONTROLS,
    )
    for s in SUBSTRATES:
        shapley_rows.append({
            "check": "pre1900_outcome_shapley",
            "outcome": outcome,
            "substrate": s,
            "shapley_r2": result["shapley"][s],
            "mediation_share": result["shapley"][s],
            "full_model_r2": result["full_model_r2"],
            "n_obs": result["n_obs"],
        })
        print(
            f"  pre1900 shapley | {s[:28]:28s} -> shapley_r2={result['shapley'][s]:.4f}"
        )

    # Mediation
    mediation_rows = []
    for substrate in SUBSTRATES:
        seed = abs(hash(("pre1900_mediation", outcome, substrate))) % (2**31)
        r = mediation_share_with_ci(
            pre1900_df,
            y_col=outcome,
            substrate=substrate,
            pathway_dummies=pw_cols,
            controls=CONTROLS,
            n_boot=300,
            seed=seed,
        )
        mediation_rows.append({
            "check": "pre1900_outcome_mediation",
            "outcome": outcome,
            "substrate": substrate,
            **r,
        })
        print(
            f"  pre1900 mediation | {substrate[:28]:28s}  ms={r['mediation_share']:+.3f}"
            f"  CI=[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}]"
        )

    print(f"  check_pre1900_outcome done in {time.time()-t0:.0f}s")
    return pd.concat([pd.DataFrame(shapley_rows), pd.DataFrame(mediation_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Robustness battery: six checks")
    print("=" * 70)
    df = pd.read_parquet(PANEL)
    print(f"Loaded panel: {len(df)} countries, columns: {df.columns.tolist()}")

    all_results = []

    # Check 1: Continent FE
    print("\n[1/6] Continent fixed effects ...")
    all_results.append(check_continent_fe(df))

    # Check 2: Leave-one-out
    print("\n[2/6] Leave-one-out ...")
    all_results.append(check_leave_one_out(df))

    # Check 3: Westfall-Young
    print("\n[3/6] Westfall-Young FWER correction ...")
    all_results.append(check_wy_correction(df, n_perm=1000))

    # Check 4: Pre-1500 climate placebo
    print("\n[4/6] Climate placebo: pre-1500 window (1421-1500) ...")
    all_results.append(check_climate_placebo(df, 1421, 1500, "placebo_pre1500_window"))

    # Check 5: Modern-window climate placebo
    print("\n[5/6] Climate placebo: modern window (1950-2008) ...")
    all_results.append(check_climate_placebo(df, 1950, 2008, "placebo_modern_window"))

    # Check 6: Pre-1900 outcome
    print("\n[6/6] Pre-1900 outcome heterogeneity ...")
    all_results.append(check_pre1900_outcome(df))

    # Combine and write
    out_df = pd.concat(all_results, ignore_index=True)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out_df)} rows)")

    # Summary tables
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check 1: continent FE vs baseline mediation
    print("\n[1] Continent FE mediation shares:")
    base = pd.read_parquet(ROOT / "analysis/data/deep_determinants/exercise2_mediation_results.parquet")
    base_piv = base.pivot(index="substrate", columns="outcome", values="mediation_share")
    cfe = out_df[out_df["check"] == "continent_fe"]
    cfe_piv = cfe.pivot(index="substrate", columns="outcome", values="mediation_share")
    print("  Baseline:")
    print(base_piv.round(3))
    print("  With Continent FE:")
    print(cfe_piv.round(3))

    # Check 2: LOO IQR
    print("\n[2] LOO median (IQR) for sigma_v_T_pre1750:")
    loo = out_df[out_df["check"] == "leave_one_out"]
    sigma_loo = loo[loo["substrate"] == "sigma_v_T_pre1750"][
        ["outcome", "loo_median", "loo_q25", "loo_q75"]
    ]
    print(sigma_loo.to_string(index=False))

    # Check 3: WY cells surviving FWER
    print("\n[3] Westfall-Young: cells with p_adj_wy < 0.05:")
    wy = out_df[out_df["check"] == "wy_correction"]
    survivors = wy[wy["p_adj_wy"] < 0.05][["outcome", "substrate", "t_obs", "p_adj_wy"]]
    if len(survivors) == 0:
        print("  No cells survive FWER correction at 0.05 level.")
    else:
        print(survivors.to_string(index=False))

    # Check 4: Pre-1500 placebo shapley for sigma_v_T
    print("\n[4] Pre-1500 window placebo Shapley R² for sigma_v_T:")
    p1500 = out_df[out_df["check"] == "placebo_pre1500_window"]
    p1500_sigma = p1500[p1500["substrate"] == "sigma_v_T_pre1750"][
        ["outcome", "shapley_r2"]
    ]
    print(p1500_sigma.to_string(index=False))

    # Check 5: Modern-window placebo shapley for sigma_v_T
    print("\n[5] Modern window (1950-2008) placebo Shapley R² for sigma_v_T:")
    pmod = out_df[out_df["check"] == "placebo_modern_window"]
    pmod_sigma = pmod[pmod["substrate"] == "sigma_v_T_pre1750"][
        ["outcome", "shapley_r2"]
    ]
    print(pmod_sigma.to_string(index=False))

    # Check 6: Pre-1950 early-transition outcome
    print("\n[6] Early-transition outcome (dt_timing_year < 1950) Shapley R²:")
    p1900 = out_df[out_df["check"] == "pre1900_outcome_shapley"]
    print(p1900[["substrate", "shapley_r2"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
