# analysis/paper5_horserace/robustness_battery.py
"""Robustness battery: continent FE, leave-one-out, Westfall-Young,
climate-window placebos, pre-1900 outcome heterogeneity, climate-A partialling.

All checks run on the 4-substrate × 6-outcome matrix:
  - climate bundle (T̄, P̄, σᵥᵀ, σᵥᴾ)
  - H_pred_pwadj
  - ancestral_yield_log
  - pandemic_intensity_norm

Output: analysis/data/deep_determinants/robustness_battery.parquet
        (long-form: rows tagged with check_name)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper5_horserace.mediation import (
    mediation_share_with_ci,
    mediation_share_partial_r2_with_ci,
    stable_seed,
)
from analysis.paper5_horserace.shapley import shapley_r2_decomposition
from analysis.paper5_horserace.exercise1_shapley import (
    CLIMATE_BUNDLE,
    FUNCTIONAL_BUNDLE,
    OUTCOMES,
    CONTROLS,
)
from analysis.paper5_horserace.subsamples import add_continent_dummies

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
CLIMATE_RAW = ROOT / "analysis/data/country_climate_1421_2025.parquet"
OUT = ROOT / "analysis/data/deep_determinants/robustness_battery.parquet"

SCALAR_SUBSTRATES = [
    "neolithic_frac",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]

SUBSTRATES_FOR_SHAPLEY = [list(CLIMATE_BUNDLE), list(FUNCTIONAL_BUNDLE)] + SCALAR_SUBSTRATES
SUBSTRATE_KEYS = ["climate_bundle", "functional_alleles"] + SCALAR_SUBSTRATES

PATHWAY_COLS_DEFAULT = ["pathway_1", "pathway_2", "pathway_3", "pathway_4"]


def _pathway_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("pathway_") and c != "pathway_0")


def _substrate_cells():
    """Yield (substrate_key, substrate_cols, method) per cell."""
    yield ("climate_bundle", list(CLIMATE_BUNDLE), "partial_r2")
    yield ("functional_alleles", list(FUNCTIONAL_BUNDLE), "partial_r2")
    for s in SCALAR_SUBSTRATES:
        yield (s, [s], "beta_attenuation")


def _climate_bundle_window(start: int, end: int) -> pd.DataFrame:
    """Build a 4-element climate bundle (T̄, P̄, σᵥᵀ, σᵥᴾ) over an arbitrary window."""
    df = pd.read_parquet(CLIMATE_RAW)
    sub = df[(df["year"] >= start) & (df["year"] <= end)]
    agg = sub.groupby("iso3").agg(
        t_mean_window=("t_c", "mean"),
        p_mean_window=("p_mm", "mean"),
        sigma_v_T_window=("t_c", "std"),
        sigma_v_P_window=("p_mm", "std"),
    ).reset_index()
    return agg


def _joint_f_stat(
    df: pd.DataFrame,
    y_col: str,
    substrate_cols: list[str],
    pathway_dummies: list[str],
    controls: list[str],
    return_df: bool = False,
):
    """Joint F-statistic for the null β_substrate = 0 in the mediation regression.

    If return_df is True, returns (F, df_num, df_den) where df_num = number of
    restricted substrate columns and df_den = residual df of the regression.
    Otherwise returns the scalar F (back-compat).
    """
    needed = [y_col, *substrate_cols, *pathway_dummies, *controls]
    d = df.dropna(subset=needed)
    if len(d) < 10:
        return (np.nan, np.nan, np.nan) if return_df else np.nan
    X = sm.add_constant(d[substrate_cols + controls + pathway_dummies])
    try:
        res = sm.OLS(d[y_col], X).fit()
        # Joint F-test of substrate_cols (excluding constant, controls, pathways)
        # Build the R matrix manually to be safe across statsmodels versions
        param_names = list(res.params.index)
        R = np.zeros((len(substrate_cols), len(param_names)))
        for i, c in enumerate(substrate_cols):
            R[i, param_names.index(c)] = 1.0
        ft = res.f_test(R)
        f_val = float(np.asarray(ft.fvalue).flatten()[0])
        if return_df:
            return f_val, float(ft.df_num), float(ft.df_denom)
        return f_val
    except Exception:
        return (np.nan, np.nan, np.nan) if return_df else np.nan


# ---------------------------------------------------------------------------
# Check 1: Continent FE
# ---------------------------------------------------------------------------
def check_continent_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run Exercise 2 mediation with 6 continent dummies added to controls."""
    t0 = time.time()
    df = add_continent_dummies(df)
    continent_cols = [c for c in df.columns if c.startswith("continent_")]
    continent_cols = [c for c in continent_cols if c != "continent_OC"]
    pw_cols = _pathway_cols(df)

    rows = []
    total = len(OUTCOMES) * 4
    done = 0
    for outcome in OUTCOMES:
        for skey, scols, method in _substrate_cells():
            done += 1
            seed = stable_seed("continent_fe", outcome, skey)
            if method == "partial_r2":
                r = mediation_share_partial_r2_with_ci(
                    df, y_col=outcome, substrate=scols,
                    pathway_dummies=pw_cols,
                    controls=CONTROLS + continent_cols,
                    n_boot=500, seed=seed,
                )
            else:
                r = mediation_share_with_ci(
                    df, y_col=outcome, substrate=scols[0],
                    pathway_dummies=pw_cols,
                    controls=CONTROLS + continent_cols,
                    n_boot=500, seed=seed,
                )
            rows.append({
                "check": "continent_fe",
                "outcome": outcome,
                "substrate": skey,
                "method": method,
                **r,
            })
            print(f"  [{done:02d}/{total}] continent_fe | {outcome[:22]:22s} ~ {skey:22s}"
                  f"  ms={r['mediation_share']:+.3f}  CI=[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}]")
    print(f"  check_continent_fe done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 2: Leave-one-out
# ---------------------------------------------------------------------------
def check_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """LOO mediation share (point estimate only) for each (outcome, substrate)."""
    t0 = time.time()
    pw_cols = _pathway_cols(df)
    iso_list = sorted(df["iso3"].dropna().unique())
    n_iso = len(iso_list)

    cell_shares: dict[tuple, list[float]] = {
        (out, skey): [] for out in OUTCOMES for skey, _, _ in _substrate_cells()
    }

    for iso in iso_list:
        sub = df[df["iso3"] != iso].copy()
        for outcome in OUTCOMES:
            for skey, scols, method in _substrate_cells():
                if method == "partial_r2":
                    r = mediation_share_partial_r2_with_ci(
                        sub, y_col=outcome, substrate=scols,
                        pathway_dummies=pw_cols, controls=CONTROLS,
                        n_boot=1, seed=0,
                    )
                else:
                    r = mediation_share_with_ci(
                        sub, y_col=outcome, substrate=scols[0],
                        pathway_dummies=pw_cols, controls=CONTROLS,
                        n_boot=1, seed=0,
                    )
                cell_shares[(outcome, skey)].append(r["mediation_share"])

    rows = []
    for (out, skey), shares in cell_shares.items():
        arr = np.array(shares, dtype=float)
        rows.append({
            "check": "leave_one_out",
            "outcome": out,
            "substrate": skey,
            "loo_median": float(np.nanmedian(arr)),
            "mediation_share": float(np.nanmedian(arr)),  # alias for figure
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
def check_wy_correction(
    df: pd.DataFrame, n_perm: int = 1000, primary: str = "minp"
) -> pd.DataFrame:
    """Westfall-Young single-step FWER on the 30-cell mediation matrix.

    Statistic per cell: the joint F-stat for significance of the substrate
    columns (vector for the climate/functional coalitions; scalar t² for the
    single-column substrates). Cells have heterogeneous numerator df
    (scalars df_num=1, climate coalition df_num=4, functional coalition
    df_num=8), so raw F-values are NOT comparable across cells.

    We therefore implement single-step **minP** as the primary correction:
    each cell's F is converted to a p-value via its OWN F(df_num, df_den)
    reference, and per permutation we take the MIN p across all 30 cells under
    the complete (global) null. The adjusted p for an observed cell is the
    fraction of permutations whose min-p is <= that cell's observed p. This is
    df-fair, unlike maxT-over-raw-F which systematically under-tests the
    high-df coalitions.

    Both corrections are stored for reproducibility:
      - p_adj_wy_minp : single-step minP (df-fair)  [primary]
      - p_adj_wy_maxt : single-step maxT over raw F  [legacy, df-biased]
    The ``p_adj_wy`` column aliases whichever is selected by ``primary``
    ('minp' or 'maxt') so downstream survivor logic stays compatible.
    """
    t0 = time.time()
    pw_cols = _pathway_cols(df)

    # Build cell specs
    cells = []
    for outcome in OUTCOMES:
        for skey, scols, method in _substrate_cells():
            cells.append((outcome, skey, scols))

    from scipy.stats import f as f_dist

    def _f_to_p(f_val, df_num, df_den):
        if (
            f_val is None or np.isnan(f_val)
            or np.isnan(df_num) or np.isnan(df_den)
        ):
            return 1.0  # null result -> least significant
        return float(f_dist.sf(f_val, df_num, df_den))

    # Observed F-stats, df, and per-cell p-values (own-F reference)
    obs_F: dict[tuple, float] = {}
    obs_dfnum: dict[tuple, float] = {}
    obs_dfden: dict[tuple, float] = {}
    obs_p: dict[tuple, float] = {}
    for outcome, skey, scols in cells:
        f_val, dn, dd = _joint_f_stat(
            df, outcome, scols, pw_cols, CONTROLS, return_df=True
        )
        obs_F[(outcome, skey)] = f_val
        obs_dfnum[(outcome, skey)] = dn
        obs_dfden[(outcome, skey)] = dd
        obs_p[(outcome, skey)] = _f_to_p(f_val, dn, dd)

    # Pre-filter data per cell
    cell_dfs: dict[tuple, pd.DataFrame] = {}
    for outcome, skey, scols in cells:
        needed = [outcome] + scols + pw_cols + CONTROLS
        cell_dfs[(outcome, skey)] = df.dropna(subset=needed).copy()

    rng = np.random.default_rng(46)
    max_F_perm = np.zeros(n_perm)   # legacy maxT (raw F)
    min_p_perm = np.ones(n_perm)    # primary minP

    print(f"  WY: running {n_perm} permutations across {len(cells)} cells ...")
    for p in range(n_perm):
        perm_F = []
        perm_p = []
        for outcome, skey, scols in cells:
            d = cell_dfs[(outcome, skey)].copy()
            d[outcome] = rng.permutation(d[outcome].values)
            f, dn, dd = _joint_f_stat(
                d, outcome, scols, pw_cols, CONTROLS, return_df=True
            )
            perm_F.append(f if not np.isnan(f) else 0.0)
            perm_p.append(_f_to_p(f, dn, dd))
        max_F_perm[p] = max(perm_F)
        min_p_perm[p] = min(perm_p)
        if (p + 1) % 200 == 0:
            print(f"  WY: {p+1}/{n_perm} permutations done ...")

    rows = []
    for outcome, skey, _ in cells:
        f_obs = obs_F[(outcome, skey)]
        p_obs = obs_p[(outcome, skey)]
        # Single-step maxT over raw F (legacy, df-biased)
        p_adj_maxt = float(np.mean(max_F_perm >= f_obs))
        # Single-step minP over own-F p-values (df-fair, primary)
        p_adj_minp = float(np.mean(min_p_perm <= p_obs))
        p_adj_wy = p_adj_minp if primary == "minp" else p_adj_maxt
        rows.append({
            "check": "wy_correction",
            "outcome": outcome,
            "substrate": skey,
            "f_obs": f_obs,
            "df_num": obs_dfnum[(outcome, skey)],
            "df_den": obs_dfden[(outcome, skey)],
            "p_obs": p_obs,
            "p_adj_wy_minp": p_adj_minp,
            "p_adj_wy_maxt": p_adj_maxt,
            "p_adj_wy": p_adj_wy,
            "mediation_share": f_obs,  # alias for figure
            "n_obs": len(cell_dfs[(outcome, skey)]),
        })
        print(f"  WY: {outcome[:22]:22s} ~ {skey:22s}"
              f"  F={f_obs:7.3f}  df_num={obs_dfnum[(outcome, skey)]:.0f}"
              f"  p_obs={p_obs:.4f}  p_minP={p_adj_minp:.3f}  p_maxT={p_adj_maxt:.3f}")
    print(f"  check_wy_correction done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 4+5: Climate-window placebos (replace climate bundle)
# ---------------------------------------------------------------------------
def check_climate_placebo(
    df: pd.DataFrame, start: int, end: int, tag: str
) -> pd.DataFrame:
    """Re-run Exercise 1 Shapley with the climate bundle replaced by an alternative
    measurement window (start, end). The full 4-element bundle is recomputed."""
    t0 = time.time()
    alt = _climate_bundle_window(start, end)
    n_alt = alt["t_mean_window"].notna().sum()
    print(f"  {tag}: climate bundle from {start}-{end}, {n_alt} countries have data")

    df2 = df.drop(columns=list(CLIMATE_BUNDLE)).merge(alt, on="iso3", how="left")
    df2 = df2.rename(columns={
        "t_mean_window": "t_mean_pre1750",
        "p_mean_window": "p_mean_pre1750",
        "sigma_v_T_window": "sigma_v_T_pre1750",
        "sigma_v_P_window": "sigma_v_P_pre1750",
    })

    rows = []
    for outcome in OUTCOMES:
        result = shapley_r2_decomposition(
            df2,
            y_col=outcome,
            substrates=SUBSTRATES_FOR_SHAPLEY,
            controls=CONTROLS,
        )
        # Map raw keys to canonical
        key_map = {"+".join(CLIMATE_BUNDLE): "climate_bundle",
                   "+".join(FUNCTIONAL_BUNDLE): "functional_alleles",
                   "neolithic_frac": "neolithic_frac",
                   "ancestral_yield_log": "ancestral_yield_log",
                   "pandemic_intensity_norm": "pandemic_intensity_norm"}
        for raw_k, phi in result["shapley"].items():
            canonical = key_map[raw_k]
            rows.append({
                "check": tag,
                "outcome": outcome,
                "substrate": canonical,
                "shapley_r2": phi,
                "mediation_share": phi,
                "full_model_r2": result["full_model_r2"],
                "baseline_r2": result["baseline_r2"],
                "n_obs": result["n_obs"],
            })
        bundle_phi = next(
            phi for raw_k, phi in result["shapley"].items() if raw_k.startswith("t_mean")
        )
        print(f"  {tag}: {outcome[:22]:22s}  shapley_climate={bundle_phi:.4f}")
    print(f"  {tag} done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 6: Early-transition outcome (dt_timing_year < 1950)
# ---------------------------------------------------------------------------
def check_pre1900_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run Shapley + mediation using only dt_timing_year on early-transition cohort."""
    t0 = time.time()
    outcome = "dt_timing_year"
    pre1900_df = df[df[outcome] < 1950].copy()
    n = pre1900_df[outcome].notna().sum()
    print(f"  pre1900_outcome: {n} countries with dt_timing_year < 1950")

    pw_cols = _pathway_cols(df)

    shapley_rows = []
    result = shapley_r2_decomposition(
        pre1900_df,
        y_col=outcome,
        substrates=SUBSTRATES_FOR_SHAPLEY,
        controls=CONTROLS,
    )
    key_map = {"+".join(CLIMATE_BUNDLE): "climate_bundle",
               "+".join(FUNCTIONAL_BUNDLE): "functional_alleles",
               "neolithic_frac": "neolithic_frac",
               "ancestral_yield_log": "ancestral_yield_log",
               "pandemic_intensity_norm": "pandemic_intensity_norm"}
    for raw_k, phi in result["shapley"].items():
        canonical = key_map[raw_k]
        shapley_rows.append({
            "check": "pre1900_outcome_shapley",
            "outcome": outcome,
            "substrate": canonical,
            "shapley_r2": phi,
            "mediation_share": phi,
            "full_model_r2": result["full_model_r2"],
            "n_obs": result["n_obs"],
        })
        print(f"  pre1900 shapley | {canonical:28s} -> shapley_r2={phi:.4f}")

    mediation_rows = []
    for skey, scols, method in _substrate_cells():
        seed = stable_seed("pre1900_mediation", outcome, skey)
        if method == "partial_r2":
            r = mediation_share_partial_r2_with_ci(
                pre1900_df, y_col=outcome, substrate=scols,
                pathway_dummies=pw_cols, controls=CONTROLS,
                n_boot=300, seed=seed,
            )
        else:
            r = mediation_share_with_ci(
                pre1900_df, y_col=outcome, substrate=scols[0],
                pathway_dummies=pw_cols, controls=CONTROLS,
                n_boot=300, seed=seed,
            )
        mediation_rows.append({
            "check": "pre1900_outcome_mediation",
            "outcome": outcome,
            "substrate": skey,
            "method": method,
            **r,
        })
        print(f"  pre1900 mediation | {skey:28s}  ms={r['mediation_share']:+.3f}")

    print(f"  check_pre1900_outcome done in {time.time()-t0:.0f}s")
    return pd.concat([pd.DataFrame(shapley_rows), pd.DataFrame(mediation_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Check 7: Climate-A partialling — partial T̄, P̄ out of A before horserace
# ---------------------------------------------------------------------------
def check_climate_A_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Robustness: partial mean T and mean P out of ancestral_yield_log.

    Galor-Özak's ancestral crop yield is constructed from GAEZ projections
    under paleo-climate. To break the (T̄, P̄)-A correlation, we regress A on
    (T̄, P̄) and use the residual as a "climate-residualised" A in the Shapley.
    """
    t0 = time.time()
    work = df.dropna(subset=["ancestral_yield_log", "t_mean_pre1750", "p_mean_pre1750"]).copy()
    X = sm.add_constant(work[["t_mean_pre1750", "p_mean_pre1750"]])
    res = sm.OLS(work["ancestral_yield_log"], X).fit()
    work["ancestral_yield_log"] = res.resid + res.params["const"]  # preserve mean
    # Merge residualised values back
    df2 = df.copy()
    df2.loc[work.index, "ancestral_yield_log"] = work["ancestral_yield_log"]

    rows = []
    for outcome in OUTCOMES:
        result = shapley_r2_decomposition(
            df2,
            y_col=outcome,
            substrates=SUBSTRATES_FOR_SHAPLEY,
            controls=CONTROLS,
        )
        key_map = {"+".join(CLIMATE_BUNDLE): "climate_bundle",
                   "+".join(FUNCTIONAL_BUNDLE): "functional_alleles",
                   "neolithic_frac": "neolithic_frac",
                   "ancestral_yield_log": "ancestral_yield_log",
                   "pandemic_intensity_norm": "pandemic_intensity_norm"}
        for raw_k, phi in result["shapley"].items():
            canonical = key_map[raw_k]
            rows.append({
                "check": "climate_A_partial",
                "outcome": outcome,
                "substrate": canonical,
                "shapley_r2": phi,
                "mediation_share": phi,
                "full_model_r2": result["full_model_r2"],
                "baseline_r2": result["baseline_r2"],
                "n_obs": result["n_obs"],
            })
        bundle_phi = next(phi for raw_k, phi in result["shapley"].items() if raw_k.startswith("t_mean"))
        a_phi = result["shapley"]["ancestral_yield_log"]
        print(f"  climate_A_partial: {outcome[:22]:22s}  shapley_climate={bundle_phi:.4f}  shapley_A={a_phi:.4f}")
    print(f"  check_climate_A_partial done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Robustness battery: seven checks (4 substrates × 6 outcomes)")
    print("=" * 70)
    df = pd.read_parquet(PANEL)
    print(f"Loaded panel: {len(df)} countries, {len(df.columns)} columns")

    all_results = []

    print("\n[1/7] Continent fixed effects ...")
    all_results.append(check_continent_fe(df))

    print("\n[2/7] Leave-one-out ...")
    all_results.append(check_leave_one_out(df))

    print("\n[3/7] Westfall-Young FWER correction ...")
    all_results.append(check_wy_correction(df, n_perm=1000))

    print("\n[4/7] Climate placebo: pre-1500 window (1421-1500) ...")
    all_results.append(check_climate_placebo(df, 1421, 1500, "placebo_pre1500_window"))

    print("\n[5/7] Climate placebo: modern window (1950-2008) ...")
    all_results.append(check_climate_placebo(df, 1950, 2008, "placebo_modern_window"))

    print("\n[6/7] Pre-1900 outcome heterogeneity ...")
    all_results.append(check_pre1900_outcome(df))

    print("\n[7/7] Climate-A partialling robustness ...")
    all_results.append(check_climate_A_partial(df))

    out_df = pd.concat(all_results, ignore_index=True)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out_df)} rows)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n[3] Westfall-Young single-step minP FWER: cells with p_adj_wy < 0.05")
    wy = out_df[out_df["check"] == "wy_correction"]
    survivors = wy[wy["p_adj_wy"] < 0.05][
        ["outcome", "substrate", "f_obs", "df_num", "p_obs",
         "p_adj_wy_minp", "p_adj_wy_maxt"]
    ]
    if len(survivors) == 0:
        print("  No cells survive FWER correction at 0.05 level.")
    else:
        print(survivors.to_string(index=False))

    print("\n[4] Pre-1500 window placebo Shapley R² for climate_bundle:")
    p1500 = out_df[out_df["check"] == "placebo_pre1500_window"]
    p1500_climate = p1500[p1500["substrate"] == "climate_bundle"][["outcome", "shapley_r2"]]
    print(p1500_climate.to_string(index=False))

    print("\n[5] Modern window (1950-2008) placebo Shapley R² for climate_bundle:")
    pmod = out_df[out_df["check"] == "placebo_modern_window"]
    pmod_climate = pmod[pmod["substrate"] == "climate_bundle"][["outcome", "shapley_r2"]]
    print(pmod_climate.to_string(index=False))

    print("\n[7] Climate-A partialling — Shapley deltas:")
    cap = out_df[out_df["check"] == "climate_A_partial"]
    cap_p = cap.pivot(index="substrate", columns="outcome", values="shapley_r2")
    print(cap_p.round(4).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
