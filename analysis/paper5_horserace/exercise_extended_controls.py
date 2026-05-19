"""Extended controls robustness: cumulative conflict + volcanic exposure.

Steps
-----
1. Build country-level conflict controls from Brecke (reusing string-matching
   patterns from paper4_shadow/malthus_with_conflict_controls.py).
2. Build country-level cumulative VSSI exposure from eVolv2k (1421-1750),
   distance-weighted by latitudinal proximity with 30° Gaussian bandwidth.
3. Merge both controls into the master horserace panel.
4. Re-run Shapley decomposition with the extended control battery.
5. Re-run Westfall-Young FWER procedure with extended controls.
6. Print side-by-side comparison table.

Outputs
-------
  analysis/data/deep_determinants/conflict_controls.parquet
  analysis/data/deep_determinants/volcanic_controls.parquet
  analysis/data/deep_determinants_horserace.parquet  (updated in-place)
  analysis/data/deep_determinants/exercise1_shapley_extended.parquet
"""

from __future__ import annotations
import re
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
BRECKE = DATA / "brecke"
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
NEW_CONTROLS = [
    "cum_war_years_1400_1900",
    "cum_war_years_1900_1999",
    "cum_vssi_distance_weighted_1421_1750",
]
CONTROLS_EXTENDED = CONTROLS_BASE + NEW_CONTROLS


# ===========================================================================
# Step 1: conflict controls (Brecke)
# Reuses _country_keywords() and _load_brecke() logic from
# paper4_shadow/malthus_with_conflict_controls.py — patterns here are
# identical; do NOT rewrite the historical name list.
# ===========================================================================

def _country_keywords() -> dict[str, list[str]]:
    """iso3 -> keyword list for Brecke Name-string matching.
    Identical to malthus_with_conflict_controls._country_keywords().
    """
    iso_path = ROOT / "hyde35_country_iso_mapping.csv"
    iso = pd.read_csv(iso_path).dropna(subset=["iso3", "name"]).copy()

    # Hand-coded historical + adjective forms (taken verbatim from paper4_shadow)
    extras = {
        "GBR": ["England", "English", "Britain", "British", "Scotland", "Wales"],
        "FRA": ["France", "French", "Burgundy", "Burgundian", "Frankish"],
        "DEU": ["Germany", "German", "Prussia", "Bavaria", "Saxony", "Holy Roman"],
        "ITA": ["Italy", "Italian", "Naples", "Florence", "Venice", "Milan", "Tuscany", "Papal"],
        "ESP": ["Spain", "Spanish", "Castile", "Aragon", "Granada"],
        "PRT": ["Portugal", "Portuguese"],
        "NLD": ["Netherlands", "Dutch", "Holland", "United Provinces"],
        "BEL": ["Belgium", "Flanders", "Flemish", "Brabant"],
        "POL": ["Poland", "Polish", "Lithuania", "Lithuanian"],
        "RUS": ["Russia", "Russian", "Muscovy", "Soviet"],
        "TUR": ["Turkey", "Turkish", "Ottoman", "Ottomans"],
        "AUT": ["Austria", "Austrian", "Habsburg"],
        "HUN": ["Hungary", "Hungarian", "Magyar"],
        "CZE": ["Bohemia", "Czech", "Czechoslovakia"],
        "SWE": ["Sweden", "Swedish"],
        "NOR": ["Norway", "Norwegian"],
        "DNK": ["Denmark", "Danish"],
        "FIN": ["Finland", "Finnish"],
        "GRC": ["Greece", "Greek", "Byzantium", "Byzantine"],
        "EGY": ["Egypt", "Egyptian", "Mamluk", "Fatimid"],
        "IRN": ["Persia", "Persian", "Iran", "Iranian", "Sassanid", "Safavid"],
        "IRQ": ["Iraq", "Mesopotamia", "Babylon"],
        "SYR": ["Syria", "Syrian", "Levant"],
        "ISR": ["Israel", "Palestine", "Judea"],
        "SAU": ["Saudi", "Arabia", "Arab", "Hijaz"],
        "MAR": ["Morocco", "Moroccan", "Moors"],
        "TUN": ["Tunisia", "Tunisian", "Carthage"],
        "DZA": ["Algeria", "Algerian", "Algiers"],
        "LBY": ["Libya", "Libyan"],
        "ETH": ["Ethiopia", "Ethiopian", "Abyssinia"],
        "NGA": ["Nigeria", "Nigerian", "Sokoto"],
        "GHA": ["Ghana", "Ashanti"],
        "ZAF": ["South Africa", "Zulu", "Boer"],
        "CHN": ["China", "Chinese", "Qing", "Ming", "Yuan", "Mongol"],
        "JPN": ["Japan", "Japanese"],
        "KOR": ["Korea", "Korean", "Joseon"],
        "IND": ["India", "Indian", "Mughal", "Maratha", "Sikh"],
        "PAK": ["Pakistan", "Punjab", "Sindh"],
        "BGD": ["Bangladesh", "Bengal"],
        "VNM": ["Vietnam", "Vietnamese", "Annam"],
        "THA": ["Thailand", "Thai", "Siam"],
        "MMR": ["Burma", "Burmese", "Myanmar"],
        "IDN": ["Indonesia", "Java", "Sumatra", "Sumatran", "Dutch East"],
        "PHL": ["Philippines", "Filipino"],
        "USA": ["United States", "American", "US", "U.S."],
        "CAN": ["Canada", "Canadian"],
        "MEX": ["Mexico", "Mexican", "Aztec"],
        "BRA": ["Brazil", "Brazilian", "Portuguese Brazil"],
        "ARG": ["Argentina", "Argentine"],
        "CHL": ["Chile", "Chilean"],
        "PER": ["Peru", "Peruvian", "Inca"],
        "COL": ["Colombia", "New Granada"],
        "VEN": ["Venezuela"],
        "AUS": ["Australia", "Australian"],
        "NZL": ["New Zealand"],
    }
    out: dict[str, list[str]] = {}
    for _, r in iso.iterrows():
        kws = [r["name"]]
        if r["iso3"] in extras:
            kws = extras[r["iso3"]] + kws
        out[r["iso3"]] = list(dict.fromkeys(kws))
    return out


def _load_brecke() -> pd.DataFrame:
    """Load and concatenate both Brecke catalogues (1400+ and pre-1400)."""
    pre = pd.read_excel(BRECKE / "Brecke-Pre-1400-European-Conflicts.xlsx")
    pre = pre.rename(columns={"Conflict": "Name"})
    pre["TotalFatalities"] = pd.to_numeric(pre.get("Fatalities"), errors="coerce")
    post = pd.read_excel(BRECKE / "Conflict-Catalog-18-vars.xlsx")
    post["TotalFatalities"] = pd.to_numeric(post["TotalFatalities"], errors="coerce")
    keep = ["Name", "StartYear", "EndYear", "TotalFatalities"]
    df = pd.concat([pre[keep], post[keep]], ignore_index=True)
    df["StartYear"] = pd.to_numeric(df["StartYear"], errors="coerce").astype("Int64")
    df["EndYear"] = pd.to_numeric(df["EndYear"], errors="coerce").astype("Int64")
    df["EndYear"] = df["EndYear"].fillna(df["StartYear"])
    df = df.dropna(subset=["StartYear", "Name"])
    df["StartYear"] = df["StartYear"].astype(int)
    df["EndYear"] = df["EndYear"].astype(int)
    return df


def build_conflict_controls() -> pd.DataFrame:
    """Country-level cumulative war years 1400-1900 and 1900-1999.

    For each Brecke conflict, identify implicated countries via keyword
    matching on the Name actor string, expand into country-year war_active
    indicator, then sum over each period.
    """
    print("Building conflict controls from Brecke ...")
    brecke = _load_brecke()
    print(f"  Brecke: {len(brecke):,} conflicts, "
          f"{brecke['StartYear'].min()}-{brecke['EndYear'].max()}")

    kws = _country_keywords()
    rx = {iso: re.compile(
              r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
              flags=re.IGNORECASE)
          for iso, keys in kws.items() if keys}

    # Find implicated (iso3, year_start, year_end) triples
    # Only consider Brecke entries in 1400-1999
    YEAR_MIN, YEAR_MAX = 1400, 1999
    impl: dict[str, list[tuple[int, int]]] = {iso: [] for iso in rx}
    for _, c in brecke.iterrows():
        ys = int(c["StartYear"]); ye = int(c["EndYear"])
        if ye < YEAR_MIN or ys > YEAR_MAX:
            continue
        name = str(c["Name"])
        for iso, r in rx.items():
            if r.search(name):
                impl[iso].append((max(ys, YEAR_MIN), min(ye, YEAR_MAX)))

    years_1400_1900 = np.arange(1400, 1901)
    years_1900_1999 = np.arange(1900, 2000)

    rows = []
    for iso, ranges in impl.items():
        if not ranges:
            continue
        # Build binary war_active array 1400-1999
        active = np.zeros(YEAR_MAX - YEAR_MIN + 1, dtype=np.int8)
        for s, e in ranges:
            active[s - YEAR_MIN: e - YEAR_MIN + 1] = 1

        cum_1400_1900 = int(active[0: 1901 - YEAR_MIN].sum())
        cum_1900_1999 = int(active[1900 - YEAR_MIN: 2000 - YEAR_MIN].sum())

        if cum_1400_1900 > 0 or cum_1900_1999 > 0:
            rows.append({
                "iso3": iso,
                "cum_war_years_1400_1900": cum_1400_1900,
                "cum_war_years_1900_1999": cum_1900_1999,
            })

    df = pd.DataFrame(rows)
    # Fill zeros for countries with no match
    all_iso = list(rx.keys())
    df = (pd.DataFrame({"iso3": all_iso})
            .merge(df, on="iso3", how="left")
            .fillna({"cum_war_years_1400_1900": 0, "cum_war_years_1900_1999": 0}))
    df["cum_war_years_1400_1900"] = df["cum_war_years_1400_1900"].astype(int)
    df["cum_war_years_1900_1999"] = df["cum_war_years_1900_1999"].astype(int)
    df["source"] = "Brecke"

    out_path = DEEP_DET / "conflict_controls.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path} ({len(df)} rows)")
    print(f"  cum_war_years_1400_1900 mean={df['cum_war_years_1400_1900'].mean():.1f}  "
          f"max={df['cum_war_years_1400_1900'].max()}")
    print(f"  cum_war_years_1900_1999 mean={df['cum_war_years_1900_1999'].mean():.1f}  "
          f"max={df['cum_war_years_1900_1999'].max()}")
    return df


# ===========================================================================
# Step 2: volcanic controls (eVolv2k)
# ===========================================================================

def _parse_eVolv2k() -> pd.DataFrame:
    """Parse eVolv2k_sigl_toohey_2024.tab, skipping the PANGAEA metadata header.

    Returns a DataFrame with columns: year, month, day, lat, vssi
    (only rows where VSSI > 0 and year in 1421-1750).
    """
    path = DATA / "eVolv2k_sigl_toohey_2024.tab"
    rows = []
    in_data = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not in_data:
                if line.strip() == "*/":
                    in_data = True
                    next(f)  # skip the column-header line
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            try:
                year = int(float(parts[0]))
                lat = float(parts[4])
                vssi = float(parts[7])
            except (ValueError, IndexError):
                continue
            rows.append({"year": year, "lat": lat, "vssi": vssi})

    df = pd.DataFrame(rows)
    # Filter to 1421-1750 and non-trivial eruptions
    df = df[(df["year"] >= 1421) & (df["year"] <= 1750) & (df["vssi"] > 0)].copy()
    print(f"  eVolv2k: {len(df)} eruptions in 1421-1750 with VSSI>0  "
          f"(total VSSI={df['vssi'].sum():.1f} Tg)")
    return df


def build_volcanic_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Country-level cumulative distance-weighted VSSI exposure 1421-1750.

    exposure_i_t = VSSI_t * exp(-((lat_volcano_t - lat_i) / 30)**2 / 2)
    cum_vssi_distance_weighted_1421_1750 = sum over t of exposure_i_t

    Signed country centroid latitudes from hyde_era5_extended_panel.parquet.
    """
    print("Building volcanic controls from eVolv2k ...")
    # Signed latitude from hyde_era5_extended_panel
    era5 = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    cents = era5.groupby("iso3")["centroid_lat"].first().reset_index()
    cents = cents.rename(columns={"centroid_lat": "lat_signed"})
    print(f"  Signed centroids: {len(cents)} countries  "
          f"(lat range [{cents['lat_signed'].min():.1f}, {cents['lat_signed'].max():.1f}])")

    volc = _parse_eVolv2k()

    # Vectorised: countries × eruptions
    lat_i = cents["lat_signed"].values[:, np.newaxis]   # (N_countries, 1)
    lat_v = volc["lat"].values[np.newaxis, :]            # (1, N_eruptions)
    vssi_v = volc["vssi"].values[np.newaxis, :]          # (1, N_eruptions)

    # Gaussian distance weight with 30° bandwidth
    w = np.exp(-((lat_i - lat_v) / 30.0) ** 2 / 2.0)   # (N_countries, N_eruptions)
    exposure = (w * vssi_v).sum(axis=1)                  # (N_countries,)

    out = cents[["iso3"]].copy()
    out["cum_vssi_distance_weighted_1421_1750"] = exposure
    out["source"] = "eVolv2k_Sigl_Toohey_2024"

    out_path = DEEP_DET / "volcanic_controls.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved {out_path} ({len(out)} rows)")
    print(f"  cum_vssi mean={out['cum_vssi_distance_weighted_1421_1750'].mean():.2f}  "
          f"max={out['cum_vssi_distance_weighted_1421_1750'].max():.2f}")
    return out


# ===========================================================================
# Step 3: merge into master panel
# ===========================================================================

def merge_controls(conflict: pd.DataFrame, volcanic: pd.DataFrame) -> pd.DataFrame:
    """Append new control variables to the master horserace panel."""
    print("Merging controls into master panel ...")
    df = pd.read_parquet(PANEL_PATH)
    # Drop old versions if re-running
    for col in NEW_CONTROLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(
        conflict[["iso3", "cum_war_years_1400_1900", "cum_war_years_1900_1999"]],
        on="iso3", how="left")
    df = df.merge(
        volcanic[["iso3", "cum_vssi_distance_weighted_1421_1750"]],
        on="iso3", how="left")

    df.to_parquet(PANEL_PATH, index=False)
    n_new = df[NEW_CONTROLS].notna().all(axis=1).sum()
    print(f"  Panel: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Rows with all 3 new controls non-null: {n_new}")
    return df


# ===========================================================================
# Step 4a: Extended Shapley
# ===========================================================================

def run_extended_shapley(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run Shapley decomposition with extended control battery."""
    print("\nRunning extended Shapley decomposition ...")
    rows = []
    for outcome in OUTCOMES:
        print(f"  {outcome} ...")
        result = shapley_r2_decomposition(
            df,
            y_col=outcome,
            substrates=SUBSTRATES,
            controls=CONTROLS_EXTENDED,
        )
        for s in SUBSTRATES:
            rows.append({
                "outcome": outcome,
                "substrate": s,
                "shapley_r2": result["shapley"][s],
                "baseline_r2": result["baseline_r2"],
                "full_model_r2": result["full_model_r2"],
                "n_obs": result["n_obs"],
            })

    out = pd.DataFrame(rows)
    out_path = DEEP_DET / "exercise1_shapley_extended.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}")
    return out


# ===========================================================================
# Step 4b: Extended FWER (Westfall-Young)
# ===========================================================================

def _pathway_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("pathway_") and c != "pathway_0")


def _ols_tstat(df: pd.DataFrame, y_col: str, substrate: str,
               pathway_dummies: list[str], controls: list[str]) -> float:
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


def run_extended_wy(df: pd.DataFrame, n_perm: int = 1000) -> pd.DataFrame:
    """Westfall-Young FWER with extended controls, 1000 permutations."""
    print(f"\nRunning extended Westfall-Young FWER ({n_perm} permutations) ...")
    t0 = time.time()
    pw_cols = _pathway_cols(df)
    cells = [(out, sub) for out in OUTCOMES for sub in SUBSTRATES]

    # Step 1: observed |t-stats|
    obs_t = {}
    for outcome, substrate in cells:
        obs_t[(outcome, substrate)] = abs(
            _ols_tstat(df, outcome, substrate, pw_cols, CONTROLS_EXTENDED))

    # Pre-filter data per cell
    cell_dfs: dict[tuple, pd.DataFrame] = {}
    for outcome, substrate in cells:
        needed = [outcome, substrate] + pw_cols + CONTROLS_EXTENDED
        cell_dfs[(outcome, substrate)] = df.dropna(subset=needed).copy()

    # Step 2: permutation null
    rng = np.random.default_rng(46)
    max_t_perm = np.zeros(n_perm)
    for p in range(n_perm):
        perm_t_vals = []
        for outcome, substrate in cells:
            d = cell_dfs[(outcome, substrate)].copy()
            d[outcome] = rng.permutation(d[outcome].values)
            t = abs(_ols_tstat(d, outcome, substrate, pw_cols, CONTROLS_EXTENDED))
            perm_t_vals.append(t if not np.isnan(t) else 0.0)
        max_t_perm[p] = max(perm_t_vals)
        if (p + 1) % 200 == 0:
            print(f"  WY: {p+1}/{n_perm} permutations ...")

    # Step 3: adjusted p-values
    rows = []
    for outcome, substrate in cells:
        t_obs = obs_t[(outcome, substrate)]
        p_adj = float(np.mean(max_t_perm >= t_obs))
        rows.append({
            "outcome": outcome,
            "substrate": substrate,
            "t_obs": t_obs,
            "p_adj_wy": p_adj,
            "n_obs": len(cell_dfs[(outcome, substrate)]),
        })

    out = pd.DataFrame(rows)
    print(f"  WY done in {time.time()-t0:.0f}s")
    return out


# ===========================================================================
# Step 5: comparison table
# ===========================================================================

def print_comparison(shapley_base: pd.DataFrame, shapley_ext: pd.DataFrame,
                     wy_base: pd.DataFrame, wy_ext: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("COMPARISON: baseline vs extended controls")
    print("=" * 80)

    # Shapley R² pivot
    piv_b = shapley_base.pivot(index="substrate", columns="outcome", values="shapley_r2")
    piv_e = shapley_ext.pivot(index="substrate", columns="outcome", values="shapley_r2")
    delta = piv_e - piv_b
    print("\nShapley R² — baseline:")
    print(piv_b.round(4).to_string())
    print("\nShapley R² — extended controls (+conflict +volcanic):")
    print(piv_e.round(4).to_string())
    print("\nDelta (extended - baseline):")
    print(delta.round(4).to_string())

    # FWER comparison
    print("\nWestfall-Young FWER comparison (sorted by extended |t|):")
    wy_b_sub = wy_base[["outcome", "substrate", "t_obs", "p_adj_wy"]].rename(
        columns={"t_obs": "t_base", "p_adj_wy": "p_base"})
    wy_e_sub = wy_ext[["outcome", "substrate", "t_obs", "p_adj_wy"]].rename(
        columns={"t_obs": "t_ext", "p_adj_wy": "p_ext"})
    cmp = wy_b_sub.merge(wy_e_sub, on=["outcome", "substrate"])
    cmp["survive_base"] = cmp["p_base"] < 0.05
    cmp["survive_ext"] = cmp["p_ext"] < 0.05
    cmp = cmp.sort_values("t_ext", ascending=False)

    header = (f"{'outcome':30s} {'substrate':25s} "
              f"{'|t| base':>9} {'p_base':>8} {'survive_b':>10} "
              f"{'|t| ext':>9} {'p_ext':>8} {'survive_e':>10}")
    print(header)
    print("-" * len(header))
    for _, r in cmp.iterrows():
        surv_b = "YES ***" if r["survive_base"] else "no"
        surv_e = "YES ***" if r["survive_ext"] else "no"
        print(f"{r['outcome']:30s} {r['substrate']:25s} "
              f"{r['t_base']:9.3f} {r['p_base']:8.3f} {surv_b:>10s} "
              f"{r['t_ext']:9.3f} {r['p_ext']:8.3f} {surv_e:>10s}")

    # Focus on the 3 FWER survivors
    survivors_base = set()
    for _, r in wy_base.iterrows():
        if r["p_adj_wy"] < 0.05:
            survivors_base.add((r["outcome"], r["substrate"]))

    print(f"\n{'='*80}")
    print("THREE FWER-SURVIVING FINDINGS (baseline) — status with extended controls:")
    print(f"{'='*80}")
    for (out, sub) in sorted(survivors_base):
        row = wy_ext[(wy_ext["outcome"] == out) & (wy_ext["substrate"] == sub)].iloc[0]
        status = "SURVIVES" if row["p_adj_wy"] < 0.05 else "DROPS OUT"
        print(f"  {out} x {sub}")
        print(f"    Baseline:  |t|={wy_base[(wy_base['outcome']==out)&(wy_base['substrate']==sub)]['t_obs'].values[0]:.3f}  "
              f"p_WY={wy_base[(wy_base['outcome']==out)&(wy_base['substrate']==sub)]['p_adj_wy'].values[0]:.3f}")
        print(f"    Extended:  |t|={row['t_obs']:.3f}  p_WY={row['p_adj_wy']:.3f}  -> {status}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("Exercise: extended conflict + volcanic exposure controls")
    print("=" * 70)

    # Step 1
    conflict = build_conflict_controls()

    # Step 2
    panel_for_vol = pd.read_parquet(PANEL_PATH)
    volcanic = build_volcanic_controls(panel_for_vol)

    # Step 3
    df = merge_controls(conflict, volcanic)

    # Step 4a
    shapley_ext = run_extended_shapley(df)

    # Step 4b
    wy_ext = run_extended_wy(df, n_perm=1000)

    # Load baselines
    shapley_base = pd.read_parquet(
        DEEP_DET / "exercise1_shapley_results.parquet")
    wy_base = pd.read_parquet(
        DATA / "deep_determinants/robustness_battery.parquet")
    wy_base = wy_base[wy_base["check"] == "wy_correction"].copy()

    # Step 5
    print_comparison(shapley_base, shapley_ext, wy_base, wy_ext)

    print("\nDone.")


if __name__ == "__main__":
    main()
