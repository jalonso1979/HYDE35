"""Agricultural impact analysis with extended HYDE+ERA5 panel (1950-2025).

Tests whether temperature shocks affect agricultural and population outcomes
at the country level, using 76 countries with full ERA5 coverage.

Analyses:
1. Panel FE regressions: temp → cropland growth, pop growth, ag per capita
2. Local projections: IRFs of temp shocks on agricultural outcomes
3. Regime-dependent effects: pre/post intensification, hot/cold regions
4. Decade-by-decade warming impact evolution
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/BIGDATA/HYDE35")
sys.path.insert(0, str(ROOT))

PANEL_PATH = ROOT / "analysis" / "data" / "hyde_era5_extended_panel.parquet"
OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load panel ──────────────────────────────────────────────────────────────
print("=" * 70)
print("AGRICULTURAL IMPACT ANALYSIS (1950-2025)")
print("=" * 70)

df = pd.read_parquet(PANEL_PATH)

# Focus on countries with full ERA5 coverage (regions 1-8)
df = df[df["era5_region"].isin(range(1, 9))].copy()
df = df[df["temperature_c"].notna()].copy()

# Drop tiny countries (< 500 km²)
df = df[df["area_km2"] >= 500].copy()

# Drop zero-population rows
df = df[df["pop"] > 0].copy()

print(f"\nPanel: {len(df):,} obs, {df['country_id'].nunique()} countries, "
      f"{df['year'].min()}-{df['year'].max()}")
print(f"ERA5 regions: {sorted(df['era5_region'].unique())}")

# Cropland growth rate
df["cropland_growth"] = df.groupby("country_id")["log_cropland"].transform(
    lambda s: s.diff()
)
# Ag land growth
df["ag_land_growth"] = df.groupby("country_id")["log_ag_land"].transform(
    lambda s: s.diff()
)
# Ag per capita growth
df["ag_pc_growth"] = df.groupby("country_id")["log_ag_per_capita"].transform(
    lambda s: s.diff()
)
# Density growth
df["density_growth"] = df.groupby("country_id")["log_density"].transform(
    lambda s: s.diff()
)

# ════════════════════════════════════════════════════════════════════════════
# 1. FIXED-EFFECTS PANEL REGRESSIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. FIXED-EFFECTS PANEL REGRESSIONS")
print("=" * 70)


def run_fe_regression(panel, dep_var, indep_vars, entity_col="country_id"):
    """Within-transformation FE regression with clustered SEs."""
    cols = [dep_var] + indep_vars + [entity_col]
    sub = panel[cols].dropna()
    if len(sub) < 50:
        return None

    # Demean (within transformation)
    for col in [dep_var] + indep_vars:
        sub[col] = sub[col] - sub.groupby(entity_col)[col].transform("mean")

    y = sub[dep_var]
    X = sm.add_constant(sub[indep_vars])
    model = sm.OLS(y, X).fit(cov_type="cluster",
                              cov_kwds={"groups": sub[entity_col]})
    return model


specs = [
    ("pop_growth",      ["temp_anomaly", "precip_anomaly"],
     "Population Growth"),
    ("cropland_growth", ["temp_anomaly", "precip_anomaly"],
     "Cropland Growth"),
    ("ag_land_growth",  ["temp_anomaly", "precip_anomaly"],
     "Agricultural Land Growth"),
    ("ag_pc_growth",    ["temp_anomaly", "precip_anomaly"],
     "Ag Per Capita Growth"),
    ("pop_growth",      ["temp_anomaly", "precip_anomaly",
                          "temp_x_irrigation", "temp_x_crop_share"],
     "Pop Growth (with interactions)"),
]

fe_results = {}
for dep, indeps, label in specs:
    model = run_fe_regression(df, dep, indeps)
    if model is None:
        print(f"\n{label}: insufficient data")
        continue
    fe_results[label] = model
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"  N={model.nobs:.0f}, R²={model.rsquared:.4f}")
    print(f"{'─' * 50}")
    for var in indeps:
        if var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:25s}  β={coef:+.6f}  SE={se:.6f}  p={p:.4f} {star}")

# ════════════════════════════════════════════════════════════════════════════
# 2. LOCAL PROJECTION IRFs: TEMP SHOCK → AG OUTCOMES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. LOCAL PROJECTION IRFs")
print("=" * 70)

from analysis.paper3_climate.local_projections import run_local_projection

lp_specs = [
    ("temp_anomaly", "log_cropland",    "Temperature → Cropland"),
    ("temp_anomaly", "log_pop",         "Temperature → Population"),
    ("temp_anomaly", "log_ag_per_capita", "Temperature → Ag Per Capita"),
    ("temp_anomaly", "crop_share",      "Temperature → Crop Share"),
    ("precip_anomaly", "log_cropland",  "Precipitation → Cropland"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

lp_results = {}
for i, (shock, response, label) in enumerate(lp_specs):
    valid = df.dropna(subset=[shock, response])
    if len(valid) < 100:
        print(f"\n{label}: insufficient data ({len(valid)} obs)")
        continue

    irf = run_local_projection(
        valid, shock_var=shock, response_var=response,
        entity_col="country_id", max_horizon=10, n_lags=2,
    )
    lp_results[label] = irf

    print(f"\n{label} (N≈{len(valid)}):")
    sig_horizons = irf[irf["pvalue"] < 0.05]
    if len(sig_horizons) > 0:
        print(f"  Significant at horizons: {sig_horizons['horizon'].tolist()}")
        for _, row in sig_horizons.iterrows():
            print(f"    h={row['horizon']:.0f}: β={row['coefficient']:.6f} (p={row['pvalue']:.4f})")
    else:
        print("  No significant effects at any horizon")

    ax = axes[i]
    ax.plot(irf["horizon"], irf["coefficient"], marker="o", linewidth=2)
    ax.fill_between(irf["horizon"], irf["ci_lower"], irf["ci_upper"],
                     alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("β")

# Clear unused subplot
if len(lp_specs) < len(axes):
    for j in range(len(lp_specs), len(axes)):
        axes[j].set_visible(False)

plt.suptitle("Local Projection IRFs: Climate Shocks → Agricultural Outcomes\n"
             f"76 countries, 1950-2025", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "irf_ag_outcomes.png", dpi=150)
print(f"\nSaved: {OUT_DIR / 'irf_ag_outcomes.png'}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 3. REGIME-DEPENDENT: PRE/POST GREEN REVOLUTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. REGIME-DEPENDENT: PRE vs POST GREEN REVOLUTION (1970)")
print("=" * 70)

pre_gr = df[df["year"] < 1970].copy()
post_gr = df[df["year"] >= 1970].copy()
print(f"Pre-Green Revolution (1950-1969):  {len(pre_gr):,} obs, {pre_gr['country_id'].nunique()} countries")
print(f"Post-Green Revolution (1970-2025): {len(post_gr):,} obs, {post_gr['country_id'].nunique()} countries")

for period_label, period_df in [("Pre-GR (1950-1969)", pre_gr),
                                 ("Post-GR (1970-2025)", post_gr)]:
    model = run_fe_regression(period_df, "pop_growth",
                               ["temp_anomaly", "precip_anomaly"])
    if model is not None:
        print(f"\n  {period_label}: Pop Growth ~ Temperature Anomaly")
        print(f"    N={model.nobs:.0f}, R²={model.rsquared:.4f}")
        for var in ["temp_anomaly", "precip_anomaly"]:
            if var in model.params:
                p = model.pvalues[var]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {var:20s}  β={model.params[var]:+.6f}  p={p:.4f} {star}")

    model2 = run_fe_regression(period_df, "cropland_growth",
                                ["temp_anomaly", "precip_anomaly"])
    if model2 is not None:
        print(f"  {period_label}: Cropland Growth ~ Temperature Anomaly")
        print(f"    N={model2.nobs:.0f}, R²={model2.rsquared:.4f}")
        for var in ["temp_anomaly", "precip_anomaly"]:
            if var in model2.params:
                p = model2.pvalues[var]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {var:20s}  β={model2.params[var]:+.6f}  p={p:.4f} {star}")

# ════════════════════════════════════════════════════════════════════════════
# 4. ROLLING WINDOW: HOW CLIMATE SENSITIVITY EVOLVES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. ROLLING WINDOW: CLIMATE SENSITIVITY OVER TIME")
print("=" * 70)

window = 20  # years
step = 5
rolling_results = []

for start in range(1950, 2025 - window + 1, step):
    end = start + window
    sub = df[(df["year"] >= start) & (df["year"] < end)].copy()

    for dep_var, dep_label in [("pop_growth", "Pop Growth"),
                                ("cropland_growth", "Cropland Growth")]:
        model = run_fe_regression(sub, dep_var, ["temp_anomaly", "precip_anomaly"])
        if model is not None and "temp_anomaly" in model.params:
            rolling_results.append({
                "mid_year": start + window // 2,
                "outcome": dep_label,
                "temp_coef": model.params["temp_anomaly"],
                "temp_se": model.bse["temp_anomaly"],
                "temp_pval": model.pvalues["temp_anomaly"],
                "precip_coef": model.params.get("precip_anomaly", np.nan),
                "nobs": model.nobs,
            })

rolling_df = pd.DataFrame(rolling_results)

if len(rolling_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, outcome in enumerate(["Pop Growth", "Cropland Growth"]):
        sub = rolling_df[rolling_df["outcome"] == outcome]
        ax = axes[i]
        ax.plot(sub["mid_year"], sub["temp_coef"], marker="o", linewidth=2, color="#d7191c")
        ax.fill_between(sub["mid_year"],
                         sub["temp_coef"] - 1.96 * sub["temp_se"],
                         sub["temp_coef"] + 1.96 * sub["temp_se"],
                         alpha=0.2, color="#d7191c")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(1970, color="gray", linestyle=":", alpha=0.5, label="Green Revolution")
        ax.set_title(f"Temperature → {outcome}\n(20-year rolling windows)")
        ax.set_xlabel("Center year of window")
        ax.set_ylabel("β (temp anomaly)")
        ax.legend()

    plt.suptitle("Evolution of Climate Sensitivity, 1950-2025", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rolling_climate_sensitivity.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'rolling_climate_sensitivity.png'}")
    plt.close()

    print("\nRolling temperature coefficients:")
    for _, r in rolling_df.iterrows():
        star = "***" if r["temp_pval"] < 0.001 else "**" if r["temp_pval"] < 0.01 else "*" if r["temp_pval"] < 0.05 else ""
        print(f"  {r['mid_year']:.0f}  {r['outcome']:18s}  β={r['temp_coef']:+.6f}  p={r['temp_pval']:.4f} {star}  (N={r['nobs']:.0f})")

# ════════════════════════════════════════════════════════════════════════════
# 5. CROSS-SECTIONAL: WARMING vs AG STRUCTURE CHANGE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. CROSS-SECTION: WARMING vs STRUCTURAL CHANGE (1950-2025)")
print("=" * 70)

# Compute per-country changes between first and last decade
first_dec = df[df["year"].between(1950, 1960)].groupby("country_id").agg(
    temp_early=("temperature_c", "mean"),
    crop_share_early=("crop_share", "mean"),
    irrigation_early=("irrigation_share", "mean"),
    urban_early=("urban_share", "mean"),
    density_early=("density", "mean"),
    pop_early=("pop", "mean"),
    iso3=("iso3", "first"),
    name=("name", "first"),
    era5_region=("era5_region", "first"),
)

last_dec = df[df["year"].between(2015, 2025)].groupby("country_id").agg(
    temp_late=("temperature_c", "mean"),
    crop_share_late=("crop_share", "mean"),
    irrigation_late=("irrigation_share", "mean"),
    urban_late=("urban_share", "mean"),
    density_late=("density", "mean"),
    pop_late=("pop", "mean"),
)

cross = first_dec.join(last_dec, how="inner")
cross["warming"] = cross["temp_late"] - cross["temp_early"]
cross["d_crop_share"] = cross["crop_share_late"] - cross["crop_share_early"]
cross["d_irrigation"] = cross["irrigation_late"] - cross["irrigation_early"]
cross["d_urban"] = cross["urban_late"] - cross["urban_early"]
cross["d_density"] = cross["density_late"] - cross["density_early"]
cross["pop_growth_total"] = np.log(cross["pop_late"] / cross["pop_early"])

print(f"\nCross-sectional sample: {len(cross)} countries")
print(f"Mean warming: {cross['warming'].mean():.2f} °C")
print(f"Range: [{cross['warming'].min():.2f}, {cross['warming'].max():.2f}] °C")

if len(cross) > 10:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plots = [
        ("warming", "d_crop_share", "Warming (°C)", "Δ Crop Share"),
        ("warming", "d_irrigation", "Warming (°C)", "Δ Irrigation Share"),
        ("warming", "d_urban", "Warming (°C)", "Δ Urban Share"),
        ("warming", "pop_growth_total", "Warming (°C)", "Log Population Growth"),
    ]

    for ax, (x, y, xlabel, ylabel) in zip(axes.ravel(), plots):
        valid = cross[[x, y]].dropna()
        ax.scatter(valid[x], valid[y], alpha=0.6, s=30)

        # Add country labels for extremes
        if len(valid) > 5:
            for _, row in cross.nlargest(3, x).iterrows():
                if pd.notna(row[y]):
                    ax.annotate(row["iso3"], (row[x], row[y]), fontsize=7, alpha=0.7)
            for _, row in cross.nsmallest(3, x).iterrows():
                if pd.notna(row[y]):
                    ax.annotate(row["iso3"], (row[x], row[y]), fontsize=7, alpha=0.7)

        # OLS fit line
        X_ols = sm.add_constant(valid[x])
        m = sm.OLS(valid[y], X_ols).fit()
        x_range = np.linspace(valid[x].min(), valid[x].max(), 50)
        ax.plot(x_range, m.params[0] + m.params[1] * x_range,
                color="red", linewidth=1.5, alpha=0.7)

        p = m.pvalues.iloc[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.set_title(f"{ylabel} vs {xlabel}\n"
                     f"β={m.params.iloc[1]:.4f}, p={p:.3f}{star}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cross-Section: 1950s vs 2015-2025 (76 countries)", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cross_section_warming_structure.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'cross_section_warming_structure.png'}")
    plt.close()

    # Print OLS results
    print("\nCross-sectional OLS:")
    for x, y, xlabel, ylabel in plots:
        valid = cross[[x, y]].dropna()
        X_ols = sm.add_constant(valid[x])
        m = sm.OLS(valid[y], X_ols).fit()
        p = m.pvalues.iloc[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {ylabel:25s} ~ Warming: β={m.params.iloc[1]:+.5f}  p={p:.4f} {star}  N={len(valid)}")

# ════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: KEY COEFFICIENTS")
print("=" * 70)

print(f"\n{'Specification':<45} {'β_temp':>10} {'p-value':>10} {'N':>6}")
print("─" * 75)

for label, model in fe_results.items():
    if "temp_anomaly" in model.params:
        b = model.params["temp_anomaly"]
        p = model.pvalues["temp_anomaly"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"FE: {label:<41} {b:+10.6f} {p:10.4f}{star} {model.nobs:6.0f}")

for label, irf in lp_results.items():
    h0 = irf[irf["horizon"] == 0]
    if len(h0) > 0:
        b = h0.iloc[0]["coefficient"]
        p = h0.iloc[0]["pvalue"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"LP(h=0): {label:<38} {b:+10.6f} {p:10.4f}{star}")

    h5 = irf[irf["horizon"] == 5]
    if len(h5) > 0:
        b = h5.iloc[0]["coefficient"]
        p = h5.iloc[0]["pvalue"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"LP(h=5): {label:<38} {b:+10.6f} {p:10.4f}{star}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nFigures saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
