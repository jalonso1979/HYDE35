"""Run empirical analysis with updated ERA5 data.

Uses the extended ERA5 panel (1950-2025 for regions 1-8) to:
1. Build climate shock variables (anomalies, volatility)
2. Run local projection IRFs: temperature → agricultural outcomes
3. Test regime-dependent responses (pre/post intensification)
4. Run counterfactual: swap climate between regions
5. Print summary statistics and key results
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
sys.path.insert(0, str(ROOT))

from analysis.paper3_climate.climate_shocks import build_climate_shock_panel
from analysis.paper3_climate.local_projections import run_local_projection
from analysis.paper3_climate.regime_switching import (
    split_sample_by_regime,
    interaction_local_projection,
)
from analysis.paper3_climate.counterfactuals import run_counterfactual_experiment

OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load extended ERA5 panel ──────────────────────────────────────────────
print("=" * 70)
print("EMPIRICAL ANALYSIS WITH UPDATED ERA5 DATA")
print("=" * 70)

era5 = pd.read_parquet(ROOT / "analysis" / "data" / "era5_full_panel.parquet")
print(f"\nERA5 panel: {era5.shape}")
print(f"Regions: {sorted(era5['region'].unique())}")
print(f"Years: {era5['year'].min()} – {era5['year'].max()}")
print(f"Non-null temperature rows: {era5['temperature_c'].notna().sum()}")

# Focus on regions with extended coverage (1950-2025)
extended_regions = []
for r in era5["region"].unique():
    sub = era5[era5["region"] == r]
    if sub["year"].max() >= 2020 and sub["temperature_c"].notna().sum() >= 50:
        extended_regions.append(int(r))
print(f"\nRegions with extended coverage (through 2020+): {sorted(extended_regions)}")

# ── 2. Build climate shock panel ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("2. CLIMATE SHOCK CONSTRUCTION")
print("-" * 70)

# Use regions with extended data
era5_ext = era5[era5["region"].isin(extended_regions)].copy()
shock_panel = build_climate_shock_panel(
    era5_ext,
    climate_vars=["temperature_c", "precipitation_mm"],
    entity_col="region",
    rolling_window=30,
)

print(f"\nShock panel shape: {shock_panel.shape}")
shock_cols = [c for c in shock_panel.columns if "anomaly" in c or "volatility" in c]
print(f"Shock variables: {shock_cols}")

print("\nTemperature anomaly summary:")
print(shock_panel["temperature_c_anomaly"].describe().round(4).to_string())

print("\nPrecipitation anomaly summary:")
print(shock_panel["precipitation_mm_anomaly"].describe().round(6).to_string())

# ── 3. Local Projection IRFs ────────────────────────────────────────────────
print("\n" + "-" * 70)
print("3. LOCAL PROJECTION IRFs")
print("-" * 70)

# Temperature shock → precipitation response (as a proxy for agricultural impact)
# Using precipitation as response since we have both at region level
shock_var = "temperature_c_anomaly"
response_var = "precipitation_mm"

valid = shock_panel.dropna(subset=[shock_var, response_var])
print(f"\nValid observations for LP: {len(valid)}")

if len(valid) > 50:
    irf_temp_precip = run_local_projection(
        valid,
        shock_var=shock_var,
        response_var=response_var,
        entity_col="region",
        max_horizon=8,
        n_lags=2,
    )
    print("\nIRF: Temperature anomaly → Precipitation")
    print(irf_temp_precip.to_string(index=False))

    # Plot IRF
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(irf_temp_precip["horizon"], irf_temp_precip["coefficient"],
            marker="o", color="#2c7bb6", linewidth=2, label="IRF coefficient")
    ax.fill_between(
        irf_temp_precip["horizon"],
        irf_temp_precip["ci_lower"],
        irf_temp_precip["ci_upper"],
        alpha=0.2, color="#2c7bb6", label="95% CI",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Response coefficient")
    ax.set_title("Local Projection IRF: Temperature Shock → Precipitation Response\n"
                 f"(Regions {extended_regions}, 1950–2025)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "irf_temp_precip.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'irf_temp_precip.png'}")
    plt.close()
else:
    print("Insufficient observations for local projection.")
    irf_temp_precip = None

# ── 4. Temperature persistence (self-IRF) ───────────────────────────────────
print("\n" + "-" * 70)
print("4. TEMPERATURE PERSISTENCE (SELF-IRF)")
print("-" * 70)

if len(valid) > 50:
    irf_temp_self = run_local_projection(
        valid,
        shock_var="temperature_c_anomaly",
        response_var="temperature_c",
        entity_col="region",
        max_horizon=8,
        n_lags=2,
    )
    print("\nIRF: Temperature anomaly → Temperature level")
    print(irf_temp_self.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(irf_temp_self["horizon"], irf_temp_self["coefficient"],
            marker="s", color="#d7191c", linewidth=2, label="IRF coefficient")
    ax.fill_between(
        irf_temp_self["horizon"],
        irf_temp_self["ci_lower"],
        irf_temp_self["ci_upper"],
        alpha=0.2, color="#d7191c", label="95% CI",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Response coefficient")
    ax.set_title("Temperature Persistence: Shock → Level\n"
                 f"(Regions {extended_regions}, 1950–2025)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "irf_temp_persistence.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'irf_temp_persistence.png'}")
    plt.close()

# ── 5. Regime-dependent responses ────────────────────────────────────────────
print("\n" + "-" * 70)
print("5. REGIME-DEPENDENT RESPONSES")
print("-" * 70)

# Split by median temperature level (hot vs cold regions)
if len(valid) > 50:
    median_temp = valid["temperature_c"].median()
    print(f"\nMedian temperature: {median_temp:.2f} °C")

    hot, cold = split_sample_by_regime(valid, "temperature_c", median_temp)
    print(f"Hot regime (≥{median_temp:.1f}°C): {len(hot)} obs")
    print(f"Cold regime (<{median_temp:.1f}°C): {len(cold)} obs")

    if len(hot) > 30 and len(cold) > 30:
        irf_hot = run_local_projection(
            hot, shock_var="temperature_c_anomaly",
            response_var="precipitation_mm", entity_col="region",
            max_horizon=6, n_lags=2,
        )
        irf_cold = run_local_projection(
            cold, shock_var="temperature_c_anomaly",
            response_var="precipitation_mm", entity_col="region",
            max_horizon=6, n_lags=2,
        )

        print("\nIRF in HOT regime:")
        print(irf_hot.to_string(index=False))
        print("\nIRF in COLD regime:")
        print(irf_cold.to_string(index=False))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(irf_hot["horizon"], irf_hot["coefficient"],
                marker="o", color="#d7191c", linewidth=2, label="Hot regime")
        ax.fill_between(irf_hot["horizon"], irf_hot["ci_lower"],
                        irf_hot["ci_upper"], alpha=0.15, color="#d7191c")
        ax.plot(irf_cold["horizon"], irf_cold["coefficient"],
                marker="s", color="#2c7bb6", linewidth=2, label="Cold regime")
        ax.fill_between(irf_cold["horizon"], irf_cold["ci_lower"],
                        irf_cold["ci_upper"], alpha=0.15, color="#2c7bb6")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Horizon (years)")
        ax.set_ylabel("Response coefficient")
        ax.set_title(f"Regime-Dependent IRFs: Hot (≥{median_temp:.0f}°C) vs Cold (<{median_temp:.0f}°C)\n"
                     f"Temperature Shock → Precipitation")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUT_DIR / "irf_regime_hot_cold.png", dpi=150)
        print(f"\nSaved: {OUT_DIR / 'irf_regime_hot_cold.png'}")
        plt.close()

    # Interaction LP
    print("\nInteraction Local Projection (shock × temperature level):")
    interaction_df = interaction_local_projection(
        valid,
        shock_var="temperature_c_anomaly",
        response_var="precipitation_mm",
        regime_var="temperature_c",
        entity_col="region",
        max_horizon=6,
    )
    print(interaction_df.to_string(index=False))

# ── 6. Time series visualization ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("6. CLIMATE TRENDS (1950-2025)")
print("-" * 70)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for r in extended_regions:
    sub = era5_ext[era5_ext["region"] == r].sort_values("year")
    sub = sub.dropna(subset=["temperature_c"])
    axes[0].plot(sub["year"], sub["temperature_c"], alpha=0.6, label=f"R{r}")
    axes[1].plot(sub["year"], sub["precipitation_mm"], alpha=0.6, label=f"R{r}")

axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("ERA5 Regional Temperature, 1950–2025")
axes[0].legend(ncol=4, fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("Precipitation (mm/hr mean)")
axes[1].set_xlabel("Year")
axes[1].set_title("ERA5 Regional Precipitation, 1950–2025")
axes[1].legend(ncol=4, fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "era5_climate_trends_extended.png", dpi=150)
print(f"Saved: {OUT_DIR / 'era5_climate_trends_extended.png'}")
plt.close()

# ── 7. Warming trends ───────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("7. WARMING TREND ESTIMATES")
print("-" * 70)

import statsmodels.api as sm

for r in sorted(extended_regions):
    sub = era5_ext[(era5_ext["region"] == r) & era5_ext["temperature_c"].notna()].sort_values("year")
    if len(sub) < 20:
        continue
    X = sm.add_constant(sub["year"].values)
    y = sub["temperature_c"].values
    model = sm.OLS(y, X).fit()
    trend_per_decade = model.params[1] * 10
    p = model.pvalues[1]
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Region {r:2d}: {trend_per_decade:+.3f} °C/decade  (p={p:.4f}) {star}  "
          f"[{sub['year'].min()}-{sub['year'].max()}, n={len(sub)}]")

# ── 8. Counterfactual: swap hottest and coldest region climates ──────────────
print("\n" + "-" * 70)
print("8. COUNTERFACTUAL EXPERIMENT")
print("-" * 70)

if irf_temp_precip is not None and len(extended_regions) >= 2:
    # Find hottest and coldest regions
    region_means = era5_ext.groupby("region")["temperature_c"].mean()
    hottest = int(region_means.idxmax())
    coldest = int(region_means.idxmin())
    print(f"\nHottest region: {hottest} (mean={region_means[hottest]:.1f}°C)")
    print(f"Coldest region: {coldest} (mean={region_means[coldest]:.1f}°C)")

    # Need anomaly in shock panel
    cf_result = run_counterfactual_experiment(
        irf_df=irf_temp_precip,
        climate_panel=shock_panel,
        response_panel=shock_panel,
        source_entity=coldest,
        target_entity=hottest,
        shock_var="temperature_c_anomaly",
        response_var="precipitation_mm",
        entity_col="region",
    )

    if len(cf_result) > 0:
        print(f"\nCounterfactual: Region {hottest} with Region {coldest}'s climate shocks")
        print(f"  Years: {cf_result['year'].min()} – {cf_result['year'].max()}")
        print(f"  Mean actual precipitation:        {cf_result['actual'].mean():.6f}")
        print(f"  Mean counterfactual precipitation: {cf_result['counterfactual'].mean():.6f}")
        diff_pct = (cf_result["counterfactual"].mean() / cf_result["actual"].mean() - 1) * 100
        print(f"  Difference: {diff_pct:+.2f}%")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cf_result["year"], cf_result["actual"], color="#d7191c",
                linewidth=1.5, label=f"Actual (Region {hottest})")
        ax.plot(cf_result["year"], cf_result["counterfactual"], color="#2c7bb6",
                linewidth=1.5, linestyle="--",
                label=f"Counterfactual (with Region {coldest}'s shocks)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Precipitation (mm)")
        ax.set_title(f"Counterfactual: Region {hottest} under Region {coldest}'s Climate Shocks")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "counterfactual_climate_swap.png", dpi=150)
        print(f"\nSaved: {OUT_DIR / 'counterfactual_climate_swap.png'}")
        plt.close()

# ── 9. Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nFigures saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
