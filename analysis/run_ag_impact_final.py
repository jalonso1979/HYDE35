"""Final agricultural impact analysis: clean 1950-2025 panel.

Uses only annual HYDE × annual ERA5 data. Historical climate reconstructions
(PAGES 2k, Berkeley Earth, CRU TS) enter only as cross-sectional endowments.

Analyses:
1. Panel FE: climate → ag outcomes (clustered SEs)
2. Local projections: IRFs at country level
3. Regime-dependent: pre/post Green Revolution
4. Rolling windows: evolving climate sensitivity
5. Cross-section: climate endowments → structural change
6. Historical endowments: long-run climate as cross-sectional predictor
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
CLIMATE_HIST = ROOT / "analysis" / "data" / "climate_panel_0_2025.parquet"
OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from analysis.paper3_climate.local_projections import run_local_projection

# ── Load panel ──────────────────────────────────────────────────────────────
print("=" * 70)
print("AGRICULTURAL IMPACT ANALYSIS — CLEAN 1950-2025 PANEL")
print("=" * 70)

df = pd.read_parquet(PANEL_PATH)
df = df[df["era5_region"].isin(range(1, 9))].copy()
df = df[df["temperature_c"].notna() & (df["pop"] > 0) & (df["area_km2"] >= 500)].copy()

# Derived growth rates
for var, log_var in [("cropland", "log_cropland"), ("ag_land_km2", "log_ag_land"),
                     ("ag_per_capita", "log_ag_per_capita"), ("density", "log_density")]:
    growth_col = f"{var}_growth" if var != "density" else "density_growth"
    df[growth_col] = df.groupby("country_id")[log_var].transform(lambda s: s.diff())

n_countries = df["country_id"].nunique()
print(f"\nPanel: {len(df):,} obs, {n_countries} countries, {df['year'].min()}-{df['year'].max()}")


def run_fe(panel, dep, indeps, entity="country_id"):
    cols = [dep] + indeps + [entity]
    sub = panel[cols].dropna()
    if len(sub) < 50:
        return None
    for c in [dep] + indeps:
        sub[c] = sub[c] - sub.groupby(entity)[c].transform("mean")
    y = sub[dep]
    X = sm.add_constant(sub[indeps])
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub[entity]})


def print_fe(model, label, indeps):
    print(f"\n{'─' * 55}")
    print(f"  {label}  (N={model.nobs:.0f}, R²={model.rsquared:.4f})")
    print(f"{'─' * 55}")
    for v in indeps:
        if v in model.params.index:
            b, se, p = model.params[v], model.bse[v], model.pvalues[v]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {v:30s} {b:+10.6f} ({se:.6f}) {star}")


# ════════════════════════════════════════════════════════════════════════════
# 1. PANEL FE REGRESSIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. FIXED-EFFECTS PANEL REGRESSIONS")
print("=" * 70)

specs = [
    ("pop_growth",       ["temp_anomaly", "precip_anomaly"],
     "Pop Growth ~ Climate"),
    ("cropland_growth",  ["temp_anomaly", "precip_anomaly"],
     "Cropland Growth ~ Climate"),
    ("ag_land_km2_growth", ["temp_anomaly", "precip_anomaly"],
     "Ag Land Growth ~ Climate"),
    ("pop_growth",       ["temp_anomaly", "precip_anomaly",
                          "temp_x_crop_share", "temp_x_irrigation"],
     "Pop Growth ~ Climate + Interactions"),
    ("cropland_growth",  ["temp_anomaly", "precip_anomaly",
                          "temp_x_crop_share", "temp_x_irrigation"],
     "Cropland Growth ~ Climate + Interactions"),
]

for dep, indeps, label in specs:
    m = run_fe(df, dep, indeps)
    if m:
        print_fe(m, label, indeps)

# ════════════════════════════════════════════════════════════════════════════
# 2. LOCAL PROJECTION IRFs
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. LOCAL PROJECTION IRFs")
print("=" * 70)

lp_specs = [
    ("temp_anomaly",   "log_cropland",      "Temp → Cropland"),
    ("temp_anomaly",   "log_pop",           "Temp → Population"),
    ("temp_anomaly",   "log_ag_per_capita", "Temp → Ag Per Capita"),
    ("temp_anomaly",   "crop_share",        "Temp → Crop Share"),
    ("precip_anomaly", "log_cropland",      "Precip → Cropland"),
    ("precip_anomaly", "log_pop",           "Precip → Population"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = {"Temp": "#d7191c", "Precip": "#2c7bb6"}

lp_results = {}
for i, (shock, response, label) in enumerate(lp_specs):
    valid = df.dropna(subset=[shock, response])
    irf = run_local_projection(valid, shock_var=shock, response_var=response,
                                entity_col="country_id", max_horizon=10, n_lags=2)
    lp_results[label] = irf

    sig = irf[irf["pvalue"] < 0.05]
    print(f"\n{label} (N≈{len(valid)}):")
    if len(sig) > 0:
        print(f"  Significant at h={sig['horizon'].tolist()}")
        for _, r in sig.iterrows():
            print(f"    h={r['horizon']:.0f}: β={r['coefficient']:+.6f} (p={r['pvalue']:.4f})")
    else:
        print("  No significant effects")

    ax = axes.ravel()[i]
    c = colors["Temp"] if "Temp" in label else colors["Precip"]
    ax.plot(irf["horizon"], irf["coefficient"], marker="o", linewidth=2, color=c)
    ax.fill_between(irf["horizon"], irf["ci_lower"], irf["ci_upper"], alpha=0.15, color=c)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("β")
    ax.grid(True, alpha=0.2)

plt.suptitle(f"Local Projection IRFs: Climate → Ag Outcomes ({n_countries} countries, 1950-2025)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "irf_ag_final.png", dpi=150)
print(f"\nSaved: {OUT_DIR / 'irf_ag_final.png'}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 3. PRE vs POST GREEN REVOLUTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. REGIME-DEPENDENT: PRE vs POST GREEN REVOLUTION")
print("=" * 70)

for period, mask, in [("1950-1969", df["year"] < 1970),
                       ("1970-2025", df["year"] >= 1970)]:
    sub = df[mask]
    print(f"\n  {period}: {len(sub):,} obs, {sub['country_id'].nunique()} countries")
    for dep, label in [("pop_growth", "Pop Growth"), ("cropland_growth", "Cropland Growth")]:
        m = run_fe(sub, dep, ["temp_anomaly", "precip_anomaly"])
        if m:
            b_t = m.params.get("temp_anomaly", np.nan)
            p_t = m.pvalues.get("temp_anomaly", np.nan)
            b_p = m.params.get("precip_anomaly", np.nan)
            p_p = m.pvalues.get("precip_anomaly", np.nan)
            s_t = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else ""
            s_p = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else ""
            print(f"    {label:20s}  temp: {b_t:+.6f}{s_t:4s}  precip: {b_p:+.6f}{s_p:4s}  (N={m.nobs:.0f})")

# ════════════════════════════════════════════════════════════════════════════
# 4. ROLLING WINDOWS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. ROLLING CLIMATE SENSITIVITY (20-yr windows)")
print("=" * 70)

window, step = 20, 5
roll = []
for start in range(1950, 2025 - window + 1, step):
    sub = df[(df["year"] >= start) & (df["year"] < start + window)]
    for dep, lab in [("pop_growth", "Pop"), ("cropland_growth", "Cropland")]:
        m = run_fe(sub, dep, ["temp_anomaly", "precip_anomaly"])
        if m and "temp_anomaly" in m.params:
            roll.append({
                "mid": start + window // 2, "outcome": lab,
                "beta": m.params["temp_anomaly"],
                "se": m.bse["temp_anomaly"],
                "p": m.pvalues["temp_anomaly"],
                "n": m.nobs,
            })

roll_df = pd.DataFrame(roll)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, outcome in enumerate(["Pop", "Cropland"]):
    s = roll_df[roll_df["outcome"] == outcome]
    ax = axes[i]
    ax.plot(s["mid"], s["beta"], marker="o", linewidth=2, color="#d7191c")
    ax.fill_between(s["mid"], s["beta"] - 1.96 * s["se"],
                     s["beta"] + 1.96 * s["se"], alpha=0.2, color="#d7191c")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(1970, color="gray", linestyle=":", alpha=0.5, label="Green Revolution")
    ax.set_title(f"Temp → {outcome} Growth")
    ax.set_xlabel("Center year")
    ax.set_ylabel("β (temp anomaly)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle("Evolution of Climate Sensitivity (20-yr rolling FE)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "rolling_sensitivity_final.png", dpi=150)
print(f"Saved: {OUT_DIR / 'rolling_sensitivity_final.png'}")
plt.close()

for _, r in roll_df.iterrows():
    star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
    print(f"  {r['mid']:.0f}  {r['outcome']:8s}  β={r['beta']:+.6f}  p={r['p']:.4f}{star:4s} (N={r['n']:.0f})")

# ════════════════════════════════════════════════════════════════════════════
# 5. CROSS-SECTION: WARMING vs STRUCTURAL CHANGE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. CROSS-SECTION: WARMING → STRUCTURAL CHANGE")
print("=" * 70)

early = df[df["year"].between(1950, 1960)].groupby("country_id").agg(
    t0=("temperature_c", "mean"), cs0=("crop_share", "mean"),
    ir0=("irrigation_share", "mean"), ur0=("urban_share", "mean"),
    d0=("density", "mean"), p0=("pop", "mean"),
    iso3=("iso3", "first"), region=("era5_region", "first"),
)
late = df[df["year"].between(2015, 2025)].groupby("country_id").agg(
    t1=("temperature_c", "mean"), cs1=("crop_share", "mean"),
    ir1=("irrigation_share", "mean"), ur1=("urban_share", "mean"),
    d1=("density", "mean"), p1=("pop", "mean"),
)
xs = early.join(late, how="inner")
xs["warming"] = xs["t1"] - xs["t0"]
xs["d_crop"] = xs["cs1"] - xs["cs0"]
xs["d_irrig"] = xs["ir1"] - xs["ir0"]
xs["d_urban"] = xs["ur1"] - xs["ur0"]
xs["pop_gr"] = np.log(xs["p1"] / xs["p0"])

print(f"N={len(xs)} countries, mean warming={xs['warming'].mean():.2f}°C")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plots = [("warming", "d_crop", "Δ Crop Share"), ("warming", "d_irrig", "Δ Irrigation"),
         ("warming", "d_urban", "Δ Urban Share"), ("warming", "pop_gr", "Log Pop Growth")]

for ax, (x, y, ylabel) in zip(axes.ravel(), plots):
    v = xs[[x, y]].dropna()
    ax.scatter(v[x], v[y], alpha=0.5, s=25)
    X_ols = sm.add_constant(v[x])
    m = sm.OLS(v[y], X_ols).fit()
    xr_ = np.linspace(v[x].min(), v[x].max(), 50)
    ax.plot(xr_, m.params.iloc[0] + m.params.iloc[1] * xr_, color="red", linewidth=1.5)
    p = m.pvalues.iloc[1]
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.set_title(f"{ylabel} vs Warming\nβ={m.params.iloc[1]:.4f}, p={p:.3f}{star}")
    ax.set_xlabel("Warming (°C)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    print(f"  {ylabel:20s} ~ Warming: β={m.params.iloc[1]:+.5f}  p={p:.4f}{star}")

plt.suptitle(f"Cross-Section: 1950s → 2015-25 ({len(xs)} countries)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "cross_section_final.png", dpi=150)
print(f"Saved: {OUT_DIR / 'cross_section_final.png'}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 6. HISTORICAL CLIMATE ENDOWMENTS (cross-sectional only)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. HISTORICAL CLIMATE ENDOWMENTS AS CROSS-SECTIONAL PREDICTORS")
print("=" * 70)

hist = pd.read_parquet(CLIMATE_HIST)

# Compute long-run mean and volatility by region (1-1850, pre-industrial)
pre_ind = hist[(hist["year"] >= 1) & (hist["year"] <= 1850) & hist["temperature_c"].notna()]
endow = pre_ind.groupby("region").agg(
    temp_longrun_mean=("temperature_c", "mean"),
    temp_longrun_std=("temperature_c", "std"),
    temp_longrun_range=("temperature_c", lambda x: x.max() - x.min()),
).reset_index().rename(columns={"region": "era5_region"})

print(f"  Historical endowments computed for {len(endow)} regions (1-1850 CE)")
print(endow.to_string(index=False))

# Merge with cross-sectional data
xs2 = xs.reset_index().merge(endow, left_on="region", right_on="era5_region", how="left")
xs2 = xs2.dropna(subset=["temp_longrun_std"])

print(f"\n  Countries with historical climate endowment: {len(xs2)}")

if len(xs2) > 10:
    # Does historical climate volatility predict modern structural outcomes?
    print("\n  Historical temperature volatility → Modern outcomes:")
    for dep, label in [("d_crop", "Δ Crop Share"), ("d_irrig", "Δ Irrigation"),
                        ("d_urban", "Δ Urban"), ("pop_gr", "Log Pop Growth")]:
        v = xs2[["temp_longrun_std", dep]].dropna()
        if len(v) < 10:
            continue
        X_ols = sm.add_constant(v["temp_longrun_std"])
        m = sm.OLS(v[dep], X_ols).fit()
        p = m.pvalues.iloc[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {label:20s} ~ Hist Vol: β={m.params.iloc[1]:+.5f}  p={p:.4f}{star}  (N={len(v)})")

    # Does long-run temperature level predict modern agricultural structure?
    print("\n  Historical temperature level → Modern outcomes:")
    for dep, label in [("cs0", "Crop Share 1950s"), ("ir0", "Irrigation 1950s"),
                        ("ur0", "Urban Share 1950s"), ("d_crop", "Δ Crop Share")]:
        v = xs2[["temp_longrun_mean", dep]].dropna()
        if len(v) < 10:
            continue
        X_ols = sm.add_constant(v["temp_longrun_mean"])
        m = sm.OLS(v[dep], X_ols).fit()
        p = m.pvalues.iloc[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {label:20s} ~ Hist Temp: β={m.params.iloc[1]:+.5f}  p={p:.4f}{star}  (N={len(v)})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (x, y, xl, yl) in zip(axes, [
        ("temp_longrun_std", "d_crop", "Historical Temp Volatility (°C)", "Δ Crop Share"),
        ("temp_longrun_mean", "cs0", "Historical Mean Temp (°C)", "Initial Crop Share (1950s)"),
        ("temp_longrun_std", "pop_gr", "Historical Temp Volatility (°C)", "Log Pop Growth"),
    ]):
        v = xs2[[x, y]].dropna()
        ax.scatter(v[x], v[y], alpha=0.5, s=25)
        X_ols = sm.add_constant(v[x])
        m = sm.OLS(v[y], X_ols).fit()
        xr_ = np.linspace(v[x].min(), v[x].max(), 50)
        ax.plot(xr_, m.params.iloc[0] + m.params.iloc[1] * xr_, color="red", linewidth=1.5)
        p = m.pvalues.iloc[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.set_title(f"{yl} vs {xl}\nβ={m.params.iloc[1]:.4f}, p={p:.3f}{star}")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Historical Climate Endowments → Modern Agricultural Structure",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "historical_endowments.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'historical_endowments.png'}")
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Specification':<50} {'β_temp':>10} {'p':>8} {'sig':>4}")
print("─" * 76)

for dep, indeps, label in specs:
    m = run_fe(df, dep, indeps)
    if m and "temp_anomaly" in m.params:
        b = m.params["temp_anomaly"]
        p = m.pvalues["temp_anomaly"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"FE: {label:<46} {b:+10.6f} {p:8.4f} {star}")

for label, irf in lp_results.items():
    for h in [0, 5, 10]:
        row = irf[irf["horizon"] == h]
        if len(row) > 0:
            b = row.iloc[0]["coefficient"]
            p = row.iloc[0]["pvalue"]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"LP(h={h:2d}): {label:<43} {b:+10.6f} {p:8.4f} {star}")

print(f"\nAll figures in: {OUT_DIR}")
