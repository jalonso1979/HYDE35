"""Country-level channel-substitution exercise.

The pathway-level scatter of Section 4.5 has only four data points and
returns a substitution slope ($-0.15$, $p=0.80$) that is qualitatively
suggestive but not statistically distinguishable from zero.  We escalate
to the country-level analogue: for each country i, estimate

  φ^P_i = ∂(Δ_ann log P_i) / ∂(V_i)     (demographic margin)
  φ^L_i = ∂(Δ_ann log KK10_anthro_i) / ∂(V_i)   (Boserupian margin)

from the within-country time series of decadal observations 1700-1850
(post-1700 window where the KK10 signal is identified).  Each country
contributes ~15 decadal cells.  We use a reduced-form spec to keep df
comfortable:

  g_{ann,it} = α + β_pop ln P_{i,t-1} + β_T T̄_{it} + β_σ σ^T_{it}
               + φ V_{it} + ε

and extract (φ̂^P_i, σ̂^P_i) and (φ̂^L_i, σ̂^L_i) per country.

The substitution exercise then regresses φ^L_i on φ^P_i across the ~130
countries with both coefficients estimable.  Because both variables are
estimated with error, we run a Deming regression (orthogonal-distance
regression weighted by 1/σ^2) rather than OLS, plus an inverse-variance-
weighted OLS for comparison.  Slope and CI are computed by bootstrap.

Output:
    analysis/data/country_substitution.parquet
    analysis/figures/paper4_v2/figK_country_substitution.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.odr import ODR, RealData, Model

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG  = ROOT / "analysis" / "figures" / "paper4_v2"

CDL_OUTLIERS = ['FJI','REU','NCL','WSM','TON','JAM','MUS','VUT','DMA','GLP',
                'LCA','KNA','MTQ','VCT','HTI','SLV','LSO','SLE','CIV','GNB',
                'GNQ','CMR','BRN','SGP']

# Reduced-form per-country controls (kept light for df reasons)
PCC = ["log_pop", "t_bar_dev", "t_sd_dev"]


def _build_panel() -> pd.DataFrame:
    hyde = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    kk10 = pd.read_parquet(DATA / "kk10_country_panel.parquet")

    kk10 = kk10.sort_values(["iso3", "year"]).copy()
    kk10["log_kk10_anthro"] = np.log(kk10["kk10_anthro_km2"].clip(lower=1e-6))
    g = kk10.groupby("iso3")
    nxt = g["log_kk10_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - kk10["year"]
    kk10["g_kk10_anthro_ann"] = (nxt - kk10["log_kk10_anthro"]) / dt

    merged = hyde.merge(
        kk10[["iso3", "year", "g_kk10_anthro_ann"]],
        on=["iso3", "year"], how="left"
    )
    merged = merged[merged["year"] >= 1700].copy()
    return merged


def _country_coefs(d: pd.DataFrame, lhs: str) -> tuple[float, float, int] | None:
    """Return (β_VSSI, SE_VSSI, n_obs) for one country's regression."""
    d = d.dropna(subset=[lhs, "vssi_int"] + PCC).copy()
    if len(d) < len(PCC) + 3:
        return None
    X = sm.add_constant(d[PCC + ["vssi_int"]])
    try:
        res = sm.OLS(d[lhs], X).fit()
    except Exception:
        return None
    if "vssi_int" not in res.params.index:
        return None
    return (float(res.params["vssi_int"]),
            float(res.bse["vssi_int"]),
            int(res.nobs))


def main() -> None:
    panel = _build_panel()
    print(f"Post-1700 panel: {len(panel):,} rows, "
          f"{panel['iso3'].nunique()} countries\n")

    # Per-country coefficient extraction
    rows = []
    for iso3, g in panel.groupby("iso3"):
        cluster = g["cluster"].iloc[0]
        pathway = g["pathway"].iloc[0]
        if pd.isna(cluster):
            continue
        r_P = _country_coefs(g, "g_pop_ann")
        r_L = _country_coefs(g, "g_kk10_anthro_ann")
        if r_P is None or r_L is None:
            continue
        rows.append({
            "iso3": iso3, "cluster": int(cluster), "pathway": pathway,
            "phi_P": r_P[0], "phi_P_se": r_P[1], "n_P": r_P[2],
            "phi_L": r_L[0], "phi_L_se": r_L[1], "n_L": r_L[2],
        })
    cdf = pd.DataFrame(rows)
    print(f"Country-level coefficients computed for {len(cdf)} countries")
    print(f"  by pathway:")
    print(cdf.groupby("pathway").size().to_string())
    print()

    # Drop countries with extreme SEs (poor identification) and the
    # singleton irrigation pioneer.  Keep both crop-dominant-late samples
    # (full + core) for sensitivity reporting; canonical sample uses 21-c core.
    cdf_canon = cdf.copy()
    # Drop irrigation pioneer (Egypt only)
    cdf_canon = cdf_canon[cdf_canon["cluster"] != 2]
    # Drop crop-dominant late outliers
    cdf_canon = cdf_canon[~cdf_canon["iso3"].isin(CDL_OUTLIERS)]
    # Drop top 5% widest SE on phi_P or phi_L (poor identification)
    se_max_P = cdf_canon["phi_P_se"].quantile(0.95)
    se_max_L = cdf_canon["phi_L_se"].quantile(0.95)
    cdf_canon = cdf_canon[(cdf_canon["phi_P_se"] <= se_max_P)
                            & (cdf_canon["phi_L_se"] <= se_max_L)]
    print(f"After dropping outliers + top-5% SE: {len(cdf_canon)} countries")
    print()

    # Summary stats
    print("Per-country coefficient distribution (canonical sample):")
    print(cdf_canon[["phi_P", "phi_L"]].describe().to_string())
    print()

    # --- OLS, inverse-variance-weighted OLS, Deming regression ---
    x = cdf_canon["phi_P"].values
    y = cdf_canon["phi_L"].values
    sx = cdf_canon["phi_P_se"].values
    sy = cdf_canon["phi_L_se"].values

    # OLS
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"OLS:")
    print(f"  intercept = {ols.params[0]:+.4g} (SE {ols.bse[0]:.3g}, p={ols.pvalues[0]:.3g})")
    print(f"  slope     = {ols.params[1]:+.4g} (SE {ols.bse[1]:.3g}, p={ols.pvalues[1]:.3g})")
    print(f"  R²        = {ols.rsquared:.3f}, N = {int(ols.nobs)}")
    print()

    # Inverse-variance-weighted OLS (weights only on the dependent variable side)
    w = 1.0 / sy**2
    wls = sm.WLS(y, X, weights=w).fit(cov_type="HC1")
    print(f"WLS (weighting on σ_L):")
    print(f"  intercept = {wls.params[0]:+.4g} (SE {wls.bse[0]:.3g}, p={wls.pvalues[0]:.3g})")
    print(f"  slope     = {wls.params[1]:+.4g} (SE {wls.bse[1]:.3g}, p={wls.pvalues[1]:.3g})")
    print(f"  R²        = {wls.rsquared:.3f}, N = {int(wls.nobs)}")
    print()

    # Deming / orthogonal-distance regression (errors in both variables)
    def _line(B, x):
        return B[0] + B[1] * x
    odr = ODR(RealData(x, y, sx=sx, sy=sy), Model(_line),
              beta0=[0.0, -1.0])
    odr_res = odr.run()
    deming_slope = float(odr_res.beta[1])
    deming_intercept = float(odr_res.beta[0])
    deming_slope_se = float(odr_res.sd_beta[1])
    deming_intercept_se = float(odr_res.sd_beta[0])
    # z-test for slope
    from scipy.stats import norm
    deming_p = float(2 * (1 - norm.cdf(abs(deming_slope / deming_slope_se))))
    print(f"Deming (errors in both variables, weighted by 1/σ²):")
    print(f"  intercept = {deming_intercept:+.4g} (SE {deming_intercept_se:.3g})")
    print(f"  slope     = {deming_slope:+.4g} (SE {deming_slope_se:.3g}, p={deming_p:.3g})")
    print()

    # Bootstrap for the Deming slope (500 reps, country resample)
    rng = np.random.default_rng(0xBEEF)
    boot_slopes = []
    n = len(cdf_canon)
    cdf_arr = cdf_canon.reset_index(drop=True)
    for b in range(500):
        idx = rng.integers(0, n, size=n)
        s = cdf_arr.iloc[idx]
        try:
            odr_b = ODR(
                RealData(s["phi_P"].values, s["phi_L"].values,
                          sx=s["phi_P_se"].values, sy=s["phi_L_se"].values),
                Model(_line), beta0=[0.0, deming_slope]
            )
            r = odr_b.run()
            boot_slopes.append(float(r.beta[1]))
        except Exception:
            continue
    boot_slopes = np.array(boot_slopes)
    print(f"Bootstrap Deming slope (500 country resamples):")
    print(f"  median = {np.median(boot_slopes):+.4g}")
    print(f"  95% CI = [{np.quantile(boot_slopes, 0.025):+.4g}, "
          f"{np.quantile(boot_slopes, 0.975):+.4g}]")
    print(f"  Fraction of bootstrap slopes < 0: "
          f"{(boot_slopes < 0).mean():.3f}")
    print()

    # Save results
    out_rows = [
        {"spec": "OLS",          "intercept": float(ols.params[0]),
         "intercept_se": float(ols.bse[0]),
         "slope": float(ols.params[1]),
         "slope_se": float(ols.bse[1]),
         "slope_p": float(ols.pvalues[1]),
         "r2": float(ols.rsquared), "n": int(ols.nobs)},
        {"spec": "WLS_inv_var",  "intercept": float(wls.params[0]),
         "intercept_se": float(wls.bse[0]),
         "slope": float(wls.params[1]),
         "slope_se": float(wls.bse[1]),
         "slope_p": float(wls.pvalues[1]),
         "r2": float(wls.rsquared), "n": int(wls.nobs)},
        {"spec": "Deming",       "intercept": deming_intercept,
         "intercept_se": deming_intercept_se,
         "slope": deming_slope,
         "slope_se": deming_slope_se,
         "slope_p": deming_p,
         "r2": np.nan, "n": int(len(cdf_canon))},
        {"spec": "Deming_boot",  "intercept": np.nan,
         "intercept_se": np.nan,
         "slope": float(np.median(boot_slopes)),
         "slope_se": float(boot_slopes.std()),
         "slope_p": float((boot_slopes >= 0).mean()),
         "r2": np.nan, "n": int(len(cdf_canon))},
    ]
    pd.DataFrame(out_rows).to_parquet(
        DATA / "country_substitution.parquet", index=False)

    # Save country-level coefficients too
    cdf_canon.to_parquet(DATA / "country_substitution_coefs.parquet",
                          index=False)

    # === Figure ===
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Panel A: scatter of (phi_P, phi_L) per country, coloured by pathway
    ax = axes[0]
    pname_color = {"Crop-dominant late": "#A02020",
                    "Pastoral/mixed late": "#0072B2",
                    "High-density intensive": "#009E73",
                    "Early extensifiers": "#D55E00"}
    for path, c in pname_color.items():
        sub = cdf_canon[cdf_canon["pathway"] == path]
        if len(sub) == 0:
            continue
        ax.scatter(sub["phi_P"] * 1000, sub["phi_L"] * 1000,
                   c=c, alpha=0.55, s=18, label=path, edgecolor="none")
    # Lines
    xx = np.linspace(cdf_canon["phi_P"].quantile(0.02),
                       cdf_canon["phi_P"].quantile(0.98), 100)
    ax.plot(xx * 1000, (ols.params[0] + ols.params[1] * xx) * 1000,
            ":", color="grey", linewidth=0.8,
            label=f"OLS  slope = {ols.params[1]:+.2g} (p={ols.pvalues[1]:.2g})")
    ax.plot(xx * 1000, (wls.params[0] + wls.params[1] * xx) * 1000,
            "--", color="black", linewidth=0.8,
            label=f"WLS  slope = {wls.params[1]:+.2g} (p={wls.pvalues[1]:.2g})")
    ax.plot(xx * 1000, (deming_intercept + deming_slope * xx) * 1000,
            "-", color="red", linewidth=1.0,
            label=f"Deming slope = {deming_slope:+.2g} (p={deming_p:.2g})")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.3)
    ax.set_xlabel(r"Per-country demographic coefficient $\hat\phi^P_i$  ($\times 10^{-3}$ per Tg)")
    ax.set_ylabel(r"Per-country Boserupian coefficient $\hat\phi^L_i$  ($\times 10^{-3}$ per Tg)")
    ax.set_title("(a) Country-level (φ^P, φ^L) scatter, post-1700", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    # Panel B: bootstrap distribution of Deming slope
    ax = axes[1]
    ax.hist(boot_slopes, bins=40, color="#A02020", alpha=0.7, edgecolor="white")
    ax.axvline(deming_slope, color="red", linewidth=1.2,
                label=f"point estimate = {deming_slope:+.2g}")
    ax.axvline(np.quantile(boot_slopes, 0.025), color="grey",
                linestyle="--", linewidth=0.7,
                label=f"2.5%: {np.quantile(boot_slopes, 0.025):+.2g}")
    ax.axvline(np.quantile(boot_slopes, 0.975), color="grey",
                linestyle="--", linewidth=0.7,
                label=f"97.5%: {np.quantile(boot_slopes, 0.975):+.2g}")
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Bootstrap Deming slope of φ^L on φ^P")
    ax.set_ylabel("count (500 country-resamples)")
    ax.set_title(f"(b) Bootstrap distribution, "
                  f"P(slope < 0) = {(boot_slopes < 0).mean():.2f}",
                  fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Channel substitution at country level: "
                  "post-1700 KK10 cross-validation panel", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "figK_country_substitution.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figK_country_substitution.png", dpi=160,
                 bbox_inches="tight")
    print(f"Saved {FIG/'figK_country_substitution.pdf'}")
    print(f"Saved {DATA/'country_substitution.parquet'}")


if __name__ == "__main__":
    main()
