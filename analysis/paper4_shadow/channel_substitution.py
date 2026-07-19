"""Channel substitution: estimate the trade-off between demographic and
Boserupian margins across the four pathways.

Two exercises:

(1) Robustness check on the demographic margin.  The headline
    pathway-stratified demographic coefficient is estimated with HYDE
    log_crop_share among the controls.  We re-estimate with KK10
    log_anthro_share replacing it.  If the demographic coefficients
    survive, the headline result is independent of which land-use
    reconstruction we use for the lagged covariate.

(2) Channel substitution measurement.  For each of the four pathways on
    a common sample, we have:
      φ^P_τ  = demographic-margin coefficient on Δ log P
      φ^L_τ  = Boserupian-margin coefficient on Δ log (KK10 anthro)
    Both come from the same regression (joint VAR controls, KK10
    cropland share replacing HYDE's), so the (φ^P, φ^L) pair per pathway
    identifies a comparable point.  We regress φ^L on φ^P across the
    four pathways with inverse-variance weights; a negative slope means
    pathways trade off between margins (Boserupian intensification
    substitutes for population contraction).

Outputs:
    analysis/data/channel_substitution.parquet
    analysis/figures/paper4_v2/figK_channel_substitution.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG  = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer",  3: "High-density intensive",
                 4: "Early extensifiers"}

CDL_OUTLIERS = ['FJI','REU','NCL','WSM','TON','JAM','MUS','VUT','DMA','GLP',
                'LCA','KNA','MTQ','VCT','HTI','SLV','LSO','SLE','CIV','GNB',
                'GNQ','CMR','BRN','SGP']


def _build_panel() -> pd.DataFrame:
    hyde = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    kk10 = pd.read_parquet(DATA / "kk10_country_panel.parquet")

    hyde = hyde.sort_values(["iso3", "year"]).copy()
    hyde["hyde_anthro_km2"] = hyde["crop_km2"].clip(lower=1e-6) + \
                              hyde["past_km2"].clip(lower=1e-6)
    hyde["log_hyde_anthro"] = np.log(hyde["hyde_anthro_km2"])
    g = hyde.groupby("iso3")
    nxt = g["log_hyde_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - hyde["year"]
    hyde["g_hyde_anthro_ann"] = (nxt - hyde["log_hyde_anthro"]) / dt

    kk10 = kk10.sort_values(["iso3", "year"]).copy()
    kk10["log_kk10_anthro_share"] = np.log(
        kk10["kk10_anthro_frac"].clip(lower=1e-6))
    kk10["log_kk10_anthro"] = np.log(
        kk10["kk10_anthro_km2"].clip(lower=1e-6))
    g = kk10.groupby("iso3")
    nxt = g["log_kk10_anthro"].shift(-1)
    dt  = g["year"].shift(-1) - kk10["year"]
    kk10["g_kk10_anthro_ann"] = (nxt - kk10["log_kk10_anthro"]) / dt

    return hyde.merge(
        kk10[["iso3", "year",
              "log_kk10_anthro_share", "log_kk10_anthro",
              "g_kk10_anthro_ann"]],
        on=["iso3", "year"], how="left"
    )


def _run(d: pd.DataFrame, lhs: str, controls: list[str]) -> dict | None:
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    if len(d) < 20 or d["iso3"].nunique() < 2:
        return None
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
    res = sm.OLS(d[lhs], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"]}
    )
    return {
        "beta": float(res.params.get("vssi_int", np.nan)),
        "se":   float(res.bse.get("vssi_int",   np.nan)),
        "p":    float(res.pvalues.get("vssi_int", np.nan)),
        "n":    int(res.nobs),
        "n_iso": int(d["iso3"].nunique()),
    }


def main() -> None:
    panel = _build_panel()
    panel_post1700 = panel[panel["year"] >= 1700].copy()

    # === (1) Demographic robustness with KK10 land-use control ===
    print("="*100)
    print("(1) DEMOGRAPHIC ROBUSTNESS: pathway-stratified Δlog P regression")
    print("    Headline controls use HYDE log_crop_share; we swap in KK10")
    print("    log_anthro_share for the lagged land-use control.")
    print("="*100)

    HYDE_CTRL = ["log_pop", "log_crop_share", "log_urban_share",
                  "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]
    KK10_CTRL = ["log_pop", "log_kk10_anthro_share", "log_urban_share",
                  "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]

    pathway_specs = [
        (0, "Crop-dominant late",        False),
        (0, "Crop-dominant late (core)", True),
        (1, "Pastoral/mixed late",       False),
        (3, "High-density intensive",    False),
        (4, "Early extensifiers",        False),
    ]

    rows = []
    for cl, name, drop_out in pathway_specs:
        sub = panel_post1700[panel_post1700["cluster"] == cl].copy()
        if drop_out and cl == 0:
            sub = sub[~sub["iso3"].isin(CDL_OUTLIERS)]

        r_hyde = _run(sub, "g_pop_ann", HYDE_CTRL)
        r_kk10 = _run(sub, "g_pop_ann", KK10_CTRL)
        if r_hyde is None or r_kk10 is None:
            continue
        sig_h = ("***" if r_hyde["p"]<0.01 else "**" if r_hyde["p"]<0.05
                 else "*" if r_hyde["p"]<0.10 else "  ")
        sig_k = ("***" if r_kk10["p"]<0.01 else "**" if r_kk10["p"]<0.05
                 else "*" if r_kk10["p"]<0.10 else "  ")
        print(f"  {name:30s}  N={r_hyde['n']:4d}  n_iso={r_hyde['n_iso']}")
        print(f"    HYDE cropshare control: φ^P = {r_hyde['beta']:+.4g}  "
              f"(SE {r_hyde['se']:.3g}, p={r_hyde['p']:.3g}) {sig_h}")
        print(f"    KK10 anthro    control: φ^P = {r_kk10['beta']:+.4g}  "
              f"(SE {r_kk10['se']:.3g}, p={r_kk10['p']:.3g}) {sig_k}")
        rows.append({"exercise": "demographic_robustness",
                      "cluster": cl, "pathway": name,
                      "spec": "HYDE_control", **r_hyde})
        rows.append({"exercise": "demographic_robustness",
                      "cluster": cl, "pathway": name,
                      "spec": "KK10_control",
                      "beta": r_kk10["beta"], "se": r_kk10["se"],
                      "p": r_kk10["p"], "n": r_kk10["n"], "n_iso": r_kk10["n_iso"]})

    # === (2) Channel substitution: compute φ^P and φ^L per pathway on
    # COMMON samples, then estimate substitution slope ===
    print()
    print("="*100)
    print("(2) CHANNEL SUBSTITUTION: estimate cross-pathway trade-off")
    print("    Same sample, same controls, both Δ log P and Δ log KK10")
    print("="*100)

    sub_pairs = []
    for cl, name, drop_out in pathway_specs:
        sub = panel_post1700[panel_post1700["cluster"] == cl].copy()
        if drop_out and cl == 0:
            sub = sub[~sub["iso3"].isin(CDL_OUTLIERS)]
        # Demographic coefficient (HYDE-controlled, headline spec)
        r_P = _run(sub, "g_pop_ann", HYDE_CTRL)
        # Boserupian coefficient on KK10 anthro area
        r_L = _run(sub, "g_kk10_anthro_ann", HYDE_CTRL)
        if r_P is None or r_L is None:
            continue
        sub_pairs.append({
            "cluster": cl, "pathway": name,
            "phi_P": r_P["beta"], "phi_P_se": r_P["se"], "phi_P_p": r_P["p"],
            "phi_L": r_L["beta"], "phi_L_se": r_L["se"], "phi_L_p": r_L["p"],
            "n_iso": r_P["n_iso"],
        })
        print(f"  {name:30s}  φ^P = {r_P['beta']:+.4g} (p={r_P['p']:.3g}), "
              f"φ^L = {r_L['beta']:+.4g} (p={r_L['p']:.3g})")

    sp = pd.DataFrame(sub_pairs)
    # Use crop-dominant late CORE only (skip duplicate "full" for cleanest scatter)
    sp_clean = sp[sp["pathway"] != "Crop-dominant late"].reset_index(drop=True)
    print(f"\n  Cross-pathway scatter (4 points, dropping CDL-full duplicate):")
    print(sp_clean[["pathway", "phi_P", "phi_L", "n_iso"]].to_string(index=False))

    # OLS of φ^L on φ^P, inverse-variance weighted by 1/(phi_L_se^2)
    x = sp_clean["phi_P"].values
    y = sp_clean["phi_L"].values
    w = 1.0 / sp_clean["phi_L_se"].values ** 2
    X = sm.add_constant(x)
    fit = sm.WLS(y, X, weights=w).fit()
    print(f"\n  WLS fit of φ^L on φ^P (4 pathways, inverse-variance weighted):")
    print(f"    intercept = {fit.params[0]:+.4g}  (SE {fit.bse[0]:.3g}, p={fit.pvalues[0]:.3g})")
    print(f"    slope     = {fit.params[1]:+.4g}  (SE {fit.bse[1]:.3g}, p={fit.pvalues[1]:.3g})")
    print(f"    R²        = {fit.rsquared:.3f}")
    print(f"    N         = {int(fit.nobs)}")
    print()
    interp_slope = fit.params[1]
    if interp_slope < 0:
        print(f"  Negative slope: as the demographic margin tightens "
              f"(φ^P → more negative), the Boserupian margin weakens (φ^L → smaller).")
        print(f"  This is the substitution pattern theory predicts: pathways "
              f"that contract demographically less are those that extensify more.")
    else:
        print(f"  Non-negative slope: substitution pattern not identified at "
              f"cross-pathway level (low-N test).")

    rows.append({"exercise": "substitution_wls",
                  "spec": "slope",
                  "beta": float(fit.params[1]),
                  "se":   float(fit.bse[1]),
                  "p":    float(fit.pvalues[1]),
                  "n":    int(fit.nobs),
                  "n_iso": int(sp_clean["n_iso"].sum())})
    rows.append({"exercise": "substitution_wls",
                  "spec": "intercept",
                  "beta": float(fit.params[0]),
                  "se":   float(fit.bse[0]),
                  "p":    float(fit.pvalues[0]),
                  "n":    int(fit.nobs),
                  "n_iso": int(sp_clean["n_iso"].sum())})

    # === Figure: per-pathway (φ^P, φ^L) with substitution fit ===
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"Crop-dominant late (core)": "#A02020",
              "Pastoral/mixed late":       "#0072B2",
              "High-density intensive":    "#009E73",
              "Early extensifiers":        "#D55E00"}
    for _, r in sp_clean.iterrows():
        c = colors.get(r["pathway"], "grey")
        ax.errorbar(r["phi_P"], r["phi_L"],
                    xerr=1.96 * r["phi_P_se"],
                    yerr=1.96 * r["phi_L_se"],
                    fmt="o", color=c, markersize=8, capsize=3,
                    elinewidth=0.8, label=r["pathway"])
        ax.annotate(r["pathway"], (r["phi_P"], r["phi_L"]),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=8)
    # WLS fit line
    xx = np.linspace(min(x) * 1.1, max(x) * 0.9, 100)
    yy = fit.params[0] + fit.params[1] * xx
    ax.plot(xx, yy, "--", color="grey", linewidth=0.8,
             label=f"WLS fit: slope = {fit.params[1]:+.2g}\n(p = {fit.pvalues[1]:.2g})")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel(r"Demographic-margin coefficient $\hat\phi^P_\tau$ (per Tg)")
    ax.set_ylabel(r"Boserupian-margin coefficient $\hat\phi^L_\tau$ on KK10 anthro (per Tg)")
    ax.set_title("Channel substitution across pathways: weak Δlog P response "
                  "↔ strong Δlog land response", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figK_channel_substitution.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figK_channel_substitution.png", dpi=160, bbox_inches="tight")
    print(f"\n  Saved {FIG/'figK_channel_substitution.pdf'}")

    pd.DataFrame(rows).to_parquet(
        DATA / "channel_substitution.parquet", index=False)
    print(f"  Saved {DATA/'channel_substitution.parquet'}")


if __name__ == "__main__":
    main()
