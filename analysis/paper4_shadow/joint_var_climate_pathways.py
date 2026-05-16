"""Joint VAR robustness exercise (a): re-cluster pathways using ONLY climate
primitives, re-run the three-equation joint VAR, regenerate the pathway
asymmetry table and figure.

The existing pathway typology (paper1_clustered_features.parquet) is built on
HYDE-trajectory features: peak_ag_expansion_year, peak_pop_growth_year,
max_density, density_1750, crop_share_1750, urban_1750, etc.  These are
downstream outcomes of the same processes the joint VAR is trying to
explain, so the "cross-pathway asymmetry" finding partly reduces to "the
regression differs across countries that already differ in 1750
demographics".

Here we re-cluster countries using only pre-industrial climate primitives
(productive_months, intra-annual T and P range, mean T, mean P,
inter-annual T volatility), all computed over 1421-1750 from the ModE-RA +
CRU panel.  K=5 KMeans, the same dimensionality the paper uses elsewhere.
Then re-attach the new climate-only labels to the joint VAR panel and
re-estimate the three-equation system.

Outputs:
    analysis/data/climate_pathways_country.parquet           -- new labels
    analysis/data/joint_var_climate_pathways_results.parquet -- new slopes
    analysis/figures/paper4_v2/figJ_climate_pathways.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

# Climate primitives used for the new clustering, all pre-industrial means
CLIMATE_FEATURES = [
    "productive_months",   # months with T in [5,30] and P >= 30 mm
    "sigma_s",             # intra-annual T range, max - min monthly T
    "sigma_p",             # intra-annual P range, max - min monthly P
    "t_mean",              # annual-mean T, country-mean over 1421-1750
    "p_mean",              # annual-mean P, country-mean over 1421-1750
    "t_volatility",        # inter-annual sigma of annual-mean T, 1421-1750
]


def _build_climate_features() -> pd.DataFrame:
    print("Loading ModE-RA + CRU climatology ...", flush=True)
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0.0)
    pre = df[df["year"].between(1421, 1750)].copy()

    pre["productive"] = ((pre["t_abs"] >= 5.0) & (pre["t_abs"] <= 30.0)
                         & (pre["p_abs"] >= 30.0)).astype(float)
    yr = pre.groupby(["iso3", "year"], as_index=False).agg(
        productive_months=("productive", "sum"),
        t_max=("t_abs", "max"),
        t_min=("t_abs", "min"),
        p_max=("p_abs", "max"),
        p_min=("p_abs", "min"),
        t_year=("t_abs", "mean"),
        p_year=("p_abs", "sum"),  # annual total precip
    )
    yr["sigma_s"] = yr["t_max"] - yr["t_min"]
    yr["sigma_p"] = yr["p_max"] - yr["p_min"]
    feats = yr.groupby("iso3", as_index=False).agg(
        productive_months=("productive_months", "mean"),
        sigma_s=("sigma_s", "mean"),
        sigma_p=("sigma_p", "mean"),
        t_mean=("t_year", "mean"),
        p_mean=("p_year", "mean"),
        t_volatility=("t_year", "std"),
    )
    print(f"  built climate features for {len(feats)} countries")
    return feats


def _cluster_on_climate(feats: pd.DataFrame, k: int = 5, seed: int = 42
                          ) -> tuple[pd.DataFrame, dict]:
    X = feats[CLIMATE_FEATURES].dropna()
    iso = feats.loc[X.index, "iso3"].values
    X_std = StandardScaler().fit_transform(X.values)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X_std)
    sil = silhouette_score(X_std, labels)
    out = pd.DataFrame({"iso3": iso, "climate_cluster": labels})
    return out, {"silhouette": sil, "inertia": km.inertia_, "n": len(out)}


def _label_clusters(out: pd.DataFrame, feats: pd.DataFrame) -> dict:
    """Assign descriptive names to the new climate-only clusters.

    Order clusters by mean productive_months (low -> high) and label them
    by quintile of storage demand.  This is mechanical and reproducible.
    """
    d = out.merge(feats, on="iso3", how="inner")
    cluster_means = d.groupby("climate_cluster")["productive_months"].mean().sort_values()
    names = {}
    descriptors = [
        "Low-storage (driest)",   # fewest productive months
        "Moderate-storage low",
        "Mid-storage",
        "Moderate-storage high",
        "High-storage (wettest)", # most productive months
    ]
    for descr, cl in zip(descriptors, cluster_means.index):
        names[int(cl)] = descr
    return names


def _build_joint_panel() -> pd.DataFrame:
    """Read the existing assembled joint VAR panel; it already has lagged
    log-levels, demeaned climate, and VSSI exposure attached."""
    p = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    print(f"  loaded joint VAR panel: {len(p):,} rows, "
          f"{p['iso3'].nunique()} countries")
    return p


def _run_one(d: pd.DataFrame, lhs: str, controls: list[str]) -> dict:
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    if len(d) < 20 or d["iso3"].nunique() < 2:
        return None
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}


def main() -> None:
    print("=== Joint VAR robustness (a): climate-only pathway clustering ===\n")

    # Step 1: build climate primitives
    feats = _build_climate_features()

    # Step 2: cluster on climate primitives only
    labels, meta = _cluster_on_climate(feats, k=5)
    names = _label_clusters(labels, feats)
    print(f"\nClimate-only K=5 clustering: silhouette={meta['silhouette']:.3f}, "
          f"N={meta['n']} countries")
    for cl, descr in names.items():
        n = (labels['climate_cluster'] == cl).sum()
        mean_pm = feats[feats['iso3'].isin(labels.loc[labels['climate_cluster']==cl, 'iso3'])]['productive_months'].mean()
        mean_ts = feats[feats['iso3'].isin(labels.loc[labels['climate_cluster']==cl, 'iso3'])]['sigma_s'].mean()
        print(f"  Cluster {cl} ({descr}): N={n}, prod_months={mean_pm:.2f}, "
              f"sigma_s={mean_ts:.2f}°C")
    labels.to_parquet(DATA / "climate_pathways_country.parquet", index=False)
    print(f"\nSaved {DATA/'climate_pathways_country.parquet'}")

    # Step 3: cross-tab against old HYDE-derived clusters
    old = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    old = old.dropna(subset=["iso3", "cluster"])
    cross = labels.merge(old[["iso3", "cluster"]], on="iso3", how="inner")
    cross.columns = ["iso3", "climate_cluster", "hyde_cluster"]
    print("\n=== Cross-tab: climate-only clusters vs. HYDE-derived clusters ===")
    ct = pd.crosstab(cross["climate_cluster"], cross["hyde_cluster"], margins=True)
    print(ct.to_string())

    # Step 4: load joint VAR panel, attach new clusters, re-estimate
    panel = _build_joint_panel()
    panel = panel.merge(labels, on="iso3", how="inner")
    panel["climate_pathway"] = panel["climate_cluster"].map(names)
    print(f"\nJoint VAR sample with climate clusters: {len(panel):,} rows, "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['climate_cluster'].nunique()} pathways")

    controls = ["log_pop", "log_crop_share", "log_urban_share",
                "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]

    # Pooled estimates
    eqs = [("Δ_ann log Pop", "g_pop_ann"),
           ("Δ_ann log CropShare", "g_crop_ann"),
           ("Δ_ann log UrbanShare", "g_urb_ann")]
    print("\n=== Pooled three-equation joint VAR (climate-pathway sample) ===")
    for name, dep in eqs:
        r = _run_one(panel, dep, controls)
        if r is None: continue
        b = r["params"].get("vssi_int", np.nan)
        se = r["bse"].get("vssi_int", np.nan)
        p = r["p"].get("vssi_int", np.nan)
        print(f"  {name}: VSSI β={b:+.6f} (SE {se:.6f}) p={p:.3g}, "
              f"R²={r['r2']:.4f}, N={r['n']}")

    # Pathway-stratified
    print("\n=== Pathway-stratified VSSI slopes (climate clusters) ===")
    rows = []
    for cl in sorted(panel["climate_cluster"].unique()):
        sub = panel[panel["climate_cluster"] == cl]
        if len(sub) < 20 or sub["iso3"].nunique() < 2:
            continue
        rp = _run_one(sub, "g_pop_ann", controls)
        rc = _run_one(sub, "g_crop_ann", controls)
        ru = _run_one(sub, "g_urb_ann", controls)
        if rp is None or rc is None or ru is None:
            continue
        row = {"climate_cluster": int(cl), "pathway": names[cl],
               "n": rp["n"], "n_countries": sub["iso3"].nunique(),
               "pop_vssi_beta": rp["params"].get("vssi_int", np.nan),
               "pop_vssi_se":   rp["bse"].get("vssi_int", np.nan),
               "pop_vssi_p":    rp["p"].get("vssi_int", np.nan),
               "crop_vssi_beta": rc["params"].get("vssi_int", np.nan),
               "crop_vssi_se":   rc["bse"].get("vssi_int", np.nan),
               "crop_vssi_p":    rc["p"].get("vssi_int", np.nan),
               "urb_vssi_beta": ru["params"].get("vssi_int", np.nan),
               "urb_vssi_se":   ru["bse"].get("vssi_int", np.nan),
               "urb_vssi_p":    ru["p"].get("vssi_int", np.nan)}
        rows.append(row)
    res = pd.DataFrame(rows)
    print(res[["pathway", "n", "n_countries",
                "pop_vssi_beta", "pop_vssi_p",
                "crop_vssi_beta", "crop_vssi_p",
                "urb_vssi_beta", "urb_vssi_p"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"))
    res.to_parquet(DATA / "joint_var_climate_pathways_results.parquet", index=False)

    # Step 5: figure
    if len(res):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        panel_defs = [
            ("pop_vssi", "(a) Δ log Pop", r"$\beta$ per Tg VSSI",
             "#202020"),
            ("crop_vssi", "(b) Δ log CropShare", r"$\beta$ per Tg VSSI",
             "#A02020"),
            ("urb_vssi", "(c) Δ log UrbanShare", r"$\beta$ per Tg VSSI",
             "#0072B2"),
        ]
        for ax, (col, title, xlab, c) in zip(axes, panel_defs):
            order = res.sort_values(f"{col}_beta").reset_index(drop=True)
            y = np.arange(len(order))
            ax.errorbar(order[f"{col}_beta"], y,
                        xerr=1.96 * order[f"{col}_se"],
                        fmt="o", color=c, markerfacecolor="white",
                        markeredgewidth=1, ecolor=c, elinewidth=0.7,
                        capsize=2.5)
            for i, rr in order.iterrows():
                s = ("***" if rr[f"{col}_p"] < 0.01 else "**" if rr[f"{col}_p"] < 0.05
                     else "*" if rr[f"{col}_p"] < 0.10 else "")
                ax.text(rr[f"{col}_beta"], i + 0.18,
                        f"$N={int(rr['n'])}$ {s}", ha="center", fontsize=8.5)
            ax.axvline(0, color="#404040", linewidth=0.6)
            ax.set_yticks(y); ax.set_yticklabels(order["pathway"])
            ax.set_xlabel(xlab)
            ax.set_title(title, loc="left", fontsize=10.5)
            ax.grid(alpha=0.3)
        fig.suptitle("Joint VAR with climate-only pathways: VSSI response by cluster, 1500--1900",
                     y=1.04, x=0.04, ha="left", fontsize=11.5)
        plt.tight_layout()
        fig.savefig(FIG / "figJ_climate_pathways.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figJ_climate_pathways.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"\nSaved {FIG/'figJ_climate_pathways.pdf'}")


if __name__ == "__main__":
    main()
