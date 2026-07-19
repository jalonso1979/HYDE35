"""Re-run Stage 1 with a Matranga-style storage-demand index that combines
intra-annual T variation with the share of months below growing threshold.

The Matranga 2024 argument is that storage demand drives intensive ag adoption.
Storage demand is high when many months are non-productive
(too cold OR too dry) and you need to bridge them. So:

    storage_index = (months below 10C threshold) * (annual T range)
or
    storage_index = std(monthly productivity) where
                    productivity_m = 1[5 <= T_m <= 30] * 1[P_m >= 30 mm]

We compute both and test which (if either) predicts pathway selection.

Outputs:
    analysis/data/stage1_storage_results.parquet
    analysis/figures/paper4/fig3d_storage_pathway.png
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}


def main() -> None:
    print("Loading monthly ModE-RA + CRU climatology...", flush=True)
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0.0)

    pre = df[df["year"].between(1421, 1750)].copy()

    # productivity indicator per (country, year, month): T in [5,30] and P >= 30mm
    pre["productive"] = ((pre["t_abs"] >= 5.0) & (pre["t_abs"] <= 30.0)
                         & (pre["p_abs"] >= 30.0)).astype(float)
    pre["cold_month"] = (pre["t_abs"] < 5.0).astype(float)
    pre["hot_month"] = (pre["t_abs"] > 30.0).astype(float)
    pre["dry_month"] = (pre["p_abs"] < 30.0).astype(float)

    # Year-level features
    yr = pre.groupby(["iso3", "year"], as_index=False).agg(
        productive_months=("productive", "sum"),
        cold_months=("cold_month", "sum"),
        hot_months=("hot_month", "sum"),
        dry_months=("dry_month", "sum"),
        t_max=("t_abs", "max"),
        t_min=("t_abs", "min"),
        p_max=("p_abs", "max"),
        p_min=("p_abs", "min"),
        prod_std=("productive", "std"),
    )
    yr["sigma_s"] = yr["t_max"] - yr["t_min"]
    yr["sigma_p"] = yr["p_max"] - yr["p_min"]
    yr["non_prod_months"] = 12 - yr["productive_months"]
    yr["storage_index"] = yr["non_prod_months"] * yr["sigma_s"] / 12.0
    yr["storage_index_p"] = yr["non_prod_months"] * yr["sigma_p"] / 12.0

    # Country-level pre-industrial means
    feats = yr.groupby("iso3", as_index=False).agg(
        sigma_s=("sigma_s", "mean"),
        sigma_p=("sigma_p", "mean"),
        productive_months=("productive_months", "mean"),
        non_prod_months=("non_prod_months", "mean"),
        cold_months=("cold_months", "mean"),
        dry_months=("dry_months", "mean"),
        storage_index=("storage_index", "mean"),
        storage_index_p=("storage_index_p", "mean"),
    )

    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)
    clust["cluster"] = clust["cluster"].astype(int)
    d = feats.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    d["pathway"] = d["cluster"].map(PATHWAY_NAMES)
    print(f"Sample: {len(d)} countries", flush=True)

    print("\n=== Country-level summary by pathway ===")
    print(d.groupby("pathway").agg(
        n=("iso3", "count"),
        productive_months=("productive_months", "mean"),
        non_prod_months=("non_prod_months", "mean"),
        sigma_s=("sigma_s", "mean"),
        sigma_p=("sigma_p", "mean"),
        storage_index=("storage_index", "mean"),
    ).round(2))

    # Drop singleton Egypt cluster for inferential tests
    big = d[d["cluster"] != 2].copy()
    print(f"\nAfter dropping singleton Egypt: {len(big)} countries")

    # ANOVA tests for each candidate measure
    measures = ["productive_months", "non_prod_months", "sigma_s", "sigma_p",
                "storage_index", "storage_index_p", "cold_months", "dry_months"]
    rows = []
    for m in measures:
        groups = [big.loc[big["cluster"] == k, m].values
                  for k in sorted(big["cluster"].unique())]
        groups = [g[~np.isnan(g)] for g in groups]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        rows.append({"measure": m, "F": F, "p": p})
    summary = pd.DataFrame(rows).sort_values("p")
    print("\n=== ANOVA: each climate measure across pathways ===")
    print(summary.to_string(index=False))
    summary.to_parquet(DATA / "stage1_storage_results.parquet", index=False)

    # MNLogit with storage_index
    print("\n=== MNLogit: pathway ~ storage_index ===")
    X = sm.add_constant(big[["storage_index", "productive_months"]])
    y = big["cluster"]
    res = sm.MNLogit(y, X).fit(method="bfgs", maxiter=200, disp=False)
    print(f"Pseudo-R^2 = {res.prsquared:.4f}, LLR p = {res.llr_pvalue:.4g}, N = {int(res.nobs)}")
    print(res.summary())

    # Save the best measure plot
    best = summary.iloc[0]["measure"]
    print(f"\nBest measure by p-value: {best}")
    if summary.iloc[0]["p"] < 0.10:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = (big.groupby("pathway")[best].mean().sort_values().index.tolist())
        data = [big.loc[big["pathway"] == p, best].values for p in order]
        bp = ax.boxplot(data, labels=order, showmeans=True, patch_artist=True,
                        medianprops=dict(color="black"))
        palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel(best.replace("_", " "))
        ax.set_xlabel("Agricultural pathway")
        F = summary.iloc[0]["F"]; p = summary.iloc[0]["p"]
        ax.set_title(f"Stage 1: storage-demand measure '{best}' by pathway "
                     f"(F={F:.2f}, p={p:.3g})")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        out = FIG / "fig3d_storage_pathway.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
