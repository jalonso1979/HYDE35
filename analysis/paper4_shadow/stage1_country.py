"""Stage 1 (selection) re-estimation at country level with real
intra-annual seasonality from ModE-RA + CRU climatology (1421-1750 mean).

Compares against the paper's earlier proxy-based regional estimate.

Outputs:
    analysis/data/stage1_results.parquet
    analysis/figures/paper4/fig3b_seasonality_pathway_country.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"
FIG.mkdir(parents=True, exist_ok=True)

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}


def main() -> None:
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)].copy()
    seas_country = pre.groupby("iso3", as_index=False).agg(
        sigma_s=("sigma_s", "mean"),
        sigma_s_std=("sigma_s_std", "mean"),
        t_mean=("t_mean", "mean"),
        p_annual=("p_annual", "mean"),
        growing_dd=("growing_dd", "mean"),
        monsoon=("monsoon_intensity", "mean"),
        sigma_v=("t_mean", "std"),
    )

    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)
    clust["cluster"] = clust["cluster"].astype(int)

    df = seas_country.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)
    print(f"Stage 1 sample: {len(df)} countries across {df['cluster'].nunique()} pathways")

    # Drop the singleton irrigation pioneer (Egypt) to keep multinomial well-posed
    big = df[df["cluster"] != 2].copy()
    print(f"After dropping singleton cluster 2 (Egypt): {len(big)} countries")
    print()

    # Pairwise ANOVA-style summary: sigma_s by pathway
    print("=== Seasonality (sigma_s) by pathway, country-level ===")
    summary = big.groupby("pathway").agg(
        n=("iso3", "count"),
        sigma_s_mean=("sigma_s", "mean"),
        sigma_s_sd=("sigma_s", "std"),
        sigma_v_mean=("sigma_v", "mean"),
        t_mean=("t_mean", "mean"),
        growing_dd=("growing_dd", "mean"),
        monsoon=("monsoon", "mean"),
    ).round(2)
    print(summary)
    summary.to_csv(DATA / "stage1_summary_by_pathway.csv")

    # One-way ANOVA
    groups = [big.loc[big["cluster"] == k, "sigma_s"].values
              for k in sorted(big["cluster"].unique())]
    F, p = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA F = {F:.3f}, p = {p:.4g}")

    # Multinomial logit (vs reference category = high-density intensive, cluster 3)
    print("\n=== Multinomial logit ===")
    big = big.dropna(subset=["sigma_s", "t_mean", "growing_dd", "monsoon"]).copy()
    big["log_growing_dd"] = np.log1p(big["growing_dd"])
    X = sm.add_constant(big[["sigma_s", "t_mean", "log_growing_dd"]])
    y = big["cluster"]
    try:
        model = sm.MNLogit(y, X)
        res = model.fit(method="bfgs", maxiter=200, disp=False)
        print(res.summary())
        # Save the coefficient table
        coef = res.params.copy()
        coef.columns = [PATHWAY_NAMES[int(c)] for c in coef.columns]
        coef.to_csv(DATA / "stage1_mnlogit_coefs.csv")
        pvals = res.pvalues.copy()
        pvals.columns = [PATHWAY_NAMES[int(c)] for c in pvals.columns]
        pvals.to_csv(DATA / "stage1_mnlogit_pvals.csv")
    except Exception as e:
        print(f"MNLogit failed: {e}")

    # Comparison with old proxy approach: lin reg of sigma_s on pathway dummies
    print("\n=== OLS: sigma_s ~ pathway dummies (joint test of pathway differences) ===")
    big_ols = big.copy()
    dummies = pd.get_dummies(big_ols["pathway"], drop_first=True).astype(float)
    Xo = sm.add_constant(dummies)
    ols = sm.OLS(big_ols["sigma_s"], Xo).fit()
    print(ols.summary())
    big["sigma_s_hat"] = ols.fittedvalues

    # Boxplot figure
    print("\n=== Generating Fig 3b: country-level seasonality by pathway ===")
    order = (
        big.groupby("pathway")["sigma_s"].mean().sort_values().index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [big.loc[big["pathway"] == p, "sigma_s"].values for p in order]
    bp = ax.boxplot(data, labels=order, showmeans=True, patch_artist=True,
                    medianprops=dict(color="black"))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel(r"$\sigma_s$ — intra-annual temp range (°C), 1421–1750")
    ax.set_xlabel("Agricultural pathway")
    ax.set_title("Stage 1: Country-level historical seasonality by pathway "
                 f"(F = {F:.2f}, p = {p:.3g})")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3b_seasonality_pathway_country.png", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG / 'fig3b_seasonality_pathway_country.png'}")

    # Also do a 2D scatter: sigma_s vs growing_dd colored by pathway
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = dict(zip(order, plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))))
    for p_ in order:
        sub = big[big["pathway"] == p_]
        ax.scatter(sub["growing_dd"], sub["sigma_s"], s=60,
                   color=palette[p_], alpha=0.7, edgecolor="white", label=p_)
    ax.set_xlabel("Growing degree days (base 10°C, pre-industrial mean)")
    ax.set_ylabel(r"$\sigma_s$ — intra-annual temp range (°C)")
    ax.set_title("Stage 1: climate selection space")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3c_climate_selection_space.png", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG / 'fig3c_climate_selection_space.png'}")

    df.to_parquet(DATA / "stage1_country_features.parquet", index=False)
    print(f"\nSaved stage1_country_features.parquet")


if __name__ == "__main__":
    main()
