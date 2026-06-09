"""Pathway typology validation: defend K=5 choice with silhouette scores,
alternative K, stability under bootstrap, and feature loadings.

Output:
    analysis/data/pathway_silhouette.parquet
    analysis/data/pathway_stability.parquet
    analysis/figures/paper4_v2/fig10_pathway_validation.pdf
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

FEATURE_COLS = [
    "peak_ag_expansion_year", "max_density", "density_1750", "density_1000",
    "crop_share_1750", "rice_share_1750", "irrigation_1750", "urban_1750",
    "pop_growth_0_1000", "pop_growth_1000_1750", "ag_intensity_change",
]


def main() -> None:
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)

    raw = clust[FEATURE_COLS].dropna()
    rows_used = clust.loc[raw.index, "iso3"].values
    orig_labels = clust.loc[raw.index, "cluster"].astype(int).values
    X_scaled = StandardScaler().fit_transform(raw.values)
    print(f"Clustering sample: N = {X_scaled.shape[0]} countries, "
          f"{X_scaled.shape[1]} features (z-scored, no log transform)")
    print("Original cluster sizes:")
    print(pd.Series(orig_labels).value_counts().sort_index())

    # Silhouette of the ORIGINAL paper-1 K=5 partition on these features
    sil_orig = silhouette_score(X_scaled, orig_labels)
    print(f"\nSilhouette of original K=5 partition: {sil_orig:.4f}")

    # Silhouette across K
    print("\n=== Silhouette score across K ===")
    sil = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X_scaled)
        s = silhouette_score(X_scaled, km.labels_)
        sil[k] = s
        print(f"  K = {k}: silhouette = {s:.4f}")
    sil_df = pd.DataFrame({"k": list(sil.keys()), "silhouette": list(sil.values())})
    sil_df.to_parquet(DATA / "pathway_silhouette.parquet", index=False)

    # Bootstrap stability at K=5: how often do pairs cluster together?
    print("\n=== Bootstrap stability at K=5 ===")
    n = X_scaled.shape[0]
    rng = np.random.default_rng(42)
    n_bootstrap = 200
    pair_counts = np.zeros((n, n), dtype=np.int32)
    cooccurrence = np.zeros((n, n), dtype=np.int32)
    for b in range(n_bootstrap):
        boot_idx = rng.choice(n, size=n, replace=True)
        unique_boot = np.unique(boot_idx)
        if len(unique_boot) < 5:
            continue
        X_boot = X_scaled[unique_boot]
        km = KMeans(n_clusters=5, random_state=b, n_init=20).fit(X_boot)
        labels = km.labels_
        for i, ui in enumerate(unique_boot):
            for j, uj in enumerate(unique_boot):
                pair_counts[ui, uj] += 1
                if labels[i] == labels[j]:
                    cooccurrence[ui, uj] += 1
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b+1}/{n_bootstrap}", flush=True)
    pair_prob = np.where(pair_counts > 0, cooccurrence / np.maximum(pair_counts, 1), 0.0)

    # Stability metric: use the ORIGINAL paper-1 cluster labels as reference.
    # For each country, fraction of its same-cluster partners that re-cluster
    # with it under bootstrap resampling.
    same_cluster = orig_labels[:, None] == orig_labels[None, :]
    np.fill_diagonal(same_cluster, False)
    partners_per_row = same_cluster.sum(axis=1)
    stab = np.where(partners_per_row > 0,
                     (pair_prob * same_cluster).sum(axis=1) /
                     np.maximum(partners_per_row, 1),
                     np.nan)
    stab_df = pd.DataFrame({"iso3": rows_used, "stability": stab,
                             "cluster": orig_labels})
    stab_df.to_parquet(DATA / "pathway_stability.parquet", index=False)
    print(f"Median stability: {np.nanmedian(stab):.3f}")
    print(f"25th percentile: {np.nanpercentile(stab, 25):.3f}")
    print("Stability by original cluster:")
    print(stab_df.groupby("cluster")["stability"].agg(["mean", "median", "count"]).round(3))

    # ARI between K=5 partition and other K (lumping/splitting)
    print("\n=== Adjusted Rand Index: original K=5 vs alternative K ===")
    for k_alt in [3, 4, 6, 7]:
        km_a = KMeans(n_clusters=k_alt, random_state=42, n_init=50).fit(X_scaled)
        ari = adjusted_rand_score(orig_labels, km_a.labels_)
        print(f"  ARI(orig K=5, K={k_alt}) = {ari:.3f}")

    # Plot: silhouette vs K + stability histogram
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    ax = axes[0]
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="o",
            color="#202020", markersize=5, linewidth=1.0)
    ax.axvline(5, color="#A0A0A0", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Number of clusters $K$")
    ax.set_ylabel("Silhouette score")
    ax.set_title("(a) Silhouette score by $K$", loc="left")
    ax.set_xticks(range(2, 9))

    ax = axes[1]
    ax.hist(stab_df["stability"].dropna(), bins=20,
            color="#909090", edgecolor="#202020", linewidth=0.5)
    ax.set_xlabel("Bootstrap co-cluster stability")
    ax.set_ylabel("Countries")
    ax.set_title("(b) Country-level stability, $K{=}5$", loc="left")
    ax.set_xlim(0, 1.02)

    plt.tight_layout()
    out = FIG / "fig10_pathway_validation.pdf"
    fig.savefig(out)
    fig.savefig(FIG / "fig10_pathway_validation.png")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
