"""Figure 1: world map of agricultural pathways at country centroids."""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette, MARKERS

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}
PATHWAY_ORDER = [3, 4, 0, 1, 2]


def main() -> None:
    ep = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    centroids = ep[["iso3", "centroid_lat", "centroid_lon"]].drop_duplicates()
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    df = centroids.merge(clust, on="iso3", how="inner").dropna(subset=["centroid_lat", "centroid_lon"])
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # Light coastline-y feel: draw a soft grey land box
    ax.axhspan(-90, 90, color="#FAFAFA", zorder=0)
    palette = gray_palette(len(PATHWAY_ORDER))
    for i, k in enumerate(PATHWAY_ORDER):
        sub = df[df["cluster"] == k]
        if len(sub) == 0: continue
        ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
                   s=45, marker=MARKERS[i % len(MARKERS)],
                   color=palette[i], edgecolor="#202020", linewidth=0.5,
                   alpha=0.85, label=f"{PATHWAY_NAMES[k]} (N={len(sub)})", zorder=3)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xticks([-180,-120,-60,0,60,120,180])
    ax.set_yticks([-60,-30,0,30,60])
    ax.set_title("Five agricultural pathways from HYDE 3.5 trajectory clustering",
                 loc="left")
    ax.grid(True, color="#E0E0E0", linewidth=0.4)
    ax.legend(loc="lower left", frameon=False, ncol=1, fontsize=7.5)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig01_pathways_map.{ext}")
    plt.close(fig)
    print(f"Saved {FIG / 'fig01_pathways_map.[pdf|png]'}")


if __name__ == "__main__":
    main()
