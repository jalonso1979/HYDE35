"""Figure: long-shadow choropleth.

Two-panel world map.
  (a) Pre-industrial inter-annual temperature volatility (sigma_v^T, std
      of annual mean T over 1421-1750), the long-shadow predictor.
  (b) Modern population growth, log(pop 2015-2025 / pop 1950-1960), the
      long-shadow outcome.

The visual centerpiece of Section 5. The R^2 = 0.45 finding lives in the
spatial co-pattern of the two maps.

Output: analysis/figures/paper4_v2/figXY_long_shadow_choropleth.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"


def _predictor() -> pd.DataFrame:
    """Pre-industrial sigma_v^T per country, 1421-1750."""
    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")
    pre = seas[seas["year"].between(1421, 1750)]
    return pre.groupby("iso3", as_index=False).agg(
        sigma_v=("t_mean", "std"))


def _outcome() -> pd.DataFrame:
    """Modern log pop growth, 1950-1960 base -> 2015-2025 endpoint."""
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        pop_early=("pop", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        pop_late=("pop", "mean"))
    m = early.merge(late, on="iso3", how="inner")
    m = m[(m["pop_early"] > 0) & (m["pop_late"] > 0)].copy()
    m["log_pop_growth"] = np.log(m["pop_late"] / m["pop_early"])
    return m[["iso3", "log_pop_growth"]]


def _world() -> gpd.GeoDataFrame:
    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                    name="admin_0_countries")
    gdf = gpd.read_file(shp)
    # Use ISO_A3 if present, else SOV_A3
    gdf["iso3"] = gdf["ISO_A3"].where(gdf["ISO_A3"] != "-99", gdf["SOV_A3"])
    return gdf[["iso3", "NAME", "geometry"]]


def _plot_panel(ax, gdf: gpd.GeoDataFrame, col: str, cmap: str,
                  vmin: float, vmax: float, title: str, cbar_label: str,
                  is_diverging: bool = False) -> object:
    # Background
    ax.set_facecolor("white")
    # Gray for missing
    gdf.plot(ax=ax, color="#E5E5E5", edgecolor="white", linewidth=0.3)
    # Colored where data exists
    if is_diverging:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    has_data = gdf[col].notna()
    plot = gdf[has_data].plot(ax=ax, column=col, cmap=cmap, norm=norm,
                                edgecolor="white", linewidth=0.3, legend=False)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-180, 180); ax.set_ylim(-58, 85)
    for s in ax.spines.values(): s.set_visible(False)
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                         shrink=0.55, pad=0.04, aspect=35)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)
    return plot


def main() -> None:
    print("Loading data ...")
    pred = _predictor()
    out = _outcome()
    print(f"  sigma_v predictor: {len(pred)} countries")
    print(f"  modern pop growth: {len(out)} countries")

    gdf = _world()
    print(f"  Natural Earth world: {len(gdf)} polygons")
    gdf = gdf.merge(pred, on="iso3", how="left").merge(out, on="iso3", how="left")
    print(f"  After merge: sigma_v non-null = {gdf['sigma_v'].notna().sum()}, "
          f"pop_growth non-null = {gdf['log_pop_growth'].notna().sum()}")

    # Color scales
    v_max_sigma = float(np.nanpercentile(gdf["sigma_v"], 97))
    pop_q = (np.nanpercentile(gdf["log_pop_growth"], 3),
              np.nanpercentile(gdf["log_pop_growth"], 97))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.8))

    _plot_panel(axes[0], gdf, "sigma_v", "viridis_r",
                  vmin=float(np.nanpercentile(gdf["sigma_v"], 3)),
                  vmax=v_max_sigma,
                  title=r"(a) Pre-industrial temperature volatility $\sigma_v^T$, 1421--1750",
                  cbar_label=r"$\sigma_v^T$ (°C), country-mean std of annual mean $T$")

    _plot_panel(axes[1], gdf, "log_pop_growth", "YlOrRd",
                  vmin=pop_q[0], vmax=pop_q[1],
                  title=r"(b) Modern log population growth, 1950--1960 $\to$ 2015--2025",
                  cbar_label=r"$\log(P_{2015\text{-}25}/P_{1950\text{-}60})$",
                  is_diverging=False)

    fig.suptitle(r"The long shadow of seasonality: pre-industrial climate volatility and modern population growth ($R^2 = 0.45$)",
                 y=0.998, x=0.04, ha="left", fontsize=11.5)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIG / "figXY_long_shadow_choropleth.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figXY_long_shadow_choropleth.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {FIG/'figXY_long_shadow_choropleth.pdf'}")


if __name__ == "__main__":
    main()
