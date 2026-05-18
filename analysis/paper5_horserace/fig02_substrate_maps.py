"""Figure 2: world choropleths of the four substrates, 2x2 grid."""
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
FIG = ROOT / "analysis/figures/paper5_horserace/fig02_substrate_maps.pdf"

SUBSTRATES = [
    ("sigma_v_T_pre1750", r"(a) $\sigma_v^T$ 1421--1750 (K)", "viridis"),
    ("H_pred_pwadj", r"(b) Predicted Het, PW-adjusted", "plasma"),
    ("ancestral_yield_log", r"(c) Log ancestral crop yield (kcal/ha)", "YlGn"),
    ("pandemic_intensity_norm", r"(d) Pre-1500 pandemic intensity (normalized)", "Reds"),
]


def main() -> None:
    df = pd.read_parquet(PANEL).set_index("iso3")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             subplot_kw={"projection": ccrs.Robinson()})
    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                  name="admin_0_countries")
    reader = shpreader.Reader(shp)

    for ax, (col, title, cmap) in zip(axes.flat, SUBSTRATES):
        vals = df[col].dropna()
        norm = Normalize(vmin=vals.quantile(0.02), vmax=vals.quantile(0.98))
        cmap_o = matplotlib.colormaps.get_cmap(cmap)
        for country in reader.records():
            iso3 = country.attributes.get("ADM0_A3", "")
            if iso3 in vals.index:
                v = vals.loc[iso3]
                color = cmap_o(norm(v))
            else:
                color = "lightgray"
            ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                              facecolor=color, edgecolor="black", linewidth=0.2)
        ax.set_global()
        ax.set_title(title, fontsize=11)
        sm = cm.ScalarMappable(cmap=cmap_o, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)

    plt.tight_layout()
    plt.savefig(FIG, bbox_inches="tight")
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
