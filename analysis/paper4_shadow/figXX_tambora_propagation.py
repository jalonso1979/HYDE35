"""Figure: Tambora 1815 propagation heatmap.

4x4 grid of monthly world temperature anomalies from June 1815 to
September 1816 (16 months) from ModE-RA. Shows how Tambora's April 1815
eruption propagates through the climate system, culminating in the
"Year Without a Summer" of 1816.

The figure is meant for the data section (§2) as a "wow figure" that
showcases what monthly resolution paleo-reanalysis can do, and that
single-image referees will remember.

Source: /Volumes/BIGDATA/MODERA/extracted/ModE-RA_ensmean_temp2_anom_*.nc

Output: analysis/figures/paper4_v2/figXX_tambora_propagation.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

FIG = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/paper4_v2")
MODERA = Path("/Volumes/BIGDATA/MODERA/extracted/ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc")

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main() -> None:
    print(f"Opening {MODERA} ...")
    ds = xr.open_dataset(MODERA, use_cftime=True)

    # Select June 1815 to Sep 1816 (16 months)
    sel = ds.sel(time=slice("1815-06", "1816-09"))
    print(f"Selected {sel.sizes['time']} months, "
          f"lat {sel.sizes['latitude']}, lon {sel.sizes['longitude']}")

    temp = sel.temp2.values  # [16, 96, 192]
    lats = sel.latitude.values
    lons = sel.longitude.values
    times = sel.time.values

    # Common color scale: diverging at zero, range ±2.5°C captures most
    vmax = 2.5
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(4, 4, figsize=(12.8, 7.2),
                              subplot_kw={"aspect": "auto"})
    axes_flat = axes.flat

    # Plot each month
    for k, ax in enumerate(axes_flat):
        if k >= len(times):
            ax.axis("off")
            continue
        t = times[k]
        title = f"{MONTH_NAMES[t.month - 1]} {t.year}"
        im = ax.pcolormesh(lons, lats, temp[k], cmap=cmap, norm=norm,
                            shading="auto")
        ax.set_title(title, fontsize=9.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        # Mark Tambora location (Indonesia, ~8°S, 118°E) with a red star
        ax.plot(118, -8, marker="*", color="black", markersize=8,
                markeredgecolor="white", markeredgewidth=0.6)
        # Set extent to global
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Monthly temperature anomalies (°C, vs.\\ 1901–2000) following the Tambora eruption (April 1815)",
                 y=1.005, x=0.04, ha="left", fontsize=12)
    plt.tight_layout(rect=(0, 0.06, 1, 1.0))
    # Single shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.20, 0.04, 0.6, 0.018])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                       label="Temperature anomaly (°C, ModE-RA ensemble mean vs.\\ 1901–2000)")
    cb.ax.tick_params(labelsize=8.5)
    # Add a small caption beneath
    fig.text(0.04, 0.005,
             "★ marks Tambora (8°S, 118°E). The year-without-a-summer of 1816 is visible as a coherent cold anomaly across the Northern Hemisphere mid-latitudes through June--August 1816.",
             fontsize=8.5, color="#404040", ha="left")
    fig.savefig(FIG / "figXX_tambora_propagation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figXX_tambora_propagation.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG/'figXX_tambora_propagation.pdf'}")


if __name__ == "__main__":
    main()
