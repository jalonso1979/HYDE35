"""Robustness v2: ModE-RA vs country-level ERA5 calibration (apples-to-apples).

Replaces the earlier calibration that compared ModE-RA against a 25-region
ERA5 panel. Uses the corrected era5_country_monthly.parquet built from raw
0.25 deg ERA5 grids with country-level area-weighting.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    per_m = pd.read_parquet(DATA / "era5_modera_calibration_monthly.parquet")
    per_a = pd.read_parquet(DATA / "era5_modera_calibration_annual.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.hist(per_m["t_bias"], bins=30, color="#0072B2", alpha=0.7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(r"ERA5 $-$ (ModE-RA+climatology) mean T (°C), 1950–2008")
    ax.set_ylabel("Countries")
    med_b = per_m["t_bias"].abs().median()
    ax.set_title(f"Country-month mean bias\nmedian |bias| = {med_b:.2f}°C")

    ax = axes[1]
    ax.hist(per_m["t_corr_m"], bins=30, color="#D55E00", alpha=0.7)
    med_c = per_m["t_corr_m"].median()
    ax.set_xlabel("Monthly country-level T correlation")
    ax.set_ylabel("Countries")
    ax.set_title(f"Country-month T correlation\nmedian = {med_c:.3f}")
    ax.set_xlim(0, 1)

    ax = axes[2]
    ax.hist(per_a["corr_anom"], bins=30, color="#009E73", alpha=0.7)
    med_a = per_a["corr_anom"].median()
    ax.set_xlabel("Annual T anomaly correlation (country-detrended)")
    ax.set_ylabel("Countries")
    ax.set_title(f"Year-to-year skill\nmedian = {med_a:.3f}")
    ax.set_xlim(-0.5, 1)

    plt.tight_layout()
    fig.savefig(FIG / "fig12_calibration.png", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG / 'fig12_calibration.png'}")

    print(f"\n== Headline numbers for paper ==")
    print(f"  Monthly:  median |bias| = {med_b:.2f}°C, median t-corr = {med_c:.3f}")
    print(f"  Annual:   median anomaly corr = {med_a:.3f}")
    print(f"  Monthly: % corr > 0.90 = {(per_m['t_corr_m'] > 0.90).mean()*100:.1f}%")
    print(f"  Monthly: % corr > 0.95 = {(per_m['t_corr_m'] > 0.95).mean()*100:.1f}%")
    print(f"  Annual:  % anom corr > 0.5 = {(per_a['corr_anom'] > 0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
