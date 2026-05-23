"""Fig 20 — ERA5 vs ModE-RA country-year temperature + precipitation comparison.

For the 7-country Long Shadow panel on the 1950-2008 overlap window.
Produces a 2-panel figure (T scatter + P scatter) with per-country markers
and overall + per-country correlations in the title.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ERA5_PATH = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/era5_country_annual.parquet")
OVERLAP = (1950, 2008)


def make_fig20():
    if not ERA5_PATH.exists():
        raise FileNotFoundError(f"ERA5 panel not built: {ERA5_PATH}")
    era5 = pd.read_parquet(ERA5_PATH)
    modera = assemble_panel_multi()
    keep_cols = [c for c in ("iso3", "year", "t_growing", "p_growing") if c in modera.columns]
    modera = modera[keep_cols]

    merged = era5.merge(modera, on=["iso3", "year"], how="inner")
    merged = merged.loc[merged["year"].between(*OVERLAP)].dropna(subset=["T", "P", "t_growing", "p_growing"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: T
    ax = axes[0]
    for iso in sorted(merged["iso3"].unique()):
        sub = merged[merged["iso3"] == iso]
        ax.scatter(sub["t_growing"], sub["T"], label=iso, alpha=0.6, s=15)
    ax.set_xlabel("ModE-RA t_growing (anomaly)")
    ax.set_ylabel("ERA5 T (K, annual mean)")
    ax.set_title(f"A. Temperature ({OVERLAP[0]}-{OVERLAP[1]})")
    ax.legend(fontsize=7, loc="best")

    # Panel B: P
    ax = axes[1]
    for iso in sorted(merged["iso3"].unique()):
        sub = merged[merged["iso3"] == iso]
        ax.scatter(sub["p_growing"], sub["P"], label=iso, alpha=0.6, s=15)
    ax.set_xlabel("ModE-RA p_growing (anomaly)")
    ax.set_ylabel("ERA5 P (m/h, annual mean)")
    ax.set_title(f"B. Precipitation ({OVERLAP[0]}-{OVERLAP[1]})")
    ax.legend(fontsize=7, loc="best")

    # Pooled + per-country correlations
    corr_T = float(merged[["T", "t_growing"]].corr().iloc[0, 1])
    corr_P = float(merged[["P", "p_growing"]].corr().iloc[0, 1])
    per_country_corr_T = {
        iso: float(sub[["T", "t_growing"]].corr().iloc[0, 1])
        for iso, sub in merged.groupby("iso3")
        if len(sub) >= 5
    }
    per_country_corr_P = {
        iso: float(sub[["P", "p_growing"]].corr().iloc[0, 1])
        for iso, sub in merged.groupby("iso3")
        if len(sub) >= 5
    }

    fig.suptitle(
        f"Fig 20 — ERA5 vs ModE-RA country-year (pooled corr: T={corr_T:.3f}, P={corr_P:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig20_era5_vs_modera_comparison.pdf"
    png = FIG_DIR / "fig20_era5_vs_modera_comparison.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    stats = {
        "corr_T": corr_T,
        "corr_P": corr_P,
        "per_country_corr_T": per_country_corr_T,
        "per_country_corr_P": per_country_corr_P,
        "n_obs": int(len(merged)),
        "n_countries": int(merged["iso3"].nunique()),
        "year_range": (int(merged["year"].min()), int(merged["year"].max())),
    }
    return pdf, png, stats


if __name__ == "__main__":
    pdf, png, stats = make_fig20()
    print(f"wrote {pdf}")
    print(f"corr(ERA5 T, ModE-RA t_growing) = {stats['corr_T']:+.3f}")
    print(f"corr(ERA5 P, ModE-RA p_growing) = {stats['corr_P']:+.3f}")
    print(f"N = {stats['n_obs']} ({stats['n_countries']} countries, "
          f"{stats['year_range'][0]}-{stats['year_range'][1]})")
    print("\nPer-country temperature correlations:")
    for iso, c in sorted(stats["per_country_corr_T"].items()):
        print(f"  {iso}: {c:+.3f}")
    print("\nPer-country precipitation correlations:")
    for iso, c in sorted(stats["per_country_corr_P"].items()):
        print(f"  {iso}: {c:+.3f}")
