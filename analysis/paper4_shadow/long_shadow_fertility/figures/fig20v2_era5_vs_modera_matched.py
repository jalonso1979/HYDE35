"""Fig 20v2 — ERA5 (Apr-Sep anomaly, 1961-1990 baseline) vs ModE-RA growing-season anomaly.

Re-comparison with matched aggregation over the full 1950-2008 overlap
(59 years × 7 countries). The Phase 9 run was limited to 1950-1967 because
the raw ERA5 download was partial; the archive completed in July 2026 and
the v2 panel now spans 1950-2025 (ModE-RA ends 2008, which caps the
comparison window).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
ERA5_V2_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "era5_country_annual_v2.parquet"
)
OVERLAP = (1950, 2008)  # nominal; actual overlap is constrained by ERA5 availability


def make_fig20v2():
    if not ERA5_V2_PATH.exists():
        raise FileNotFoundError(f"ERA5 v2 panel not built: {ERA5_V2_PATH}")
    era5 = pd.read_parquet(ERA5_V2_PATH)
    modera = assemble_panel_multi()
    keep = [c for c in ("iso3", "year", "t_growing", "p_growing") if c in modera.columns]
    modera = modera[keep]

    merged = era5.merge(modera, on=["iso3", "year"], how="inner")
    merged = (merged.loc[merged["year"].between(*OVERLAP)]
                .dropna(subset=["t_growing_era5", "p_growing_era5", "t_growing", "p_growing"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for iso in sorted(merged["iso3"].unique()):
        sub = merged[merged["iso3"] == iso]
        ax.scatter(sub["t_growing"], sub["t_growing_era5"], label=iso, alpha=0.6, s=15)
    lo, hi = -3, 5
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.5, ls="--")
    ax.set_xlabel(r"ModE-RA $t_{growing}$ ($^\circ$C anomaly)")
    ax.set_ylabel(r"ERA5 $t_{growing}$ ($^\circ$C anomaly, 61-90 base)")
    actual_lo = int(merged["year"].min()) if len(merged) else 0
    actual_hi = int(merged["year"].max()) if len(merged) else 0
    ax.set_title(f"A. Temperature ({actual_lo}-{actual_hi})")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    for iso in sorted(merged["iso3"].unique()):
        sub = merged[merged["iso3"] == iso]
        ax.scatter(sub["p_growing"], sub["p_growing_era5"], label=iso, alpha=0.6, s=15)
    ax.set_xlabel(r"ModE-RA $p_{growing}$ (mm anomaly)")
    ax.set_ylabel(r"ERA5 $p_{growing}$ (m/h anomaly, raw)")
    ax.set_title(f"B. Precipitation ({actual_lo}-{actual_hi})")
    ax.legend(fontsize=7, loc="best")

    corr_T = float(merged[["t_growing", "t_growing_era5"]].corr().iloc[0, 1]) if len(merged) else float("nan")
    corr_P = float(merged[["p_growing", "p_growing_era5"]].corr().iloc[0, 1]) if len(merged) else float("nan")
    per_country_T = {iso: float(sub[["t_growing", "t_growing_era5"]].corr().iloc[0, 1])
                      for iso, sub in merged.groupby("iso3") if len(sub) >= 5}
    per_country_P = {iso: float(sub[["p_growing", "p_growing_era5"]].corr().iloc[0, 1])
                      for iso, sub in merged.groupby("iso3") if len(sub) >= 5}

    fig.suptitle(
        f"Fig 20v2 — ERA5 vs ModE-RA matched growing-season anomaly "
        f"(pooled: T={corr_T:+.3f}, P={corr_P:+.3f}, N={len(merged)})",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig20v2_era5_vs_modera_matched.pdf"
    png = FIG_DIR / "fig20v2_era5_vs_modera_matched.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    stats = {
        "corr_T": corr_T, "corr_P": corr_P,
        "per_country_corr_T": per_country_T,
        "per_country_corr_P": per_country_P,
        "n_obs": int(len(merged)),
        "n_countries": int(merged["iso3"].nunique()) if len(merged) else 0,
        "year_range": (actual_lo, actual_hi),
    }
    return pdf, png, stats


if __name__ == "__main__":
    pdf, png, stats = make_fig20v2()
    print(f"wrote {pdf}")
    print(f"corr_T = {stats['corr_T']:+.3f}, corr_P = {stats['corr_P']:+.3f}")
    print(f"N = {stats['n_obs']} ({stats['n_countries']} countries, "
          f"{stats['year_range'][0]}-{stats['year_range'][1]})")
    print("\nPer-country T correlations:")
    for iso, c in sorted(stats["per_country_corr_T"].items()):
        print(f"  {iso}: {c:+.3f}")
