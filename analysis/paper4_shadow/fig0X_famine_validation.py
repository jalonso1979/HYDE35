"""Figure: famine cross-validation strip.

Five small-multiples, one per documented climate-driven famine, showing
the affected country's ModE-RA + CRU annual temperature anomaly (and
precipitation as a second line, on a twin axis) for a 30-year window
centered on the famine. The famine year(s) are highlighted with a
vertical red band.

The plot directly addresses the "is this data any good" referee question
by showing that the panel captures the cooling/drying signature of
canonical famine events.

Events:
  - Great Frost 1709 (France)
  - Tambora year-without-a-summer 1816 (Switzerland)
  - Irish Potato Famine 1845-49 (Ireland)
  - North China Famine 1876-79 (China)
  - Bengal Famine 1943 (Bangladesh)

Output: analysis/figures/paper4_v2/fig0X_famine_validation.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

FAMINES = [
    {"iso": "FRA", "country": "France",       "year": 1709, "span": (1709, 1709),
     "label": "Great Frost 1709",      "kind": "cold"},
    {"iso": "CHE", "country": "Switzerland",  "year": 1816, "span": (1816, 1816),
     "label": "Year Without a Summer 1816 (Tambora)", "kind": "cold"},
    {"iso": "IRL", "country": "Ireland",      "year": 1847, "span": (1845, 1849),
     "label": "Irish Famine 1845–49",  "kind": "cold-wet"},
    {"iso": "CHN", "country": "China",        "year": 1877, "span": (1876, 1879),
     "label": "North China Famine 1876–79", "kind": "drought"},
    {"iso": "BGD", "country": "Bangladesh",   "year": 1943, "span": (1943, 1943),
     "label": "Bengal Famine 1943",    "kind": "drought"},
]


def _country_anomalies(iso: str, year: int, window: int = 15) -> pd.DataFrame:
    """Return T and P anomalies for the country, centered on year."""
    df = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    d = df[df["iso3"] == iso].copy()
    d = d.sort_values("year").reset_index(drop=True)
    # Compute pre-event reference baseline: 30-year mean ending 5 years before
    baseline = d[(d["year"] >= year - 35) & (d["year"] <= year - 5)]
    t_ref = baseline["t_c"].mean()
    p_ref = baseline["p_mm"].mean()
    d["t_anom"] = d["t_c"] - t_ref
    d["p_anom"] = d["p_mm"] - p_ref
    d = d[(d["year"] >= year - window) & (d["year"] <= year + window)].copy()
    return d


def main() -> None:
    fig, axes = plt.subplots(5, 1, figsize=(9.5, 11.5), sharex=False)
    for ax, fam in zip(axes, FAMINES):
        d = _country_anomalies(fam["iso"], fam["year"], window=15)
        # Highlight the famine span
        s0, s1 = fam["span"]
        ax.axvspan(s0 - 0.5, s1 + 0.5, color="#d62728", alpha=0.15, zorder=0)
        # Plot T anomaly
        ax.plot(d["year"], d["t_anom"], color="#1f77b4", linewidth=1.4,
                marker="o", markersize=3.2, label="T anomaly (°C, left)")
        ax.fill_between(d["year"], 0, d["t_anom"],
                        where=(d["t_anom"] < 0), color="#1f77b4", alpha=0.20)
        ax.axhline(0, color="#202020", linewidth=0.6)
        ax.set_ylabel(r"$T$ anomaly (°C)", fontsize=9, color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

        # Twin axis for P
        ax2 = ax.twinx()
        ax2.plot(d["year"], d["p_anom"], color="#2ca02c", linewidth=1.0,
                  marker="s", markersize=2.8, alpha=0.85,
                  label="P anomaly (mm, right)")
        ax2.fill_between(d["year"], 0, d["p_anom"],
                          where=(d["p_anom"] < 0), color="#2ca02c", alpha=0.15)
        ax2.set_ylabel(r"$P$ anomaly (mm/yr)", fontsize=9, color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")

        # Title and event year(s) annotation
        ax.set_title(f"{fam['label']}  ({fam['country']})",
                      loc="left", fontsize=10.5)
        ax.set_xlim(fam["year"] - 15.5, fam["year"] + 15.5)
        ax.grid(alpha=0.25, axis="x")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        # Show the famine year vs surrounding deviation magnitude
        famine_yr_data = d[(d["year"] >= s0) & (d["year"] <= s1)]
        if len(famine_yr_data):
            t_min = famine_yr_data["t_anom"].min()
            p_min = famine_yr_data["p_anom"].min()
            note = ""
            if fam["kind"] in ("cold", "cold-wet"):
                note = f"min $T$ anom during famine: {t_min:+.2f}°C"
            elif fam["kind"] == "drought":
                note = f"min $P$ anom during famine: {p_min:+.1f} mm"
            if note:
                ax.text(0.02, 0.93, note, transform=ax.transAxes,
                        fontsize=8.5, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#404040", alpha=0.9))

    axes[-1].set_xlabel("Calendar year (CE)")
    fig.suptitle("Cross-validation: ModE-RA + CRU climate anomalies in five documented climate-driven famines",
                 y=0.998, x=0.04, ha="left", fontsize=11.5)
    plt.tight_layout()
    fig.savefig(FIG / "fig0X_famine_validation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig0X_famine_validation.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {FIG/'fig0X_famine_validation.pdf'}")


if __name__ == "__main__":
    main()
