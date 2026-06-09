"""Documentary cross-validation: do ModE-RA and HYDE capture five well-known
historical climate-driven famines?

Events:
  1709 Great Frost — Europe, severe winter -> spring food shortages (FRA, GBR, DEU)
  1816 Year Without Summer — post-Tambora cold in Europe + N America (NLD, GBR, USA)
  1845-49 Irish Potato Famine — primarily blight but cool/wet years (IRL)
  1876-79 North China Famine — drought disaster (CHN)
  1943 Bengal Famine — wartime + cyclone + crop failure (IND, BGD)

For each: check the country-month climate anomaly and the contemporaneous
decadal HYDE population trajectory. Output table + 5-panel figure.
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


EVENTS = [
    ("Great Frost", 1709, ["FRA", "GBR", "DEU", "ESP", "POL"]),
    ("Year Without Summer", 1816, ["NLD", "GBR", "USA", "DEU", "CHE"]),
    ("Irish Famine", 1846, ["IRL", "GBR"]),
    ("N China Famine", 1877, ["CHN"]),
    ("Bengal Famine", 1943, ["IND", "BGD"]),
]


def _load_hyde_pop_wide() -> pd.DataFrame:
    sub = pd.read_csv(ROOT / "gbc2025_7apr_base" / "subpop_4apr2025.csv")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    sub = sub.dropna(subset=["isolink"]).copy()
    sub["iso_num"] = (sub["isolink"].astype(int) // 1000).astype(int)
    sub["iso3"] = sub["iso_num"].map(num_to_iso3)
    sub = sub.dropna(subset=["iso3"]).copy()
    year_cols = [c for c in sub.columns if c.startswith("y")]
    agg = sub.groupby("iso3", as_index=False)[year_cols].sum(min_count=1)
    return agg


def main() -> None:
    climate = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    mod = mod.merge(clim, on=["iso3", "month"], how="inner")
    mod["t_abs"] = mod["t_anom_c"] + mod["tmp_c_clim"]
    pop_wide = _load_hyde_pop_wide()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()

    rows = []
    for idx, (name, y, isos) in enumerate(EVENTS):
        ax = axes_flat[idx]
        for iso in isos:
            sub = climate[(climate["iso3"] == iso) &
                          (climate["year"].between(y - 10, y + 10))]
            anom = sub["t_c_anom_1971_2000"].values
            yrs = sub["year"].values
            ax.plot(yrs, anom, marker="o", label=iso, alpha=0.7)

            # Compute T-shock at the event year
            base = sub.loc[sub["year"].between(y - 5, y - 1), "t_c"].mean()
            post = sub.loc[sub["year"].between(y, y + 2), "t_c"].mean()
            t_shock = post - base

            # HYDE decadal pop change around the event
            ya = (y // 10) * 10
            yb = ya + 10
            ya_col, yb_col = f"y{ya}", f"y{yb}"
            yc_col = f"y{ya - 10}"
            if iso in pop_wide["iso3"].values and {ya_col, yb_col, yc_col} <= set(pop_wide.columns):
                p = pop_wide.set_index("iso3").loc[iso]
                p_before = p.get(yc_col, np.nan)
                p_at = p.get(ya_col, np.nan)
                p_after = p.get(yb_col, np.nan)
                if p_before and p_at and p_after and p_before > 0 and p_at > 0 and p_after > 0:
                    g_pre = np.log(p_at / p_before) / 10
                    g_post = np.log(p_after / p_at) / 10
                    rows.append({
                        "event": name, "year": y, "iso3": iso,
                        "t_shock_C": t_shock,
                        "pop_growth_pre10": g_pre, "pop_growth_post10": g_post,
                        "growth_drop_pp": (g_post - g_pre) * 100,
                    })

        ax.axvline(y, color="red", lw=0.8, alpha=0.6, label=f"{y}")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"{name} ({y})")
        ax.set_xlabel("Year")
        ax.set_ylabel("T anomaly (°C, vs 1971-2000)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    # 6th panel: summary table as text
    ax = axes_flat[5]
    ax.axis("off")
    out = pd.DataFrame(rows)
    ax.text(0, 1, "Demographic response\n(annualized pop growth, percentage points)",
            fontsize=10, fontweight="bold", va="top")
    ypos = 0.88
    for ev in out["event"].unique():
        sub = out[out["event"] == ev]
        med_t = sub["t_shock_C"].median()
        med_d = sub["growth_drop_pp"].median()
        ax.text(0, ypos, f"{ev}: T-shock={med_t:+.2f}°C  Δgrowth={med_d:+.2f}pp",
                fontsize=9, va="top")
        ypos -= 0.12

    plt.tight_layout()
    out_path = FIG / "fig13_documentary_validation.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\n=== Documentary cross-validation ===")
    print(out.to_string(index=False))
    out.to_parquet(DATA / "documentary_validation.parquet", index=False)


if __name__ == "__main__":
    main()
