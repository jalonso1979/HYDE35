"""Updated Figure 9: T and P uncertainty by decade, 1421--2025.

Builds a two-panel figure showing the country-monthly uncertainty of the
combined ModE-RA + ERA5 climate panel that the rest of the paper uses.
Three pieces of information are layered into each panel:

  - ModE-RA ensemble 1sigma (the paleo-reanalysis intrinsic uncertainty),
    available 1421--2008.  Plotted as the grey band + black median line.
  - ERA5 vs ModE-RA monthly RMSE in the 1950--2008 overlap window: a
    cross-validation diagnostic, plotted as a dashed red line.
  - Post-2008 ERA5-only noise floor: ERA5 country-monthly does not have
    its own ensemble in our setup, so we use the ERA5 official monthly
    instrumental uncertainty estimate (~0.1 degree C for T, ~10% for P)
    plotted as a thin grey horizontal stub.

The merged-panel uncertainty that propagates into the regressions is the
lower envelope of the three: pre-1950 = ModE-RA ensemble 1sigma, 1950--2008
= min(ModE-RA, ERA5--ModERA RMSE), post-2008 = ERA5 noise floor.

Outputs:
    analysis/figures/paper4_v2/fig09_ensemble_uncertainty.{pdf,png}
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

# ERA5 monthly noise floor (conservative literature values for area-averaged
# country-month outputs after our bias correction).
ERA5_T_NOISE = 0.10   # °C
ERA5_P_NOISE = 5.0    # mm/month, conservative for country-monthly aggregation


def _decade_quantiles(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    g = df.groupby("decade")[value_col]
    out = g.agg(
        med=lambda s: s.median(),
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        n=("size"),
    ).reset_index()
    return out


def _build_era5_modera_rmse() -> pd.DataFrame:
    """Decade RMSE of ERA5 minus ModE-RA in the overlap window, after country-
    mean bias correction (the same correction applied in panel construction).

    We use absolute T and P. ModE-RA is anomaly + CRU 1901-1950 climatology;
    ERA5 is absolute. The country-mean bias estimated over 1950-2008 is
    removed first, then we compute the residual month-by-month RMSE.
    """
    era5 = pd.read_parquet(DATA / "era5_country_monthly.parquet")
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    era5 = era5.rename(columns={"t2m_c": "t_c_era", "tp_mm": "p_mm_era"})

    # ModE-RA absolute T = anomaly + CRU 1901-1950 monthly climatology
    cru = pd.read_parquet(DATA / "cru_country_monthly_1901_1950.parquet")
    clim = cru.groupby(["iso3", "month"], as_index=False).agg(
        t_clim=("tmp_c", "mean"), p_clim=("pre_mm", "mean"))
    mod = mod.merge(clim, on=["iso3", "month"], how="left")
    mod["t_c_mod"] = mod["t_anom_c"] + mod["t_clim"]
    mod["p_mm_mod"] = mod["p_anom_mm"] + mod["p_clim"]

    # Restrict to the common 1950-2008 overlap window FIRST
    overlap = era5.merge(mod[["iso3", "year", "month", "t_c_mod", "p_mm_mod"]],
                          on=["iso3", "year", "month"], how="inner")
    overlap = overlap[(overlap["year"] >= 1950) & (overlap["year"] <= 2008)].copy()

    # Country-mean bias over overlap
    bias_t = overlap.groupby("iso3").apply(
        lambda d: (d["t_c_era"] - d["t_c_mod"]).mean()).rename("bias_t").reset_index()
    bias_p = overlap.groupby("iso3").apply(
        lambda d: (d["p_mm_era"] - d["p_mm_mod"]).mean()).rename("bias_p").reset_index()
    overlap = overlap.merge(bias_t, on="iso3").merge(bias_p, on="iso3")

    # Residual after bias correction (mean over overlap is zero by construction)
    overlap["t_resid"] = (overlap["t_c_era"] - overlap["bias_t"]) - overlap["t_c_mod"]
    overlap["p_resid"] = (overlap["p_mm_era"] - overlap["bias_p"]) - overlap["p_mm_mod"]

    overlap["decade"] = (overlap["year"] // 10) * 10
    rmse = overlap.groupby("decade").agg(
        t_rmse=("t_resid", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        p_rmse=("p_resid", lambda s: float(np.sqrt(np.mean(s ** 2)))),
    ).reset_index()
    return rmse


def _build_levels_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country-year T and P levels by decade, expressed as anomalies from each
    country's mean over the common 1901-1950 reference period so that polar
    and tropical countries can share an axis. ModE-RA (1421-1949) is spliced
    with ERA5 (1950-2025).
    Returns: t_levels, p_levels with columns [decade, med, q25, q75, source]
    """
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    cru = pd.read_parquet(DATA / "cru_country_monthly_1901_1950.parquet")
    clim = cru.groupby(["iso3", "month"], as_index=False).agg(
        t_clim=("tmp_c", "mean"), p_clim=("pre_mm", "mean"))
    mod = mod.merge(clim, on=["iso3", "month"], how="left")
    mod["t_abs"] = mod["t_anom_c"] + mod["t_clim"]
    mod["p_abs"] = mod["p_anom_mm"] + mod["p_clim"]
    mod_year = mod.groupby(["iso3", "year"], as_index=False).agg(
        t=("t_abs", "mean"), p=("p_abs", "sum"))

    era5 = pd.read_parquet(DATA / "era5_country_monthly.parquet")
    era_year = era5.groupby(["iso3", "year"], as_index=False).agg(
        t=("t2m_c", "mean"), p=("tp_mm", "sum"))

    # Common 1901-1950 reference (ModE-RA's CRU climatology window) for both
    # products. ERA5 starts at 1950, so we use the 1950-1980 period as its
    # reference (a 30-year window that overlaps with the late-CRU period).
    mod_ref = (mod_year[(mod_year["year"] >= 1901) & (mod_year["year"] <= 1950)]
               .groupby("iso3", as_index=False).agg(
                   t_ref=("t", "mean"), p_ref=("p", "mean")))
    mod_year = mod_year.merge(mod_ref, on="iso3", how="left")
    mod_year["t_anom"] = mod_year["t"] - mod_year["t_ref"]
    mod_year["p_anom"] = mod_year["p"] - mod_year["p_ref"]

    era_ref = (era_year[(era_year["year"] >= 1950) & (era_year["year"] <= 1980)]
               .groupby("iso3", as_index=False).agg(
                   t_ref=("t", "mean"), p_ref=("p", "mean")))
    era_year = era_year.merge(era_ref, on="iso3", how="left")
    era_year["t_anom"] = era_year["t"] - era_year["t_ref"]
    era_year["p_anom"] = era_year["p"] - era_year["p_ref"]

    # Re-baseline ERA5 anomalies onto the ModE-RA reference: compute the
    # difference between the two products' reference means in the 1950-1980
    # window and subtract it from ERA5 anomalies so the two series splice
    # smoothly at 1950.
    splice = (mod_year[(mod_year["year"] >= 1950) & (mod_year["year"] <= 1980)]
              .groupby("iso3", as_index=False).agg(
                  t_splice=("t_anom", "mean"), p_splice=("p_anom", "mean")))
    era_year = era_year.merge(splice, on="iso3", how="left")
    era_year["t_anom"] = era_year["t_anom"] + era_year["t_splice"].fillna(0.0)
    era_year["p_anom"] = era_year["p_anom"] + era_year["p_splice"].fillna(0.0)

    mod_sub = mod_year[mod_year["year"] < 1950][["iso3", "year", "t_anom", "p_anom"]].copy()
    mod_sub["source"] = "ModE-RA"
    era_sub = era_year[era_year["year"] >= 1950][["iso3", "year", "t_anom", "p_anom"]].copy()
    era_sub["source"] = "ERA5"
    annual = pd.concat([mod_sub, era_sub], ignore_index=True)
    annual["decade"] = (annual["year"] // 10) * 10

    def _agg(col):
        return annual.groupby(["decade", "source"])[col].agg(
            med="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
        ).reset_index()

    return _agg("t_anom"), _agg("p_anom")


def main() -> None:
    print("Building updated Figure 9: ModE-RA + ERA5 levels and uncertainty, 1421--2025")
    modera = pd.read_parquet(DATA / "modera_country_uncertainty.parquet")
    print(f"  ModE-RA uncertainty: N={len(modera):,}, "
          f"years {modera['year'].min()}-{modera['year'].max()}")

    t_unc = _decade_quantiles(modera[["year", "t_std"]], "t_std")
    p_unc = _decade_quantiles(modera[["year", "p_std"]], "p_std")
    print(f"  decade aggregates: {len(t_unc)} decades")

    rmse = _build_era5_modera_rmse()
    print(f"  ERA5-ModE-RA RMSE (1950--2008 overlap):")
    print(rmse.to_string(index=False))

    t_lvl, p_lvl = _build_levels_panel()
    print(f"\n  Decadal levels panel: T decades = {t_lvl['decade'].nunique()}, "
          f"P decades = {p_lvl['decade'].nunique()}")

    # ── Figure: 2x2 grid (levels top, uncertainty bottom) ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col")

    # ----- (a) TEMPERATURE LEVELS (top-left) -----
    ax = axes[0, 0]
    mod_t = t_lvl[t_lvl["source"] == "ModE-RA"].sort_values("decade")
    era_t = t_lvl[t_lvl["source"] == "ERA5"].sort_values("decade")
    ax.fill_between(mod_t["decade"], mod_t["q25"], mod_t["q75"],
                     color="#B0B0B0", alpha=0.55, linewidth=0,
                     label=r"IQR across countries")
    ax.plot(mod_t["decade"], mod_t["med"], color="#202020", linewidth=1.4,
             label=r"ModE-RA $+$ CRU median")
    ax.fill_between(era_t["decade"], era_t["q25"], era_t["q75"],
                     color="#9ECAE1", alpha=0.55, linewidth=0)
    ax.plot(era_t["decade"], era_t["med"], color="#0072B2", linewidth=1.4,
             label=r"ERA5 median")
    ax.set_ylabel(r"$T$ anomaly vs 1901--1950 (°C)")
    ax.set_title(r"(a) Temperature anomaly by decade", loc="left", fontsize=10.5)
    ax.axhline(0, color="#404040", linewidth=0.4, linestyle="-")
    ax.set_xlim(1421, 2030)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    for x in [1500, 1700, 1850, 1950]:
        ax.axvline(x, color="#808080", linewidth=0.5, linestyle=":")

    # ----- (b) PRECIPITATION LEVELS (top-right) -----
    ax = axes[0, 1]
    mod_p = p_lvl[p_lvl["source"] == "ModE-RA"].sort_values("decade")
    era_p = p_lvl[p_lvl["source"] == "ERA5"].sort_values("decade")
    ax.fill_between(mod_p["decade"], mod_p["q25"], mod_p["q75"],
                     color="#B0B0B0", alpha=0.55, linewidth=0,
                     label=r"IQR across countries")
    ax.plot(mod_p["decade"], mod_p["med"], color="#202020", linewidth=1.4,
             label=r"ModE-RA $+$ CRU median")
    ax.fill_between(era_p["decade"], era_p["q25"], era_p["q75"],
                     color="#9ECAE1", alpha=0.55, linewidth=0)
    ax.plot(era_p["decade"], era_p["med"], color="#0072B2", linewidth=1.4,
             label=r"ERA5 median")
    ax.set_ylabel(r"$P$ anomaly vs 1901--1950 (mm)")
    ax.set_title(r"(b) Precipitation anomaly by decade", loc="left", fontsize=10.5)
    ax.axhline(0, color="#404040", linewidth=0.4, linestyle="-")
    ax.set_xlim(1421, 2030)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    for x in [1500, 1700, 1850, 1950]:
        ax.axvline(x, color="#808080", linewidth=0.5, linestyle=":")

    # ----- (c) TEMPERATURE UNCERTAINTY (bottom-left) -----
    ax = axes[1, 0]
    # ModE-RA ensemble band 1421-2008
    pre1950 = t_unc[t_unc["decade"] < 1950]
    overlap = t_unc[(t_unc["decade"] >= 1950) & (t_unc["decade"] <= 2000)]
    post2008 = t_unc[t_unc["decade"] >= 2010] if (t_unc["decade"] >= 2010).any() else None

    ax.fill_between(pre1950["decade"], pre1950["q25"], pre1950["q75"],
                     color="#B0B0B0", alpha=0.55, linewidth=0,
                     label=r"ModE-RA $1\sigma$ (IQR across countries)")
    ax.plot(pre1950["decade"], pre1950["med"],
             color="#202020", linewidth=1.4)
    ax.fill_between(overlap["decade"], overlap["q25"], overlap["q75"],
                     color="#B0B0B0", alpha=0.55, linewidth=0)
    ax.plot(overlap["decade"], overlap["med"], color="#202020", linewidth=1.4)

    # ERA5 vs ModE-RA RMSE in overlap
    ax.plot(rmse["decade"], rmse["t_rmse"], color="#A02020",
             linewidth=1.4, linestyle="--", marker="o", markersize=3.0,
             markerfacecolor="white", markeredgewidth=1.0,
             label=r"ERA5$-$ModE-RA RMSE (overlap)")

    # ERA5 noise floor 2010-2025
    era5_decs = [2010, 2020]
    ax.plot(era5_decs, [ERA5_T_NOISE, ERA5_T_NOISE], color="#0072B2",
             linewidth=2.0, label=r"ERA5 instrumental floor")

    ax.set_xlabel("Year")
    ax.set_ylabel(r"$1\sigma$ in temperature (°C)")
    ax.set_title(r"(c) Temperature uncertainty by decade", loc="left", fontsize=10.5)
    ax.set_xlim(1421, 2030)
    ax.set_ylim(0, 1.7)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)

    # Vertical era markers
    for x in [1500, 1700, 1850, 1950]:
        ax.axvline(x, color="#808080", linewidth=0.5, linestyle=":")
    for x, txt in [(1460, "pre-instr."), (1590, "early\ninstr."),
                    (1775, "dense\ninstr."), (1900, "modern"),
                    (2000, "ERA5\nera")]:
        ax.text(x, 1.55, txt, ha="center", fontsize=7.5, color="#404040")

    # ----- (d) PRECIPITATION UNCERTAINTY (bottom-right) -----
    ax = axes[1, 1]
    pre1950 = p_unc[p_unc["decade"] < 1950]
    overlap = p_unc[(p_unc["decade"] >= 1950) & (p_unc["decade"] <= 2000)]

    ax.fill_between(pre1950["decade"], pre1950["q25"], pre1950["q75"],
                     color="#B0B0B0", alpha=0.55, linewidth=0,
                     label=r"ModE-RA $1\sigma$ (IQR across countries)")
    ax.plot(pre1950["decade"], pre1950["med"],
             color="#202020", linewidth=1.4)
    ax.fill_between(overlap["decade"], overlap["q25"], overlap["q75"],
                     color="#B0B0B0", alpha=0.55, linewidth=0)
    ax.plot(overlap["decade"], overlap["med"], color="#202020", linewidth=1.4)

    ax.plot(rmse["decade"], rmse["p_rmse"], color="#A02020",
             linewidth=1.4, linestyle="--", marker="o", markersize=3.0,
             markerfacecolor="white", markeredgewidth=1.0,
             label=r"ERA5$-$ModE-RA RMSE (overlap)")

    ax.plot([2010, 2020], [ERA5_P_NOISE, ERA5_P_NOISE], color="#0072B2",
             linewidth=2.0, label=r"ERA5 instrumental floor")

    ax.set_xlabel("Year")
    ax.set_ylabel(r"$1\sigma$ in precipitation (mm/month)")
    ax.set_title(r"(d) Precipitation uncertainty by decade", loc="left", fontsize=10.5)
    ax.set_xlim(1421, 2030)
    pmax = max(float(pre1950["q75"].max()), float(overlap["q75"].max() if len(overlap) else 0),
               float(rmse["p_rmse"].max()))
    ax.set_ylim(0, pmax * 1.18)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    for x in [1500, 1700, 1850, 1950]:
        ax.axvline(x, color="#808080", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    out = FIG / "fig09_ensemble_uncertainty"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {out}.pdf / .png")

    # Persist the underlying numbers for the paper text
    summary = []
    for era_label, lo, hi in [("1421-1500", 1420, 1500),
                                ("1500-1700", 1500, 1700),
                                ("1700-1850", 1700, 1850),
                                ("1850-1949", 1850, 1950),
                                ("1950-2008 (overlap)", 1950, 2010),
                                ("2010-2025 (ERA5 only)", 2010, 2030)]:
        sub = modera[(modera["year"] >= lo) & (modera["year"] < hi)]
        if len(sub) == 0:
            continue
        summary.append({"era": era_label,
                         "t_std_med": float(sub["t_std"].median()),
                         "p_std_med": float(sub["p_std"].median())})
    s = pd.DataFrame(summary)
    print("\n  Era-averaged country-month medians:")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    s.to_csv(DATA / "ensemble_uncertainty_by_era.csv", index=False)


if __name__ == "__main__":
    main()
