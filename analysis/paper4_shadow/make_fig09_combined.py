"""Figure 9: ModE-RA country-month climate levels and uncertainty by decade, 1421-2008.

Two-row figure for the ModE-RA + CRU climate panel the paper uses:
  (a,b) decadal temperature/precipitation anomaly levels (median + IQR across
        countries), anomalies relative to each country's 1901-1950 reference.
  (c,d) ModE-RA ensemble 1sigma country-month uncertainty by decade.

ERA5 has been removed from the project: absolute levels are anchored to the CRU
1901-1950 climatology, and the working uncertainty is ModE-RA's published ensemble
standard deviation. The panel ends at ModE-RA's 2008 horizon.

Output: analysis/figures/paper4_v2/fig09_ensemble_uncertainty.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import pandas as pd

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

ERA_MARKERS = [1500, 1700, 1850, 1950]


def _decade_quantiles(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    g = df.groupby("decade")[value_col]
    return g.agg(
        med=lambda s: s.median(),
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()


def _build_levels_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ModE-RA + CRU country-year T/P anomalies vs each country's 1901-1950 mean,
    aggregated to decade median/IQR. 1421-2008, no ERA5."""
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    cru = pd.read_parquet(DATA / "cru_country_monthly_1901_1950.parquet")
    clim = cru.groupby(["iso3", "month"], as_index=False).agg(
        t_clim=("tmp_c", "mean"), p_clim=("pre_mm", "mean"))
    mod = mod.merge(clim, on=["iso3", "month"], how="left")
    mod["t_abs"] = mod["t_anom_c"] + mod["t_clim"]
    mod["p_abs"] = mod["p_anom_mm"] + mod["p_clim"]
    yr = mod.groupby(["iso3", "year"], as_index=False).agg(
        t=("t_abs", "mean"), p=("p_abs", "sum"))
    ref = (yr[(yr["year"] >= 1901) & (yr["year"] <= 1950)]
           .groupby("iso3", as_index=False)
           .agg(t_ref=("t", "mean"), p_ref=("p", "mean")))
    yr = yr.merge(ref, on="iso3", how="left")
    yr["t_anom"] = yr["t"] - yr["t_ref"]
    yr["p_anom"] = yr["p"] - yr["p_ref"]
    yr["decade"] = (yr["year"] // 10) * 10

    def _agg(col):
        return yr.groupby("decade")[col].agg(
            med="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
        ).reset_index()

    return _agg("t_anom"), _agg("p_anom")


def _levels_panel(ax, d, ylabel, title):
    d = d.sort_values("decade")
    ax.fill_between(d["decade"], d["q25"], d["q75"], color="#B0B0B0", alpha=0.55,
                    linewidth=0, label="IQR across countries")
    ax.plot(d["decade"], d["med"], color="#202020", linewidth=1.4,
            label=r"ModE-RA $+$ CRU median")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.axhline(0, color="#404040", linewidth=0.4)
    ax.set_xlim(1421, 2010)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    for x in ERA_MARKERS:
        ax.axvline(x, color="#808080", linewidth=0.5, linestyle=":")


def _unc_panel(ax, d, ylabel, title, xlabel="Year"):
    d = d.sort_values("decade")
    ax.fill_between(d["decade"], d["q25"], d["q75"], color="#B0B0B0", alpha=0.55,
                    linewidth=0, label=r"ModE-RA $1\sigma$ (IQR across countries)")
    ax.plot(d["decade"], d["med"], color="#202020", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlim(1421, 2010)
    ax.set_ylim(0, float(d["q75"].max()) * 1.18)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    for x in ERA_MARKERS:
        ax.axvline(x, color="#808080", linewidth=0.5, linestyle=":")


def main() -> None:
    print("Building Figure 9: ModE-RA climate levels + uncertainty, 1421-2008 (no ERA5)")
    modera = pd.read_parquet(DATA / "modera_country_uncertainty.parquet")
    t_unc = _decade_quantiles(modera[["year", "t_std"]], "t_std")
    p_unc = _decade_quantiles(modera[["year", "p_std"]], "p_std")
    t_lvl, p_lvl = _build_levels_panel()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col")
    _levels_panel(axes[0, 0], t_lvl, r"$T$ anomaly vs 1901--1950 (°C)",
                  r"(a) Temperature anomaly by decade")
    _levels_panel(axes[0, 1], p_lvl, r"$P$ anomaly vs 1901--1950 (mm)",
                  r"(b) Precipitation anomaly by decade")
    _unc_panel(axes[1, 0], t_unc, r"$1\sigma$ in temperature (°C)",
               r"(c) Temperature uncertainty by decade")
    _unc_panel(axes[1, 1], p_unc, r"$1\sigma$ in precipitation (mm/month)",
               r"(d) Precipitation uncertainty by decade")

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig09_ensemble_uncertainty.pdf")
    fig.savefig(FIG / "fig09_ensemble_uncertainty.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG / 'fig09_ensemble_uncertainty.pdf'}")


if __name__ == "__main__":
    main()
