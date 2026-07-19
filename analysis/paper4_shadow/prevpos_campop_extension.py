"""Pre-1751 prevpos extension using CamPOP 26-parish annual aggregate CBR/CDR.

Adds England 1541-1850 CamPop annual rates to the existing 1850-1900 GBR
sample from FertilityData. Runs aggregate-rate prevpos: Δlog CBR and
Δlog CDR on T anomaly, P anomaly, 5-yr T volatility, with country FE and
country-linear trend, country-clustered SEs.

This is the longest continuously-observed pre-industrial demographic panel
we can assemble from public sources: 313 years of England parish records
1538-1851 plus contemporaneous European HMD/HFD post-1850. The pre-1751
segment alone gives us 210 years of pure pre-industrial Malthusian regime,
which is what the §4.3.2 design specified.

Output: analysis/data/prevpos_campop.parquet
        analysis/figures/paper4_v2/figK_prevpos_campop.pdf
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"
ICLOUD = Path("/Users/jalonso/Library/Mobile Documents/com~apple~CloudDocs/FertilityData.xlsx")


def _load_combined_panel() -> pd.DataFrame:
    """England 1541-1899 (CamPop 26-parish aggregate, single country)."""
    cp = pd.read_csv(DATA / "wrigley_schofield" / "campop_england_annual.csv")
    cp = cp[["year", "cbr_26par", "cdr_26par"]].rename(
        columns={"cbr_26par": "CBR", "cdr_26par": "CDR"})
    cp["iso3"] = "GBR"
    cp["source"] = "CamPop"
    panel = cp.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Climate from ModE-RA + CRU
    clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    clim = clim[["iso3", "year", "t_c", "p_mm"]]
    panel = panel.merge(clim, on=["iso3", "year"], how="left")
    g = panel.groupby("iso3")
    panel["t_anom"] = panel["t_c"] - g["t_c"].transform("mean")
    panel["p_anom"] = panel["p_mm"] - g["p_mm"].transform("mean")
    panel["t_roll_sd"] = panel.groupby("iso3")["t_c"].transform(
        lambda s: s.rolling(5, center=True, min_periods=3).std())

    # log-differences (per country, with NaN where the next year is from a
    # different source - we will simply drop those rows in regression).
    panel["log_CBR"] = np.log(panel["CBR"].clip(1e-6))
    panel["log_CDR"] = np.log(panel["CDR"].clip(1e-6))
    panel["d_log_CBR"] = panel.groupby("iso3")["log_CBR"].diff()
    panel["d_log_CDR"] = panel.groupby("iso3")["log_CDR"].diff()
    return panel


def _run(df: pd.DataFrame, lhs: str, controls: list,
          year_min: int = None, year_max: int = None) -> dict:
    d = df.copy()
    if year_min is not None: d = d[d["year"] >= year_min]
    if year_max is not None: d = d[d["year"] <= year_max]
    d = d.dropna(subset=[lhs] + controls + ["iso3"])
    if len(d) < 30 or d["iso3"].nunique() < 1:
        return None
    g = d.groupby("iso3")
    for c in controls + [lhs]:
        d[c] = d[c] - g[c].transform("mean")
    d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
    X = sm.add_constant(d[controls + ["trend"]])
    n_clusters = d["iso3"].nunique()
    if n_clusters > 1:
        res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                      cov_kwds={"groups": d["iso3"]})
    else:
        res = sm.OLS(d[lhs], X).fit(cov_type="HC1")
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared),
            "n_countries": n_clusters}


def main() -> None:
    panel = _load_combined_panel()
    print(f"Combined panel: N={len(panel)}, "
          f"countries={panel['iso3'].nunique()}, "
          f"year range {panel['year'].min()}-{panel['year'].max()}")
    print(f"  by source:\n{panel.groupby('source').size()}")
    print(f"  GBR coverage: {panel[panel.iso3=='GBR'].year.min()}-{panel[panel.iso3=='GBR'].year.max()}")

    controls = ["t_anom", "p_anom", "t_roll_sd"]
    eqs = [
        ("CamPop full 1541-1850", 1541, 1850),
        ("Pre-1751 deep pre-industrial", 1541, 1750),
        ("1751-1850 transition", 1751, 1850),
    ]
    rows = []
    for sp_name, y0, y1 in eqs:
        for lhs in ["d_log_CBR", "d_log_CDR"]:
            r = _run(panel, lhs, controls, year_min=y0, year_max=y1)
            if r is None:
                continue
            stars = ("***" if r['p'].get('t_anom', 1) < 0.01 else
                     "**" if r['p'].get('t_anom', 1) < 0.05 else
                     "*" if r['p'].get('t_anom', 1) < 0.10 else "")
            print(f"\n  {sp_name} / {lhs}:")
            print(f"    N={r['n']}, countries={r['n_countries']}, R²={r['r2']:.4f}")
            for c in controls + ["trend"]:
                if c in r["params"]:
                    print(f"      {c:>14}: β = {r['params'][c]:+.5f}  "
                          f"SE = {r['bse'][c]:.5f}  p = {r['p'][c]:.3g}")
            for c in controls:
                rows.append({"subperiod": sp_name, "outcome": lhs,
                              "regressor": c,
                              "beta": r["params"].get(c, np.nan),
                              "se": r["bse"].get(c, np.nan),
                              "p": r["p"].get(c, np.nan),
                              "n": r["n"],
                              "n_countries": r["n_countries"]})
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "prevpos_campop.parquet", index=False)
    print(f"\nSaved {DATA/'prevpos_campop.parquet'}")

    # Figure: T anomaly coefficient on CDR by sub-period
    cdr = out[(out["outcome"] == "d_log_CDR") & (out["regressor"] == "t_anom")]
    if len(cdr) >= 1:
        fig, ax = plt.subplots(figsize=(8, 3))
        y = np.arange(len(cdr))
        ax.errorbar(cdr["beta"], y, xerr=1.96 * cdr["se"], fmt="o",
                    color="#202020", markerfacecolor="white",
                    markeredgewidth=1.0, ecolor="#404040", elinewidth=0.8,
                    capsize=2.5)
        for i, r in cdr.iterrows():
            stars = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
                     else "*" if r["p"] < 0.10 else "")
            ax.text(r["beta"], i + 0.18,
                    f"$N={int(r['n'])}$ {stars}",
                    ha="center", fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels(cdr["subperiod"])
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_xlabel(r"Coefficient on $T$ anomaly ($\Delta \log$ death rate, per °C)")
        ax.set_title("Pre-industrial summer mortality: pushed back to 1541",
                     loc="left", fontsize=10.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG / "figK_prevpos_campop.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figK_prevpos_campop.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"Saved {FIG/'figK_prevpos_campop.pdf'}")


if __name__ == "__main__":
    main()
