"""Decompose the Malthusian climate response into preventive (fertility) and
positive (mortality) checks using annual European HMD/HFD data.

Data source: FertilityData.xlsx (iCloud Documents), MAIN PANEL sheet. The
panel has annual age-specific mortality (m0 infant, m5, m10, m15) plus a
fertility rate (`fert`, equivalent to CBR / 1000), real income, and an
epidemic dummy for 10 European countries 1751--2022. Pre-1900 climate columns
are empty in the FertilityData panel; we supply ModE-RA-based temperature and
precipitation anomalies from our country panel.

Decomposition. The Malthusian regression in Section~\ref{sec:stage2}
recovers a single coefficient on log population growth. Population growth is
the difference between births and deaths, so the climate-on-pop-growth
coefficient is the sum of a preventive-check term (fertility response) and a
positive-check term (mortality response, taken with opposite sign):

    γ^pop_T  =  γ^fert_T  -  γ^mort_T.

We estimate γ^fert_T and γ^mort_T separately on the FertilityData annual
panel, with country fixed effects, year-trend controls, and country-clustered
SEs. The decomposition speaks directly to the Malthusian mechanism that the
headline regression captures only in reduced form.

Outputs:
    analysis/data/prevpos_panel.parquet  -- merged panel
    analysis/data/prevpos_results.parquet -- regression coefficients
    analysis/figures/paper4_v2/figK_prevpos.{pdf,png}
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
ICLOUD_FERTILITY = Path(
    "/Users/jalonso/Library/Mobile Documents/com~apple~CloudDocs/FertilityData.xlsx"
)


def _build_panel(year_max: int = 1900) -> pd.DataFrame:
    fd = pd.read_excel(ICLOUD_FERTILITY, sheet_name="MAIN PANEL")
    fd = fd.rename(columns={"code": "iso3"})
    fd = fd[fd["year"] <= year_max].copy()
    # Merge in our ModE-RA climate (interval-mean T and within-year volatility)
    clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    clim = clim[["iso3", "year", "t_c", "p_mm"]]
    panel = fd.merge(clim, on=["iso3", "year"], how="inner")
    # Country-level demeaned climate (interpret as within-country anomaly)
    g = panel.groupby("iso3")
    panel["t_anom"] = panel["t_c"] - g["t_c"].transform("mean")
    panel["p_anom"] = panel["p_mm"] - g["p_mm"].transform("mean")
    # Rolling 5-yr T volatility per country
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel["t_roll_sd"] = (panel.groupby("iso3")["t_c"]
                                 .transform(lambda s: s.rolling(window=5,
                                                                  min_periods=3,
                                                                  center=True).std()))
    return panel


def _logdiff(panel: pd.DataFrame, var: str) -> pd.DataFrame:
    panel = panel.sort_values(["iso3", "year"]).copy()
    panel[f"log_{var}"] = np.log(panel[var].clip(1e-6))
    panel[f"d_log_{var}"] = panel.groupby("iso3")[f"log_{var}"].diff()
    return panel


def _run(df: pd.DataFrame, lhs: str, controls: list[str]) -> dict:
    d = df.dropna(subset=[lhs] + controls).copy()
    g = d.groupby("iso3")
    for c in controls + [lhs]:
        d[c] = d[c] - g[c].transform("mean")
    # Optional linear country-trend control: subtract year × country mean
    d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
    X = sm.add_constant(d[controls + ["trend"]])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}


def main() -> None:
    print("=== Preventive (fertility) vs positive (mortality) check ===\n")
    panel = _build_panel()
    print(f"Panel: N={len(panel):,}, "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")
    print(f"  obs by country:")
    print(panel.groupby("iso3").size().sort_values(ascending=False).to_string())

    panel = _logdiff(panel, "fert")
    for m in ["m0", "m5", "m10"]:
        panel = _logdiff(panel, m)
    panel.to_parquet(DATA / "prevpos_panel.parquet", index=False)

    controls = ["t_anom", "p_anom", "t_roll_sd"]
    eqs = [
        ("Δ log fertility (preventive)",        "d_log_fert"),
        ("Δ log infant mortality m0 (positive)", "d_log_m0"),
        ("Δ log child mortality m5 (positive)",  "d_log_m5"),
        ("Δ log child mortality m10 (positive)", "d_log_m10"),
    ]
    rows = []
    for label, lhs in eqs:
        r = _run(panel, lhs, controls)
        print(f"\n  {label}: N={r['n']}, R²={r['r2']:.4f}")
        for c in controls + ["trend"]:
            if c in r["params"]:
                stars = ("***" if r['p'][c] < 0.01 else "**" if r['p'][c] < 0.05
                         else "*" if r['p'][c] < 0.10 else "")
                print(f"    {c:>14}: β = {r['params'][c]:+.5f}  "
                      f"SE = {r['bse'][c]:.5f}  p = {r['p'][c]:.3g} {stars}")
        for c in controls:
            rows.append({"eq": label, "regressor": c,
                          "beta": r['params'].get(c, np.nan),
                          "se": r['bse'].get(c, np.nan),
                          "p": r['p'].get(c, np.nan),
                          "n": r['n']})
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "prevpos_results.parquet", index=False)

    # --- Figure: T anomaly and T volatility coefficients across equations ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, regr, title in [(axes[0], "t_anom", r"(a) Coefficient on $T$ anomaly"),
                              (axes[1], "t_roll_sd", r"(b) Coefficient on 5-yr $T$ volatility")]:
        sub = out[out["regressor"] == regr].reset_index(drop=True)
        y = np.arange(len(sub))
        ax.errorbar(sub["beta"], y, xerr=1.96 * sub["se"],
                    fmt="o", markerfacecolor="white", markeredgewidth=1.0,
                    color="#202020", ecolor="#404040", elinewidth=0.8, capsize=2.5)
        for i, r in sub.iterrows():
            s = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
                 else "*" if r["p"] < 0.10 else "")
            ax.text(r["beta"], i + 0.18, f"$N={int(r['n'])}$  {s}",
                    ha="center", fontsize=8.5)
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_yticks(y); ax.set_yticklabels(sub["eq"].str.replace(" (positive)", "")
                                                       .str.replace(" (preventive)", ""))
        ax.set_xlabel("Coefficient (per °C)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Preventive vs positive Malthusian checks, European annual panel 1751--1900",
                 y=1.04, x=0.04, ha="left", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / "figK_prevpos.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figK_prevpos.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {FIG/'figK_prevpos.pdf'}")


if __name__ == "__main__":
    main()
