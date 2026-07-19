"""Section 4.3.4 — Modern-era prevpos: did the warm-year mortality channel
flip sign as sanitation improved?

Uses FertilityData MAIN PANEL 1900-2022, ERA5 climate post-1950,
ModE-RA/CRU climate 1900-1949. Three sub-periods: 1900-1950 vs 1950-2022
vs full modern, testing for the predicted sign-flip.
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


def _build_panel() -> pd.DataFrame:
    fd = pd.read_excel(ICLOUD, sheet_name="MAIN PANEL")
    fd = fd.rename(columns={"code": "iso3"})
    fd = fd[fd["year"] >= 1900].copy()

    # ModE-RA + CRU absolute T/P for the whole modern window (1900-2008).
    # ERA5 removed: the modern climate is now the same ModE-RA+CRU panel used
    # throughout the paper, which ends at ModE-RA's 2008 horizon.
    mod = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    clim = mod[["iso3", "year", "t_c", "p_mm"]]
    clim = clim[clim["year"] >= 1900]

    panel = fd.merge(clim, on=["iso3", "year"], how="inner")
    g = panel.groupby("iso3")
    panel["t_anom"] = panel["t_c"] - g["t_c"].transform("mean")
    panel["p_anom"] = panel["p_mm"] - g["p_mm"].transform("mean")
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel["t_roll_sd"] = panel.groupby("iso3")["t_c"].transform(
        lambda s: s.rolling(5, center=True, min_periods=3).std())

    for v in ["fert", "m0", "m5", "m10"]:
        panel[f"log_{v}"] = np.log(panel[v].clip(1e-6))
        panel[f"d_log_{v}"] = panel.groupby("iso3")[f"log_{v}"].diff()
    return panel


def _run(df: pd.DataFrame, lhs: str, controls: list) -> dict:
    d = df.dropna(subset=[lhs] + controls).copy()
    if len(d) < 30 or d["iso3"].nunique() < 2:
        return None
    g = d.groupby("iso3")
    for c in controls + [lhs]:
        d[c] = d[c] - g[c].transform("mean")
    d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
    X = sm.add_constant(d[controls + ["trend"]])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}


def main() -> None:
    panel = _build_panel()
    print(f"Modern panel: N={len(panel)}, "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")

    controls = ["t_anom", "p_anom", "t_roll_sd"]
    outcomes = [("d_log_fert", "fertility"), ("d_log_m0", "m0"),
                ("d_log_m5", "m5"), ("d_log_m10", "m10")]
    subperiods = [
        ("1900-1950", 1900, 1950),
        ("1950-2022", 1950, 2023),
        ("1900-2022 full", 1900, 2023),
    ]
    rows = []
    for sp_name, y0, y1 in subperiods:
        sub = panel[(panel["year"] >= y0) & (panel["year"] < y1)]
        for lhs, label in outcomes:
            r = _run(sub, lhs, controls)
            if r is None:
                print(f"  {sp_name} / {label}: SKIPPED (insufficient sample)")
                continue
            stars = ("***" if r['p'].get('t_anom', 1) < 0.01 else
                     "**" if r['p'].get('t_anom', 1) < 0.05 else
                     "*" if r['p'].get('t_anom', 1) < 0.10 else "")
            print(f"  {sp_name} / {label}: N={r['n']}, "
                  f"β_T={r['params'].get('t_anom', np.nan):+.5f} "
                  f"(p={r['p'].get('t_anom', np.nan):.3g}) {stars}")
            for c in controls:
                rows.append({"subperiod": sp_name, "outcome": label,
                              "regressor": c,
                              "beta": r["params"].get(c, np.nan),
                              "se": r["bse"].get(c, np.nan),
                              "p": r["p"].get(c, np.nan), "n": r["n"]})
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "prevpos_modern.parquet", index=False)
    print(f"\nSaved {DATA/'prevpos_modern.parquet'}: {len(out)} rows")

    # Figure: T anomaly coefficient on m0 by sub-period
    m0 = out[(out["outcome"] == "m0") & (out["regressor"] == "t_anom")]
    if len(m0) >= 1:
        fig, ax = plt.subplots(figsize=(7, 3))
        y = np.arange(len(m0))
        ax.errorbar(m0["beta"], y, xerr=1.96 * m0["se"], fmt="o",
                    color="#202020", markerfacecolor="white",
                    markeredgewidth=1.0, ecolor="#404040", elinewidth=0.8,
                    capsize=2.5)
        for i, r in m0.iterrows():
            stars = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
                     else "*" if r["p"] < 0.10 else "")
            ax.text(r["beta"], i + 0.18,
                    f"$N={int(r['n'])}$  {stars}",
                    ha="center", fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels(m0["subperiod"])
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_xlabel(r"Coefficient on $T$ anomaly ($\Delta \log m_0$)")
        ax.set_title("Modern-era reversal of summer-mortality channel",
                     loc="left", fontsize=10.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG / "figK_prevpos_modern.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figK_prevpos_modern.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"Saved {FIG/'figK_prevpos_modern.pdf'}")


if __name__ == "__main__":
    main()
