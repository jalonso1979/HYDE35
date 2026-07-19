"""Joint specification and Justinianic-focused analysis.

(A) Joint specification: lead-Z on pandemic intensity, volcanic forcing,
    war intensity, and climate controls. Disentangles the channels.

(B) Justinianic-focused IRF: this was the only family-specific IRF that
    passed parallel trends in script 01. Re-run with longer leads/lags
    and richer controls.

(C) Cumulative volcanic forcing decade-by-decade — analogous to our
    Sigl-style continuous forcing in the Long Shadow paper.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "paper4_shadow"))
from figstyle import set_style

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "roman_jevons"
FIG = OUT / "figures"


def _load() -> pd.DataFrame:
    return pd.read_csv(DATA / "roman_v2_panel.csv").sort_values("year_ce")


def joint_static() -> None:
    df = _load()
    df = df.dropna(subset=["lead_z", "volcanic", "pandemic_v2_intensity",
                            "war_v2_intensity", "temp"])
    X = sm.add_constant(df[["pandemic_v2_intensity", "volcanic",
                              "war_v2_intensity", "temp"]])
    y = df["lead_z"]
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    print("=== (A) Joint static specification: lead-Z on all channels ===")
    print(f"N = {int(r.nobs)}, R² = {r.rsquared:.3f}")
    for v in ["pandemic_v2_intensity", "volcanic", "war_v2_intensity", "temp"]:
        print(f"  {v:<30s}: β = {r.params[v]:>+8.4f}  "
              f"SE = {r.bse[v]:.4f}  p = {r.pvalues[v]:.4g}")
    pd.DataFrame({"variable": r.params.index, "beta": r.params.values,
                   "se": r.bse.values, "p": r.pvalues.values}).to_csv(
        OUT / "joint_static.csv", index=False)


def justinianic_irf() -> None:
    """Longer-window Justinianic IRF, with controls for volcanic and war."""
    df = _load()
    df = df.dropna(subset=["lead_z", "pandemic_v2_family_justinianic",
                            "volcanic", "war_v2_intensity"])
    leads_lags = list(range(-10, 21))
    p = df.sort_values("year_ce").copy()
    cols = []
    for h in leads_lags:
        col = f"jus_h{h:+d}"
        p[col] = p["pandemic_v2_family_justinianic"].shift(-h)
        cols.append(col)
    p = p.dropna(subset=["lead_z"] + cols + ["volcanic", "war_v2_intensity"])
    X = sm.add_constant(p[cols + ["volcanic", "war_v2_intensity"]])
    y = p["lead_z"]
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    rows = []
    for h in leads_lags:
        col = f"jus_h{h:+d}"
        rows.append({"lag": h, "beta": r.params[col],
                      "se": r.bse[col], "p": r.pvalues[col]})
    irf = pd.DataFrame(rows)
    irf["ci_lo"] = irf["beta"] - 1.96 * irf["se"]
    irf["ci_hi"] = irf["beta"] + 1.96 * irf["se"]
    irf.to_csv(OUT / "irf_justinianic_long.csv", index=False)

    leads = [c for h, c in zip(leads_lags, cols) if h < 0]
    ft = r.f_test([f"{c} = 0" for c in leads])
    print(f"\n=== (B) Justinianic IRF, -10 to +20, controls: volcanic + war ===")
    print(f"Parallel-trends F (h<0): F = {float(ft.fvalue):.3f}, p = {float(ft.pvalue):.4g}")
    print(f"N = {int(r.nobs)}")
    print(irf.round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.errorbar(irf["lag"], irf["beta"], yerr=1.96 * irf["se"],
                fmt="o-", color="#202020", markerfacecolor="white",
                markeredgewidth=1, ecolor="#606060", linewidth=1.0, capsize=2)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.axvline(0, color="#606060", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Year relative to Justinianic onset")
    ax.set_ylabel(r"$\hat\beta$ on Roman lead-Z")
    ax.set_title(r"Justinianic plague IRF, controls: volcanic $+$ war",
                  loc="left", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG / "fig_justinianic_long_irf.pdf")
    plt.close(fig)


def decade_volcanic_irf() -> None:
    """Sigl-style continuous-forcing: cumulative VSSI in 10-year bins."""
    df = _load()
    # Aggregate to decades
    df["decade"] = (df["year_ce"] // 10) * 10
    dec = df.groupby("decade", as_index=False).agg(
        lead_z=("lead_z", "mean"),
        volcanic_sum=("volcanic", "sum"),
        pandemic_sum=("pandemic_v2_intensity", "sum"),
        war_sum=("war_v2_intensity", "sum"),
    )
    dec = dec.dropna(subset=["lead_z"])
    # Same lead/lag approach in decade units
    leads_lags = list(range(-3, 4))
    for h in leads_lags:
        dec[f"volc_h{h:+d}"] = dec["volcanic_sum"].shift(-h)
    dec = dec.dropna()
    X = sm.add_constant(dec[[f"volc_h{h:+d}" for h in leads_lags] +
                              ["pandemic_sum", "war_sum"]])
    y = dec["lead_z"]
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    print(f"\n=== (C) Decade-level volcanic IRF (controls: pandemic, war) ===")
    print(f"N decades = {int(r.nobs)}")
    rows = []
    for h in leads_lags:
        col = f"volc_h{h:+d}"
        print(f"  volc(decade {h:+d}): β = {r.params[col]:+.5f}  "
              f"SE = {r.bse[col]:.5f}  p = {r.pvalues[col]:.4g}")
        rows.append({"lag": h, "beta": r.params[col],
                      "se": r.bse[col], "p": r.pvalues[col]})
    leads = [f"volc_h{h:+d}" for h in leads_lags if h < 0]
    ft = r.f_test([f"{c} = 0" for c in leads])
    print(f"  Parallel-trends F: F = {float(ft.fvalue):.3f}, "
          f"p = {float(ft.pvalue):.4g}")
    pd.DataFrame(rows).to_csv(OUT / "irf_volcanic_decade.csv", index=False)


def main() -> None:
    joint_static()
    justinianic_irf()
    decade_volcanic_irf()
    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
