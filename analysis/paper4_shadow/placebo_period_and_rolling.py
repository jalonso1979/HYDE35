"""Two quick robustness checks.

(1) PLACEBO PERIOD for the long-shadow regression.
    The headline result uses sigma_v(1421-1750). What if we replace it with
    sigma_v computed from a different historical window? If the 1421-1750
    window is doing real work, the post-industrial 1850-1900 window should
    weaken or eliminate the effect.

(2) ROLLING-WINDOW MALTHUSIAN BETA.
    The subperiod-stability finding (Section on Stage 2) uses three discrete
    subperiods. A rolling 100-year window shows the beta(tau, t) trajectory
    continuously, making the "Malthus is dead by 1900" finding more visible.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette, LINESTYLES

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def placebo_period() -> pd.DataFrame:
    """Long-shadow regression with sigma_v computed over different windows."""
    ext = pd.read_parquet(DATA / "hyde_era5_extended_panel.parquet")
    early = ext[ext["year"].between(1950, 1960)].groupby("iso3", as_index=False).agg(
        pop_e=("pop", "mean"))
    late = ext[ext["year"].between(2015, 2025)].groupby("iso3", as_index=False).agg(
        pop_l=("pop", "mean"))
    out = early.merge(late, on="iso3")
    out = out[(out["pop_e"] > 0) & (out["pop_l"] > 0)]
    out["log_pop_growth"] = np.log(out["pop_l"] / out["pop_e"])

    seas = pd.read_parquet(DATA / "country_seasonality_1421_2008.parquet")

    windows = [
        ("1421-1500", 1421, 1500),
        ("1500-1600", 1500, 1600),
        ("1600-1700", 1600, 1700),
        ("1700-1750", 1700, 1750),
        ("1421-1750 (HEADLINE)", 1421, 1750),
        ("1750-1850", 1750, 1850),
        ("1850-1900 (placebo)",  1850, 1900),
        ("1900-1950 (placebo)",  1900, 1950),
        ("1950-2008 (placebo)",  1950, 2008),
    ]
    print("=== Placebo: σ_v computed over different windows ===")
    rows = []
    for label, lo, hi in windows:
        sub = seas[seas["year"].between(lo, hi - 1)]
        sv = sub.groupby("iso3")["t_mean"].std().rename("sigma_v").reset_index()
        df = out.merge(sv, on="iso3").dropna()
        if len(df) < 30: continue
        X = sm.add_constant(df[["sigma_v"]])
        r = sm.OLS(df["log_pop_growth"], X).fit(cov_type="HC1")
        b = r.params["sigma_v"]; se = r.bse["sigma_v"]; p = r.pvalues["sigma_v"]
        rsq = r.rsquared; n = int(r.nobs)
        print(f"  σ_v({label:<22s}):  β = {b:>+7.3f}  SE = {se:.3f}  "
              f"p = {p:>9.4g}  R² = {rsq:.3f}  N = {n}")
        rows.append({"window": label, "lo": lo, "hi": hi, "beta": b,
                      "se": se, "p": p, "rsq": rsq, "n": n})
    return pd.DataFrame(rows)


def rolling_malthus() -> pd.DataFrame:
    panel = pd.read_parquet(DATA / "preindustrial_malthus_panel_extended.parquet")
    panel = panel.dropna(subset=["pop_growth_ann", "log_density",
                                   "t_mean_int", "t_std_int", "cluster"])
    panel["t_anom_int"] = panel["t_mean_int"] - panel.groupby("iso3")["t_mean_int"].transform("mean")

    # Rolling 100-year windows, stepped by 25 years
    rows = []
    for cl in sorted(panel["cluster"].unique()):
        if cl not in PATHWAY_NAMES: continue
        sub_p = panel[panel["cluster"] == cl]
        if len(sub_p) < 40: continue
        for centre in range(1500, 1925, 25):
            lo = centre - 50
            hi = centre + 50
            sub = sub_p[sub_p["year"].between(lo, hi)].copy()
            if len(sub) < 30: continue
            g = sub.groupby("iso3")
            for c in ["pop_growth_ann", "log_density", "t_anom_int", "t_std_int"]:
                sub[c] = sub[c] - g[c].transform("mean")
            X = sm.add_constant(sub[["log_density", "t_anom_int", "t_std_int"]])
            try:
                r = sm.OLS(sub["pop_growth_ann"], X).fit(
                    cov_type="cluster", cov_kwds={"groups": sub["iso3"].values})
                rows.append({
                    "cluster": cl, "pathway": PATHWAY_NAMES[cl],
                    "centre_year": centre,
                    "beta_d": r.params["log_density"],
                    "se_d":   r.bse["log_density"],
                    "p_d":    r.pvalues["log_density"],
                    "delta_T": r.params["t_std_int"],
                    "p_delta_T": r.pvalues["t_std_int"],
                    "N": int(r.nobs),
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def main() -> None:
    placebo = placebo_period()
    placebo.to_parquet(DATA / "placebo_period_long_shadow.parquet", index=False)

    print("\n=== Rolling 100-year-window Malthusian β by pathway ===")
    roll = rolling_malthus()
    roll.to_parquet(DATA / "rolling_malthus.parquet", index=False)
    for cl, g in roll.groupby("cluster"):
        print(f"\n  {PATHWAY_NAMES[cl]}:")
        for _, r in g.iterrows():
            mark = ("***" if r["p_d"] < 0.01 else "**" if r["p_d"] < 0.05
                     else "*" if r["p_d"] < 0.10 else "")
            print(f"    centre={int(r['centre_year']):>4}  β={r['beta_d']:>+8.5f}  "
                  f"SE={r['se_d']:.5f}  p={r['p_d']:.3g}  {mark}  N={r['N']}")

    # Figure: placebo bars + rolling β trajectory
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    ax = axes[0]
    pl = placebo.copy()
    # Order by window start
    pl = pl.sort_values("lo").reset_index(drop=True)
    y = np.arange(len(pl))
    is_head = pl["window"].str.contains("HEADLINE")
    is_plac = pl["window"].str.contains("placebo")
    colors = np.where(is_head, "#202020",
                      np.where(is_plac, "#A0A0A0", "#606060"))
    ax.barh(y, pl["beta"], color=colors, height=0.7,
            edgecolor="#202020", linewidth=0.4)
    for i, row in pl.iterrows():
        mark = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 \
                else "*" if row["p"] < 0.10 else ""
        ax.text(row["beta"] + (0.15 if row["beta"] < 0 else 0.15),
                i, f"{mark}", va="center", fontsize=9)
    ax.axvline(0, color="#404040", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(pl["window"], fontsize=8)
    ax.set_xlabel(r"$\hat\beta_{\sigma_v}$ on log pop growth")
    ax.set_title("(a) Placebo periods for $\sigma_v$", loc="left")

    ax = axes[1]
    pal = gray_palette(len(roll["cluster"].unique()))
    for i, (cl, g) in enumerate(roll.groupby("cluster")):
        g = g.sort_values("centre_year")
        ax.plot(g["centre_year"], g["beta_d"], marker="o", markersize=3,
                linewidth=1.0, color=pal[i],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=PATHWAY_NAMES[cl].split()[0])
        ax.fill_between(g["centre_year"],
                         g["beta_d"] - 1.96*g["se_d"],
                         g["beta_d"] + 1.96*g["se_d"],
                         color=pal[i], alpha=0.10, linewidth=0)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xlabel("Window centre year")
    ax.set_ylabel(r"$\hat\beta_{\ln \text{d}}$ on pop growth")
    ax.set_title("(b) Rolling 100-yr Malthus β by pathway", loc="left")
    ax.legend(loc="best", fontsize=7.5, ncol=2)

    plt.tight_layout()
    fig.savefig(FIG / "fig17_placebo_and_rolling.pdf")
    fig.savefig(FIG / "fig17_placebo_and_rolling.png")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig17_placebo_and_rolling.pdf'}")


if __name__ == "__main__":
    main()
