"""Conflict-and-pandemic-robust re-runs of the pre-1500 volcanic price
regressions.

Two exercises:
  1.  Continuous-VSSI distributed-lag price regression with city + decade FE,
      augmented with: war_active (0/1), n_active_wars, plague_active (0/1),
      siege_active (0/1).  Reports each VSSI lag and the cumulative response
      with and without controls.
  2.  Allen Samalas-1257 event study with the same controls absorbed at each
      cell.

Outputs:
    analysis/data/volcanic_price_robust_continuous.parquet
    analysis/data/volcanic_price_robust_event.parquet
    analysis/figures/paper4_v2/figA4_volcanic_price_robust.{pdf,png}
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"
START_YEAR, END_YEAR = 1259, 1500


def _parse_sigl() -> pd.DataFrame:
    with open(DATA / "eVolv2k_sigl_toohey_2024.tab") as f:
        lines = f.read().splitlines()
    data_start = next(i + 1 for i, l in enumerate(lines) if l.startswith("*/"))
    rows = []
    for line in lines[data_start + 1:]:
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < 13: continue
        rows.append({"year": parts[0], "vssi": parts[7]})
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vssi"] = pd.to_numeric(df["vssi"], errors="coerce")
    return df.dropna(subset=["year", "vssi"])[["year", "vssi"]]


def _build_european_panel() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "allen_silver_wheat_prices_1259_1914.parquet")
    raw = pd.read_csv(DATA / "allen_wage_panel.csv")
    raw["year_CE"] = pd.to_numeric(raw["year_CE"], errors="coerce")
    raw["wheat_price"] = pd.to_numeric(raw["wheat_price"], errors="coerce")
    raw = raw.dropna(subset=["year_CE", "wheat_price"])
    tus = raw[(raw["region"] == "Tuscany") & (raw["wheat_price"] > 0)].copy()
    tus = tus.assign(log_price=np.log(tus["wheat_price"]),
                     city="Tuscany",
                     year=tus["year_CE"].astype(int))[["city", "year", "log_price"]]
    panel = pd.concat([p[["city", "year", "log_price"]], tus], ignore_index=True)
    panel = panel[(panel["year"] >= START_YEAR) & (panel["year"] < END_YEAR)].copy()
    panel = panel.dropna(subset=["log_price"])

    sigl = _parse_sigl()
    sigl = sigl[(sigl["year"] >= START_YEAR - 10) & (sigl["year"] < END_YEAR)].copy()
    sigl["year"] = sigl["year"].astype(int)
    annual = (sigl.groupby("year", as_index=False)["vssi"].sum()
                  .set_index("year").reindex(range(START_YEAR - 10, END_YEAR))
                  .fillna(0.0).reset_index().rename(columns={"index": "year"}))
    for L in range(0, 6):
        annual[f"vssi_L{L}"] = annual["vssi"].shift(L).fillna(0.0)

    cp = pd.read_parquet(DATA / "conflict_pandemic_panel.parquet")
    df = panel.merge(annual, on="year", how="left").merge(cp, on=["city", "year"], how="left")
    df["decade"] = (df["year"] // 10) * 10
    return df


def _run_dl(df: pd.DataFrame, controls: list[str]) -> dict:
    """Distributed-lag regression with city + decade FE, optionally adding
    controls. Returns dict with per-lag and cumulative coefficients.
    """
    lag_cols = [f"vssi_L{L}" for L in range(0, 6)]
    city_dum = pd.get_dummies(df["city"], prefix="city", drop_first=True).astype(float)
    decade_dum = pd.get_dummies(df["decade"], prefix="dec", drop_first=True).astype(float)
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df[lag_cols], city_dum, decade_dum]
    if controls:
        parts.append(df[controls].fillna(0.0).astype(float))
    X = pd.concat(parts, axis=1).astype(float)
    y = df["log_price"]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})
    lags = pd.DataFrame([{"lag": L, "beta": res.params[f"vssi_L{L}"],
                          "se": res.bse[f"vssi_L{L}"],
                          "p": res.pvalues[f"vssi_L{L}"]} for L in range(6)])
    V = res.cov_params().loc[lag_cols, lag_cols].values
    cum = []
    for k in range(6):
        wts = np.zeros(len(lag_cols)); wts[:k + 1] = 1.0
        b = sum(res.params[f"vssi_L{L}"] for L in range(k + 1))
        se = float(np.sqrt(wts @ V @ wts))
        p = 2 * (1 - norm.cdf(abs(b / se))) if se > 0 else np.nan
        cum.append({"horizon": k, "beta_cum": b, "se_cum": se, "p_cum": p})
    cum = pd.DataFrame(cum)
    ctrl_rows = []
    for c in controls:
        if c in res.params:
            ctrl_rows.append({"name": c, "beta": res.params[c], "se": res.bse[c],
                              "p": res.pvalues[c]})
    return {"lags": lags, "cum": cum, "controls": pd.DataFrame(ctrl_rows),
            "n": int(res.nobs), "r2": float(res.rsquared)}


def _run_event(df: pd.DataFrame, controls: list[str], onset: int,
                outcome: str = "log_price") -> pd.DataFrame:
    """Event-study with city FE and bin dummies, ±25-yr bins, h=-25 reference."""
    d = df.dropna(subset=[outcome]).copy()
    d["h"] = d["year"] - onset
    bin_edges = np.array([-np.inf, -37.5, -12.5, 12.5, 37.5, 62.5, 87.5, np.inf])
    bin_labels = ["-50", "-25", "+0", "+25", "+50", "+75", "+100"]
    d["hbin"] = pd.cut(d["h"], bin_edges, labels=bin_labels, right=False)
    d = d.dropna(subset=["hbin"])
    D = pd.get_dummies(d["hbin"], prefix="h").astype(float)
    if "h_-25" in D.columns: D = D.drop(columns="h_-25")
    F = pd.get_dummies(d["city"].astype(str), prefix="city",
                       drop_first=True).astype(float)
    parts = [pd.Series(1.0, index=d.index, name="const"), D, F]
    if controls:
        parts.append(d[controls].fillna(0.0).astype(float))
    X = pd.concat(parts, axis=1).astype(float)
    n_clusters = d["city"].nunique()
    if n_clusters > 1:
        res = sm.OLS(d[outcome], X).fit(cov_type="cluster",
                                         cov_kwds={"groups": d["city"]})
    else:
        res = sm.OLS(d[outcome], X).fit(cov_type="HC1")
    rows = []
    for col in D.columns:
        rows.append({"horizon_bin": col.replace("h_", ""),
                     "beta": res.params[col], "se": res.bse[col],
                     "p": res.pvalues[col]})
    rows.append({"horizon_bin": "-25", "beta": 0.0, "se": 0.0, "p": np.nan})
    out = pd.DataFrame(rows)
    out["_ord"] = out["horizon_bin"].astype(int)
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    out["n"] = int(res.nobs); out["n_cities"] = n_clusters
    return out


def main() -> None:
    print("=== Pre-1500 volcanic price regressions, conflict-robust ===\n")
    df = _build_european_panel()
    print(f"European panel: N = {len(df):,}, cities = {df['city'].nunique()}, "
          f"years {df['year'].min()}-{df['year'].max()}")

    controls = ["war_active", "n_active_wars", "plague_active", "siege_active"]
    print(f"\nControls: {controls}")

    # ── (1) Continuous-VSSI distributed-lag ──────────────────────────────────
    print("\n=== Continuous VSSI distributed-lag ===")
    base = _run_dl(df, controls=[])
    ctrl = _run_dl(df, controls=controls)
    cmp_lags = base["lags"].merge(ctrl["lags"], on="lag",
                                    suffixes=("_base", "_ctrl"))
    cmp_cum = base["cum"].merge(ctrl["cum"], on="horizon",
                                  suffixes=("_base", "_ctrl"))
    print(f"  N = {base['n']}, R² base = {base['r2']:.3f}, "
          f"R² with controls = {ctrl['r2']:.3f}")
    print("\n  Per-lag VSSI coefficients (baseline vs. + conflict/pandemic controls):")
    print("  lag |   baseline (SE)        |   with controls (SE)")
    for _, r in cmp_lags.iterrows():
        s_base = "" if r["p_base"] >= 0.10 else ("*" if r["p_base"] >= 0.05
                                                  else "**" if r["p_base"] >= 0.01 else "***")
        s_ctrl = "" if r["p_ctrl"] >= 0.10 else ("*" if r["p_ctrl"] >= 0.05
                                                  else "**" if r["p_ctrl"] >= 0.01 else "***")
        print(f"   L{int(r['lag'])} | {r['beta_base']:+.5f} ({r['se_base']:.5f}){s_base:<3}"
              f" | {r['beta_ctrl']:+.5f} ({r['se_ctrl']:.5f}){s_ctrl}")
    print("\n  Cumulative response (sum of lags 0..k):")
    print("    k |   baseline             |   with controls")
    for _, r in cmp_cum.iterrows():
        print(f"   L0..L{int(r['horizon'])} | "
              f"β_cum = {r['beta_cum_base']:+.5f} (SE {r['se_cum_base']:.5f}, p={r['p_cum_base']:.3f}) "
              f"| {r['beta_cum_ctrl']:+.5f} (SE {r['se_cum_ctrl']:.5f}, p={r['p_cum_ctrl']:.3f})")

    if len(ctrl["controls"]):
        print("\n  Control coefficients (in the controls model):")
        for _, r in ctrl["controls"].iterrows():
            stars = "" if r["p"] >= 0.10 else ("*" if r["p"] >= 0.05
                                                else "**" if r["p"] >= 0.01 else "***")
            print(f"    {r['name']:<20}: β = {r['beta']:+.4f}  SE = {r['se']:.4f}  p = {r['p']:.3g} {stars}")

    cmp_cum.to_parquet(DATA / "volcanic_price_robust_continuous.parquet", index=False)

    # ── (2) Samalas event study, with vs. without controls ──────────────────
    print("\n=== Samalas 1257 event study, with vs. without controls ===")
    es_base = _run_event(df, controls=[], onset=1257)
    es_ctrl = _run_event(df, controls=controls, onset=1257)
    print("  horizon | baseline (SE)        | with controls (SE)")
    merged = es_base.merge(es_ctrl, on="horizon_bin", suffixes=("_base", "_ctrl"))
    for _, r in merged.iterrows():
        s_base = "" if r["p_base"] >= 0.10 else ("*" if r["p_base"] >= 0.05
                                                  else "**" if r["p_base"] >= 0.01 else "***")
        s_ctrl = "" if r["p_ctrl"] >= 0.10 else ("*" if r["p_ctrl"] >= 0.05
                                                  else "**" if r["p_ctrl"] >= 0.01 else "***")
        print(f"   {r['horizon_bin']:>3} | {r['beta_base']:+.4f} ({r['se_base']:.4f}){s_base:<3} | "
              f"{r['beta_ctrl']:+.4f} ({r['se_ctrl']:.4f}){s_ctrl}")
    merged.to_parquet(DATA / "volcanic_price_robust_event.parquet", index=False)

    # ── Figure: side-by-side IRFs ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.errorbar(base["lags"]["lag"], base["lags"]["beta"],
                yerr=1.96 * base["lags"]["se"], fmt="o-", color="#202020",
                markerfacecolor="white", markeredgewidth=1.0,
                ecolor="#404040", elinewidth=0.8, capsize=2.5, label="Baseline")
    ax.errorbar(ctrl["lags"]["lag"] + 0.18, ctrl["lags"]["beta"],
                yerr=1.96 * ctrl["lags"]["se"], fmt="s-", color="#A02020",
                markerfacecolor="white", markeredgewidth=1.0,
                ecolor="#A02020", elinewidth=0.8, capsize=2.5,
                label="With conflict + pandemic")
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xlabel(r"Lag (years)")
    ax.set_ylabel(r"$\beta_L$ per Tg of VSSI")
    ax.set_title("(a) Per-lag VSSI response", loc="left", fontsize=10.5)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bb = es_base; cc = es_ctrl
    xb = bb["horizon_bin"].astype(int).values
    xc = cc["horizon_bin"].astype(int).values
    ax.errorbar(xb, bb["beta"], yerr=1.96 * bb["se"], fmt="o-",
                color="#202020", markerfacecolor="white", markeredgewidth=1.0,
                ecolor="#404040", elinewidth=0.8, capsize=2.5, label="Baseline")
    ax.errorbar(xc + 2.5, cc["beta"], yerr=1.96 * cc["se"], fmt="s-",
                color="#A02020", markerfacecolor="white", markeredgewidth=1.0,
                ecolor="#A02020", elinewidth=0.8, capsize=2.5,
                label="With conflict + pandemic")
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.axvline(0, color="#a02020", linewidth=0.6, linestyle="--")
    ax.set_xlabel(r"Years from eruption (25-yr bins)")
    ax.set_ylabel(r"$\Delta$ log wheat price")
    ax.set_title("(b) Samalas 1257 event study", loc="left", fontsize=10.5)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.3)

    fig.suptitle("Conflict + pandemic robustness for pre-1500 volcanic-price regressions",
                 y=1.03, x=0.04, ha="left", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / "figA4_volcanic_price_robust.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figA4_volcanic_price_robust.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {FIG/'figA4_volcanic_price_robust.pdf'}")


if __name__ == "__main__":
    main()
