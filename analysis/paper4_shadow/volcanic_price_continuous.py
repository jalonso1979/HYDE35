"""Continuous-VSSI price regression on the Allen-Studer European panel,
1259--1500.

Complements the decade-IRF Sigl regression (1500--1900) on population by
running the same continuous-forcing design at annual resolution on grain
prices.  Distributed-lag specification: log(price)_{c,t} on contemporaneous
and lagged VSSI exposure (years 0--5) with city + year fixed effects.

Output:
    analysis/data/volcanic_price_continuous.parquet
    analysis/figures/paper4_v2/figA3_volcanic_price_continuous.{pdf,png}
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


def main() -> None:
    print(f"=== Continuous VSSI → European wheat prices, {START_YEAR}--{END_YEAR} ===")

    # Load Allen silver-gram wheat prices and add Tuscany (Malanima) prices
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

    # Build annual VSSI series and lags
    sigl = _parse_sigl()
    sigl = sigl[(sigl["year"] >= START_YEAR - 10) & (sigl["year"] < END_YEAR)].copy()
    sigl["year"] = sigl["year"].astype(int)
    annual = (sigl.groupby("year", as_index=False)["vssi"].sum()
                  .set_index("year").reindex(range(START_YEAR - 10, END_YEAR))
                  .fillna(0.0).reset_index().rename(columns={"index": "year"}))
    for L in range(0, 6):
        annual[f"vssi_L{L}"] = annual["vssi"].shift(L).fillna(0.0)

    df = panel.merge(annual, on="year", how="left")
    print(f"  N city-year obs = {len(df):,}, cities = {df['city'].nunique()}, "
          f"years span {df['year'].min()}–{df['year'].max()}")

    # Distributed-lag regression with city + year FE; cluster SE by city
    city_dum = pd.get_dummies(df["city"], prefix="city", drop_first=True).astype(float)
    year_dum = pd.get_dummies(df["year"], prefix="yr", drop_first=True).astype(float)
    lag_cols = [f"vssi_L{L}" for L in range(0, 6)]
    # year FE would absorb the global VSSI signal exactly — use decade FE instead
    df["decade"] = (df["year"] // 10) * 10
    decade_dum = pd.get_dummies(df["decade"], prefix="dec", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   df[lag_cols], city_dum, decade_dum], axis=1).astype(float)
    y = df["log_price"]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})

    print(f"\n  Regression: log(price) ~ Σ_L VSSI_{{t-L}} + city FE + decade FE")
    print(f"  N = {int(res.nobs)}, R² = {res.rsquared:.3f}")
    rows = []
    for L in range(0, 6):
        c = f"vssi_L{L}"
        if c in res.params:
            beta, se, p = res.params[c], res.bse[c], res.pvalues[c]
            rows.append({"lag": L, "beta": beta, "se": se, "p": p})
            print(f"    lag {L}: β = {beta:+.5f}  SE = {se:.5f}  p = {p:.3g}")
    coefs = pd.DataFrame(rows)
    # Cumulative response (sum of lags 0..k) with delta-method SE
    print("\n  Cumulative response (sum of lags):")
    V = res.cov_params().loc[lag_cols, lag_cols].values
    cumrows = []
    for k in range(0, 6):
        sel = lag_cols[:k + 1]
        wts = np.zeros(len(lag_cols)); wts[:k + 1] = 1.0
        beta_cum = sum(res.params[c] for c in sel)
        se_cum = float(np.sqrt(wts @ V @ wts))
        from scipy.stats import norm
        p_cum = 2 * (1 - norm.cdf(abs(beta_cum / se_cum))) if se_cum > 0 else np.nan
        cumrows.append({"horizon": k, "beta_cum": beta_cum,
                        "se_cum": se_cum, "p_cum": p_cum})
        print(f"    L0..L{k}: β_cum = {beta_cum:+.5f}  SE = {se_cum:.5f}  p = {p_cum:.3g}")
    cum = pd.DataFrame(cumrows)
    full = coefs.merge(cum, left_on="lag", right_on="horizon")
    full.to_parquet(DATA / "volcanic_price_continuous.parquet", index=False)

    # ── Figure: per-lag and cumulative IRF ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax = axes[0]
    ax.errorbar(coefs["lag"], coefs["beta"], yerr=1.96 * coefs["se"],
                fmt="o-", color="#202020", markerfacecolor="white",
                markeredgewidth=1, ecolor="#404040", elinewidth=0.8, capsize=2.5)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xlabel(r"Lag (years)")
    ax.set_ylabel(r"$\beta_L$ per Tg of VSSI")
    ax.set_title(rf"(a) Per-lag response, log(wheat price)~$\sum_L$~VSSI$_{{t-L}}$",
                 loc="left", fontsize=10.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(cum["horizon"], cum["beta_cum"], yerr=1.96 * cum["se_cum"],
                fmt="o-", color="#202020", markerfacecolor="white",
                markeredgewidth=1, ecolor="#404040", elinewidth=0.8, capsize=2.5)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xlabel(r"Horizon (years included)")
    ax.set_ylabel(r"Cumulative $\sum_{l=0}^k \beta_l$ per Tg VSSI")
    ax.set_title(r"(b) Cumulative response (sum L0..Lk)", loc="left", fontsize=10.5)
    ax.grid(alpha=0.3)

    fig.suptitle(rf"Continuous-VSSI wheat-price regression, {START_YEAR}--{END_YEAR}",
                 y=1.04, x=0.04, ha="left", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / "figA3_volcanic_price_continuous.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figA3_volcanic_price_continuous.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\n  Saved {FIG/'figA3_volcanic_price_continuous.pdf'}")


if __name__ == "__main__":
    main()
