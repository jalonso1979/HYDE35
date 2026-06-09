"""Price- and wage-side event studies around major pre-1500 eruptions.

The main-text volcanic exercise tests the *demographic* response. This script
tests the *welfare* response over the same channel pre-1500, using three
independent annual outcome series that the demographic regression cannot reach:

  (a) Allen-Studer European city wheat prices (silver g/hl), 1259--1500,
      around the 1257 Samalas eruption (eVolv2k VSSI ≈ 59 Tg, the largest of
      the past 2500 years).
  (b) Allen-Studer European city real wages, same window, around 1257.
  (c) Harper (2016) Roman Egyptian wheat prices (silver g/hl), 45--650 CE,
      around the 536+540 CE LALIA doublet (50+ Tg combined).

Design: log-price (resp. log-wage) regressed on event-time dummies in
±25-year bins (with $h=-25$ as the reference) plus city fixed effects. Cluster
SEs by city. The price-side prediction is symmetric to the demographic one in
sign: a major volcanic cooling raises grain prices and depresses real wages,
with a recovery within 1--2 decades.

Outputs:
    analysis/data/volcanic_price_event_results.parquet
    analysis/figures/paper4_v2/figA2_volcanic_price_event.{pdf,png}
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
FIG.mkdir(parents=True, exist_ok=True)

# Bin edges (centred at -25, 0, +25, ...): reference is h=-25
BIN_EDGES = np.array([-np.inf, -37.5, -12.5, 12.5, 37.5, 62.5, 87.5, np.inf])
BIN_LABELS = ["-50", "-25", "+0", "+25", "+50", "+75", "+100"]
REF_LABEL = "-25"


def _bin(h: np.ndarray) -> pd.Categorical:
    return pd.cut(h, BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)


def _event_study_long(panel: pd.DataFrame, outcome_col: str, onset: int,
                       cluster_col: str = "city",
                       window_low: int = -75, window_hi: int = 100,
                       silent: bool = False) -> pd.DataFrame:
    """One-event panel event study with bin dummies and city fixed effects.

    panel:  long table with columns ['year', 'city', outcome_col]
    onset:  treatment year
    Returns: dataframe with horizon_bin, beta, se, n
    """
    d = panel.dropna(subset=[outcome_col, "year", cluster_col]).copy()
    d["h"] = d["year"] - onset
    d = d[(d["h"] >= window_low) & (d["h"] <= window_hi)].copy()
    if d.empty:
        return pd.DataFrame()
    d["hbin"] = _bin(d["h"].values)
    d = d[d["hbin"].notna()].copy()

    D = pd.get_dummies(d["hbin"], prefix="h").astype(float)
    # drop reference column
    ref_col = f"h_{REF_LABEL}"
    if ref_col in D.columns:
        D = D.drop(columns=ref_col)
    # city FE (drop one)
    F = pd.get_dummies(d[cluster_col].astype(str), prefix="city",
                       drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"), D, F], axis=1).astype(float)
    y = d[outcome_col].astype(float)
    # Use clustered SEs if >1 cluster, else HC1 robust SE
    n_clusters = d[cluster_col].nunique()
    if n_clusters > 1:
        res = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": d[cluster_col].astype(str)})
    else:
        res = sm.OLS(y, X).fit(cov_type="HC1")
    rows = []
    for col in D.columns:
        rows.append({"horizon_bin": col.replace("h_", ""),
                     "beta": res.params[col], "se": res.bse[col],
                     "p": res.pvalues[col]})
    # Add reference at 0
    rows.append({"horizon_bin": REF_LABEL, "beta": 0.0, "se": 0.0, "p": np.nan})
    out = pd.DataFrame(rows)
    # Sort by numeric value of label
    out["_ord"] = out["horizon_bin"].astype(int)
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    out["n"] = int(res.nobs); out["n_cities"] = d[cluster_col].nunique()
    if not silent:
        print(f"  {outcome_col} | onset={onset} | N={int(res.nobs)} "
              f"({d[cluster_col].nunique()} clusters, "
              f"{int(d['year'].min())}-{int(d['year'].max())})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Allen-Studer European panel: prices from raw xls (silver-grams normalised),
# wages from the existing parsed CSV, Tuscany prices from Malanima (parsed CSV).
# Because we always use log(price)+city FE, native units only need to be stable
# within city.
# ─────────────────────────────────────────────────────────────────────────────
def _load_allen_panel() -> pd.DataFrame:
    """Wide-tidy long table: year, city, log_price, log_rwage."""
    # Silver-gram wheat prices, 17 Allen cities (silver g per hectoliter)
    p = pd.read_parquet(DATA / "allen_silver_wheat_prices_1259_1914.parquet")
    p = p[["city", "year", "log_price"]].rename(columns={"log_price": "log_price"})

    # Wages and Tuscany prices from the existing parsed CSV
    raw = pd.read_csv(DATA / "allen_wage_panel.csv")
    raw["year_CE"] = pd.to_numeric(raw["year_CE"], errors="coerce")
    raw["wheat_price"] = pd.to_numeric(raw["wheat_price"], errors="coerce")
    raw["real_wage"] = pd.to_numeric(raw["real_wage"], errors="coerce")
    raw = raw.rename(columns={"year_CE": "year", "region": "city"})
    raw = raw.dropna(subset=["year", "city"]).copy()
    raw["log_rwage"] = np.log(raw["real_wage"].where(raw["real_wage"] > 0))
    # Tuscany wheat prices in Malanima Tuscan units — keep as their own series
    tus = raw[(raw["city"] == "Tuscany") & raw["wheat_price"].notna()].copy()
    tus["log_price"] = np.log(tus["wheat_price"])
    tus = tus[["city", "year", "log_price"]].dropna()
    prices = pd.concat([p, tus], ignore_index=True)
    # Wages by city
    wages = (raw.dropna(subset=["log_rwage"])
                .groupby(["city", "year"], as_index=False)["log_rwage"].mean())
    return prices, wages


# ─────────────────────────────────────────────────────────────────────────────
# Harper Roman Egypt: single-province price + wage series
# ─────────────────────────────────────────────────────────────────────────────
def _load_harper() -> pd.DataFrame:
    p = pd.read_csv(DATA / "harper_egypt" / "harper_egypt_wheat_prices.csv")
    w = pd.read_csv(DATA / "harper_egypt" / "harper_egypt_wheat_wages.csv")
    p["city"] = "Roman Egypt"; w["city"] = "Roman Egypt"
    p["log_price"] = pd.to_numeric(p["log_price"], errors="coerce")
    w["log_wage"] = pd.to_numeric(w["log_wage"], errors="coerce")
    return p[["year", "city", "log_price"]], w[["year", "city", "log_wage"]]


def _plot_event_grid(results: dict[str, pd.DataFrame], title: str,
                     out_path: Path) -> None:
    """Plot one panel per (event, outcome) tuple in a 2×2 grid."""
    n = len(results)
    cols = 2; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9.5, 3.3 * rows), sharex=False)
    axes = np.atleast_2d(axes).flatten()
    for ax, (label, df) in zip(axes, results.items()):
        if df.empty:
            ax.set_axis_off(); continue
        x = df["horizon_bin"].astype(int).values
        y = df["beta"].values; e = 1.96 * df["se"].values
        ax.errorbar(x, y, yerr=e, fmt="o-", color="#202020",
                    markerfacecolor="white", markeredgewidth=1.0,
                    ecolor="#404040", elinewidth=0.8, capsize=2.5)
        ax.axhline(0, color="#404040", linewidth=0.5)
        ax.axvline(0, color="#a02020", linewidth=0.7, linestyle="--")
        n_obs = int(df["n"].iloc[0]) if "n" in df.columns else 0
        nc = int(df["n_cities"].iloc[0]) if "n_cities" in df.columns else 0
        ax.set_title(label + f"  ($N={n_obs}$, {nc} cities)", loc="left", fontsize=10.5)
        ax.set_xlabel(r"Years from eruption (25-yr bins)")
        ax.set_ylabel(r"$\Delta$ log outcome (vs.\ $h=-25$)")
        ax.grid(alpha=0.3)
    for ax in axes[len(results):]:
        ax.set_axis_off()
    fig.suptitle(title, y=1.02, x=0.04, ha="left", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\n  Saved {out_path}")


def main() -> None:
    print("=== Pre-1500 volcanic event studies, price/wage side ===\n")

    # ── Allen panel ──────────────────────────────────────────────────────────
    prices, wages = _load_allen_panel()
    print("Allen-Studer European panel:")
    print(f"  price cities (n_obs): "
          f"{prices.groupby('city').size().to_dict()}")
    print(f"  wage cities (n_obs):  "
          f"{wages.groupby('city').size().to_dict()}")
    print(f"  wheat_price 1180-1320 obs by city:")
    print(prices[(prices['year'] >= 1180) & (prices['year'] <= 1320)]
          .groupby('city').size().to_string())
    print(f"  real_wage 1180-1320 obs by city:")
    print(wages[(wages['year'] >= 1180) & (wages['year'] <= 1320)]
          .groupby('city').size().to_string())
    print()

    results = {}

    print("\n1. Samalas 1257 – wheat prices (Allen + Malanima Tuscany)")
    samalas_p = _event_study_long(prices, "log_price", onset=1257)
    results["Samalas 1257 – European wheat prices"] = samalas_p

    print("\n2. Samalas 1257 – real wages (Allen)")
    samalas_w = _event_study_long(wages, "log_rwage", onset=1257,
                                  window_low=-50, window_hi=100)
    results["Samalas 1257 – European real wages"] = samalas_w

    print("\n3. 1453 Kuwae – wheat prices (broader city panel)")
    kuwae = _event_study_long(prices, "log_price", onset=1453,
                              window_low=-75, window_hi=75)
    results["Kuwae 1453 – European wheat prices"] = kuwae

    # ── Harper panel: LALIA 536+540 and Cyprian 249 ──────────────────────────
    hp_price, hp_wage = _load_harper()

    print("\n3. LALIA 536+540 – Egyptian wheat prices (Harper)")
    lalia = _event_study_long(hp_price, "log_price", onset=540,
                              cluster_col="city",
                              window_low=-100, window_hi=100)
    if not lalia.empty:
        results["LALIA 540 – Egyptian wheat prices"] = lalia

    print("\n4. Cyprian 249 – Egyptian wheat prices (Harper)")
    cyprian = _event_study_long(hp_price, "log_price", onset=249,
                                cluster_col="city",
                                window_low=-100, window_hi=100)
    if not cyprian.empty:
        results["Cyprian onset 249 – Egyptian wheat prices"] = cyprian

    # ── Persist ──────────────────────────────────────────────────────────────
    all_rows = []
    for label, df in results.items():
        if df.empty: continue
        df = df.copy(); df["spec"] = label; all_rows.append(df)
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_parquet(DATA / "volcanic_price_event_results.parquet", index=False)

        # Print compact summary
        print("\n=== Coefficient summary (key horizon bins) ===")
        for label, df in results.items():
            if df.empty: continue
            highlights = df[df["horizon_bin"].isin(["+0", "+25"])]
            print(f"  {label}")
            for _, r in highlights.iterrows():
                stars = ("***" if r["p"] < 0.01
                         else "**" if r["p"] < 0.05
                         else "*" if r["p"] < 0.10 else "")
                print(f"     h={r['horizon_bin']:>3}: β = {r['beta']:+.3f}  "
                      f"SE = {r['se']:.3f}  {stars}")

        _plot_event_grid(results,
                         "Volcanic-eruption event studies, pre-1500 (Allen-Studer + Harper-Egypt)",
                         FIG / "figA2_volcanic_price_event.pdf")


if __name__ == "__main__":
    main()
