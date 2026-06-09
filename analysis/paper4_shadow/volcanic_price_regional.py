"""Region-aware VSSI exposure for the pre-1500 price regressions.

eVolv2k VSSI is a global atmospheric forcing, but the *climate response* is
hemispherically asymmetric.  PMIP4 / Toohey-Sigl convention:

  - Tropical eruptions (|lat|<25°) deposit sulfate aerosol that spreads
    poleward in both hemispheres; their forcing reaches all latitudes.
  - Extra-tropical NH eruptions (lat>25°) cool the NH almost exclusively;
    their forcing leaks weakly into the SH (we use a 20% leakage factor).
  - Extra-tropical SH eruptions (lat<-25°) cool the SH; only ~20% reaches
    the NH.

All Allen panel cities sit between 32°N (Madrid) and 58°N (Gdansk), and Roman
Egypt at ~26-30°N — so they should respond fully to tropical + NH-extra
forcing, and only weakly to SH-only events. We build three specifications of
city-year VSSI exposure:

  - vssi_global        : sum of all eruptions, regardless of latitude (baseline)
  - vssi_nh_relevant   : tropical (weight 1.0) + NH-extra (1.0) + SH-extra (0.2)
  - vssi_dist          : same as nh_relevant, weighted by an additional Gaussian
                         in |phi_c - phi_e| with bandwidth 30° latitude.

Outputs:
    analysis/data/volcanic_price_regional.parquet
    analysis/figures/paper4_v2/figA5_volcanic_price_regional.{pdf,png}
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

# ── City latitudes (approximate centroids) ──────────────────────────────────
CITY_LAT = {
    "Amsterdam": 52.37, "Antwerp": 51.22, "Augsburg": 48.37, "Gdansk": 54.35,
    "Krakow": 50.06, "Leipzig": 51.34, "London": 51.51, "Lwow": 49.84,
    "Madrid": 40.42, "Munich": 48.14, "Naples": 40.85, "Northern Italy": 45.00,
    "Paris": 48.86, "Strasbourg": 48.58, "Tuscany": 43.77, "Valencia": 39.47,
    "Vienna": 48.21, "Warsaw": 52.23, "Florence": 43.77,
}

# Leakage factor for opposite-hemisphere extra-tropical eruptions
OPP_HEMISPHERE_LEAK = 0.20
# Gaussian bandwidth on |Δlat| for distance-decay spec (degrees)
LAT_BANDWIDTH = 30.0


def _parse_sigl_with_lat() -> pd.DataFrame:
    with open(DATA / "eVolv2k_sigl_toohey_2024.tab") as f:
        lines = f.read().splitlines()
    data_start = next(i + 1 for i, l in enumerate(lines) if l.startswith("*/"))
    rows = []
    for line in lines[data_start + 1:]:
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < 13: continue
        rows.append({"year": parts[0], "lat": parts[4],
                     "vssi": parts[7], "loc": parts[10]})
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["vssi"] = pd.to_numeric(df["vssi"], errors="coerce")
    return df.dropna(subset=["year", "vssi", "lat"])


def _exposure(eruptions: pd.DataFrame, city_lat: float) -> tuple:
    """Return (annual VSSI global, NH-relevant, dist-weighted) series for one
    city, in a year-indexed Series with no gaps."""
    e = eruptions.copy()
    e["year"] = e["year"].astype(int)
    # Hemispheric classification of city
    city_hem = "NH" if city_lat >= 0 else "SH"
    # Eruption class
    abs_lat = e["lat"].abs()
    is_tropical = abs_lat < 25
    is_NH = (e["lat"] >= 25) & (~is_tropical)
    is_SH = (e["lat"] <= -25) & (~is_tropical)
    # Hemispheric weight given city
    same_hem = (city_hem == "NH") * is_NH + (city_hem == "SH") * is_SH
    opp_hem = (city_hem == "NH") * is_SH + (city_hem == "SH") * is_NH
    hem_w = np.where(is_tropical, 1.0,
            np.where(same_hem, 1.0,
            np.where(opp_hem, OPP_HEMISPHERE_LEAK, 0.0)))
    # Distance weight
    dlat = (e["lat"] - city_lat).abs()
    dist_w = np.exp(-(dlat ** 2) / (2 * LAT_BANDWIDTH ** 2))
    return e["year"], e["vssi"], hem_w, dist_w


def _build_city_year_exposure(eruptions: pd.DataFrame,
                                year_min: int, year_max: int) -> pd.DataFrame:
    out_rows = []
    for city, lat in CITY_LAT.items():
        yrs, vssi, hw, dw = _exposure(eruptions, lat)
        # Sum each contribution by year
        e = pd.DataFrame({"year": yrs.values, "vssi": vssi.values,
                          "hw": hw, "dw": dw})
        e["vssi_global"] = e["vssi"]
        e["vssi_nh"]     = e["vssi"] * e["hw"]
        e["vssi_dist"]   = e["vssi"] * e["hw"] * e["dw"]
        agg = e.groupby("year", as_index=False).agg(
            vssi_global=("vssi_global", "sum"),
            vssi_nh=("vssi_nh", "sum"),
            vssi_dist=("vssi_dist", "sum"))
        # Reindex to full year coverage
        full = pd.DataFrame({"year": range(year_min - 10, year_max)})
        agg = full.merge(agg, on="year", how="left").fillna(0.0)
        agg["city"] = city
        out_rows.append(agg)
    return pd.concat(out_rows, ignore_index=True)


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
    panel["decade"] = (panel["year"] // 10) * 10
    return panel


def _add_lags(df: pd.DataFrame, var: str, n: int = 6) -> pd.DataFrame:
    """Calendar-year lags via a dense (city, year) reindex. shift() by row
    position would break for cities with year gaps (Tuscany has many)."""
    out_parts = []
    for city, sub in df.sort_values(["city", "year"]).groupby("city"):
        ymin, ymax = int(sub["year"].min()), int(sub["year"].max())
        full = pd.DataFrame({"year": range(ymin - n, ymax + 1)})
        full = full.merge(sub[["year", var]], on="year", how="left")
        full[var] = full[var].fillna(0.0)
        for L in range(0, n):
            full[f"{var}_L{L}"] = full[var].shift(L).fillna(0.0)
        # Keep only rows that were present in original
        valid = full.merge(sub[["year"]], on="year", how="inner")
        valid["city"] = city
        out_parts.append(valid)
    lagged = pd.concat(out_parts, ignore_index=True)
    return df.merge(lagged[["city", "year"] + [f"{var}_L{L}" for L in range(n)]],
                    on=["city", "year"], how="left")


def _run_dl(df: pd.DataFrame, vssi_col: str,
            controls: list[str] | None = None) -> dict:
    lag_cols = [f"{vssi_col}_L{L}" for L in range(0, 6)]
    city_dum = pd.get_dummies(df["city"], prefix="city", drop_first=True).astype(float)
    decade_dum = pd.get_dummies(df["decade"], prefix="dec", drop_first=True).astype(float)
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df[lag_cols], city_dum, decade_dum]
    if controls:
        parts.append(df[controls].fillna(0.0).astype(float))
    X = pd.concat(parts, axis=1).astype(float)
    y = df["log_price"]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})
    lags = pd.DataFrame([{"lag": L, "beta": res.params[lag_cols[L]],
                          "se": res.bse[lag_cols[L]],
                          "p": res.pvalues[lag_cols[L]]} for L in range(6)])
    V = res.cov_params().loc[lag_cols, lag_cols].values
    cum_rows = []
    for k in range(6):
        wts = np.zeros(len(lag_cols)); wts[:k + 1] = 1.0
        b = float(sum(res.params[c] for c in lag_cols[:k + 1]))
        se = float(np.sqrt(wts @ V @ wts))
        p = 2 * (1 - norm.cdf(abs(b / se))) if se > 0 else np.nan
        cum_rows.append({"horizon": k, "beta_cum": b, "se_cum": se, "p_cum": p})
    return {"lags": lags, "cum": pd.DataFrame(cum_rows),
            "n": int(res.nobs), "r2": float(res.rsquared)}


def _print_lag_table(specs: dict[str, dict]) -> None:
    print("\n  Per-lag VSSI coefficients across specs:")
    rows = next(iter(specs.values()))["lags"]["lag"]
    header = "  lag |" + " |".join(f" {name:>18}" for name in specs.keys())
    print(header)
    for L in rows:
        cells = []
        for name, r in specs.items():
            b = r["lags"].iloc[L]["beta"]; se = r["lags"].iloc[L]["se"]; p = r["lags"].iloc[L]["p"]
            stars = ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")
            cells.append(f" {b:+.5f}({se:.5f}){stars}")
        print(f"   L{int(L)} |" + " |".join(cells))


def main() -> None:
    print("=== Regional / latitude-aware VSSI exposure ===\n")
    eruptions = _parse_sigl_with_lat()
    eruptions = eruptions[(eruptions["year"] >= START_YEAR - 10)
                          & (eruptions["year"] < END_YEAR)].copy()
    print(f"Pre-1500 eruptions in window: {len(eruptions)}")
    e = eruptions.copy()
    e["lat_class"] = pd.cut(e["lat"], bins=[-91, -25, 25, 91],
                              labels=["SH_extra", "tropical", "NH_extra"])
    print(f"  By latitudinal class:")
    print(e.groupby("lat_class").agg(
        n=("vssi", "size"), total_vssi=("vssi", "sum")
    ).round(2).to_string())

    expo = _build_city_year_exposure(eruptions, START_YEAR, END_YEAR)
    print(f"\nCity-year exposure panel: {len(expo):,} rows, "
          f"{expo['city'].nunique()} cities")

    price = _build_european_panel()
    df = price.merge(expo, on=["city", "year"], how="left")
    df = df.dropna(subset=["log_price"])
    # Add lags for each VSSI variant
    df = _add_lags(df, "vssi_global")
    df = _add_lags(df, "vssi_nh")
    df = _add_lags(df, "vssi_dist")

    # Merge conflict + pandemic controls for the final spec
    cp = pd.read_parquet(DATA / "conflict_pandemic_panel.parquet")
    df = df.merge(cp, on=["city", "year"], how="left")

    print(f"\nMerged regression panel: N={len(df):,}, "
          f"cities={df['city'].nunique()}, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # Run 4 specifications
    specs = {
        "(1) global, no ctrl": _run_dl(df, "vssi_global"),
        "(2) NH-relevant":     _run_dl(df, "vssi_nh"),
        "(3) dist-weighted":   _run_dl(df, "vssi_dist"),
        "(4) NH + conflict":   _run_dl(df, "vssi_nh",
                                       controls=["war_active", "n_active_wars",
                                                 "plague_active", "siege_active"]),
    }

    _print_lag_table(specs)

    print("\n  Cumulative response sum_{L=0..k} β_L:")
    for name, r in specs.items():
        c = r["cum"]
        print(f"  {name}:")
        for _, row in c.iterrows():
            stars = ("***" if row["p_cum"] < 0.01 else "**" if row["p_cum"] < 0.05
                     else "*" if row["p_cum"] < 0.10 else "")
            print(f"    L0..L{int(row['horizon'])}: β = {row['beta_cum']:+.5f} "
                  f"(SE {row['se_cum']:.5f}, p={row['p_cum']:.3f}) {stars}")

    # Persist
    rows = []
    for name, r in specs.items():
        for _, row in r["cum"].iterrows():
            rows.append({"spec": name, **row.to_dict(), "n": r["n"], "r2": r["r2"]})
    pd.DataFrame(rows).to_parquet(DATA / "volcanic_price_regional.parquet", index=False)

    # ── Figure: per-lag coefficients across specs ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = {"(1) global, no ctrl": "#404040",
              "(2) NH-relevant":     "#0072B2",
              "(3) dist-weighted":   "#009E73",
              "(4) NH + conflict":   "#A02020"}
    ax = axes[0]
    for i, (name, r) in enumerate(specs.items()):
        x = r["lags"]["lag"].values + i * 0.13 - 0.2
        ax.errorbar(x, r["lags"]["beta"], yerr=1.96 * r["lags"]["se"],
                    fmt="o", color=colors[name], markerfacecolor="white",
                    markeredgewidth=1.0, ecolor=colors[name], elinewidth=0.7,
                    capsize=2, label=name)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xlabel(r"Lag (years)")
    ax.set_ylabel(r"$\beta_L$ per Tg of VSSI exposure")
    ax.set_title("(a) Per-lag coefficients", loc="left", fontsize=10.5)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, (name, r) in enumerate(specs.items()):
        x = r["cum"]["horizon"].values + i * 0.10 - 0.15
        ax.errorbar(x, r["cum"]["beta_cum"], yerr=1.96 * r["cum"]["se_cum"],
                    fmt="o-", color=colors[name], markerfacecolor="white",
                    markeredgewidth=1.0, ecolor=colors[name], elinewidth=0.7,
                    capsize=2, label=name)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xlabel(r"Horizon (years)")
    ax.set_ylabel(r"Cumulative $\sum_{l=0}^k \beta_l$")
    ax.set_title("(b) Cumulative response", loc="left", fontsize=10.5)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.3)

    fig.suptitle("Regional VSSI exposure: latitude-banded vs. distance-weighted vs. baseline",
                 y=1.03, x=0.04, ha="left", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG / "figA5_volcanic_price_regional.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figA5_volcanic_price_regional.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {FIG/'figA5_volcanic_price_regional.pdf'}")


if __name__ == "__main__":
    main()
