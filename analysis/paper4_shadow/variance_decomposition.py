"""Variance decomposition of the conflict/pandemic regressors after
two-way (city + decade) fixed effects.

For each regressor x_{c,t}:
  total_var       = Var(x)
  between_city    = Var( mean_t x_{c,t} | c )
  between_decade  = Var( mean_c x_{c,t} | d ) where d = floor(t/10)*10
  residual        = Var( x_{c,t} - mean_t - mean_d + mean_total )

If residual variance is a substantial share of total, the regressor can be
separately identified inside a two-way FE specification.  If residual variance
collapses to near zero, the regressor is mechanically absorbed by the fixed
effects and the data cannot speak to its independent effect.

Restricted to the city-year subset that the volcanic price regression actually
uses (1259-1500 for European cities, 100-650 for Roman Egypt).
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def _twoway_decomp(df: pd.DataFrame, x: str, city_col: str = "city",
                    decade_col: str = "decade") -> dict:
    """Two-way variance decomposition for one regressor."""
    d = df.dropna(subset=[x]).copy()
    grand = d[x].mean()
    city_means = d.groupby(city_col)[x].transform("mean")
    decade_means = d.groupby(decade_col)[x].transform("mean")
    resid = d[x] - city_means - decade_means + grand
    return {"x": x,
            "n": int(len(d)),
            "total_var": float(d[x].var(ddof=0)),
            "between_city_var": float((city_means - grand).var(ddof=0)),
            "between_decade_var": float((decade_means - grand).var(ddof=0)),
            "residual_var": float(resid.var(ddof=0)),
            "mean": float(d[x].mean())}


def _format(row) -> str:
    tv = row["total_var"]; bc = row["between_city_var"]
    bd = row["between_decade_var"]; rv = row["residual_var"]
    bc_pct = 100 * bc / tv if tv > 0 else 0
    bd_pct = 100 * bd / tv if tv > 0 else 0
    rv_pct = 100 * rv / tv if tv > 0 else 0
    return (f"  {row['x']:<22} N={row['n']:>5}  mean={row['mean']:.4f}  "
            f"total_var={tv:.5f}  "
            f"city {bc_pct:5.1f}%  decade {bd_pct:5.1f}%  "
            f"residual {rv_pct:5.1f}%")


def main() -> None:
    print("Loading conflict-pandemic panel + Allen/Harper city panels …")
    cp = pd.read_parquet(DATA / "conflict_pandemic_panel.parquet")
    # Allen Europe panel cells
    allen = pd.read_parquet(DATA / "allen_silver_wheat_prices_1259_1914.parquet")
    raw = pd.read_csv(DATA / "allen_wage_panel.csv")
    raw["year_CE"] = pd.to_numeric(raw["year_CE"], errors="coerce")
    raw["wheat_price"] = pd.to_numeric(raw["wheat_price"], errors="coerce")
    raw = raw.dropna(subset=["year_CE", "wheat_price"])
    tus = raw[raw["region"] == "Tuscany"].copy()
    tus["city"] = "Tuscany"; tus["year"] = tus["year_CE"].astype(int)
    tus["silver_g_per_hl"] = tus["wheat_price"]
    tus = tus[["city", "year", "silver_g_per_hl"]]
    european = pd.concat([allen[["city", "year", "silver_g_per_hl"]], tus],
                          ignore_index=True)
    european = european[(european["year"] >= 1259) & (european["year"] < 1500)]
    european["decade"] = (european["year"] // 10) * 10

    # Merge in conflict regressors
    eu_with_conflict = european.merge(cp, on=["city", "year"], how="left")
    print(f"\nEuropean panel cells (1259-1499, with conflict merge): {len(eu_with_conflict):,}")
    print(f"  unique cities = {eu_with_conflict['city'].nunique()}")
    print(f"  unique decades = {eu_with_conflict['decade'].nunique()}")

    print("\n=== Variance decomposition on the European wheat-price subset ===")
    print("(percentages reflect share of total variance for each regressor)")
    rows = []
    for x in ["war_active", "n_active_wars", "log_fatalities",
              "plague_active", "siege_active"]:
        if x not in eu_with_conflict.columns: continue
        rows.append(_twoway_decomp(eu_with_conflict, x))
        print(_format(rows[-1]))

    # Same for Egyptian panel (Harper)
    print("\n=== Variance decomposition on the Roman Egypt subset ===")
    hp = pd.read_csv(DATA / "harper_egypt" / "harper_egypt_wheat_prices.csv")
    hp["city"] = "Roman Egypt"
    hp["decade"] = (hp["year"] // 10) * 10
    hp_c = hp.merge(cp, on=["city", "year"], how="left")
    print(f"  cells: {len(hp_c):,}; unique decades = {hp_c['decade'].nunique()}")
    # With only one city, between-city variance is 0; just decade FE applies.
    # We report total vs decade-FE-residual.
    print("  (only 1 cluster → between-city = 0; show decade-FE residual share)")
    for x in ["war_active", "n_active_wars", "log_fatalities",
              "plague_active", "siege_active"]:
        if x not in hp_c.columns: continue
        d = hp_c.dropna(subset=[x]).copy()
        if len(d) < 5: continue
        grand = d[x].mean()
        decade_means = d.groupby("decade")[x].transform("mean")
        resid = d[x] - decade_means + grand
        tv = float(d[x].var(ddof=0))
        bd = float((decade_means - grand).var(ddof=0))
        rv = float(resid.var(ddof=0))
        bd_pct = 100 * bd / tv if tv > 0 else 0
        rv_pct = 100 * rv / tv if tv > 0 else 0
        print(f"  {x:<22} N={len(d):>3}  mean={d[x].mean():.3f}  "
              f"total_var={tv:.5f}  decade {bd_pct:5.1f}%  residual {rv_pct:5.1f}%")

    # Persist
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "conflict_variance_decomp.parquet", index=False)
    print(f"\nSaved {DATA/'conflict_variance_decomp.parquet'}")


if __name__ == "__main__":
    main()
