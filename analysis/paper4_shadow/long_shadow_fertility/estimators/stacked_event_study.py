"""Multi-country stacked event study with country FE + country trends + cluster SEs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def stacked_event_study(
    df: pd.DataFrame,
    y: str,
    eruption_year: int,
    pre: int = 5,
    post: int = 10,
    reference_h: int = -1,
    cluster_col: str = "iso3",
) -> pd.DataFrame:
    sub = df[df["year"].between(eruption_year - pre, eruption_year + post)].copy()
    sub = sub.dropna(subset=[y, cluster_col]).sort_values([cluster_col, "year"]).reset_index(drop=True)
    sub["h"] = sub["year"] - eruption_year
    horizons = [h for h in range(-pre, post + 1) if h != reference_h]
    for h in horizons:
        sub[f"D_h{h}"] = (sub["h"] == h).astype(int)

    country_dums = pd.get_dummies(sub[cluster_col], drop_first=True, dtype=float)
    sub["year_c"] = sub["year"] - sub["year"].mean()
    countries = sorted(sub[cluster_col].unique())
    for c in countries:
        sub[f"trend_{c}"] = ((sub[cluster_col] == c).astype(float) * sub["year_c"])

    X = sm.add_constant(pd.concat([
        sub[[f"D_h{h}" for h in horizons]],
        country_dums,
        sub[[f"trend_{c}" for c in countries]],
    ], axis=1))
    yv = sub[y].to_numpy()
    cluster = sub[cluster_col].astype("category").cat.codes.to_numpy()
    n_clusters = len(np.unique(cluster))
    ols = sm.OLS(yv, X.to_numpy())
    if n_clusters >= 2:
        res = ols.fit(cov_type="cluster", cov_kwds={"groups": cluster})
    else:
        # Cluster SEs require >=2 clusters; fall back to HC1 for single-country case.
        res = ols.fit(cov_type="HC1")

    # Reference horizon is omitted by construction (baseline category).
    rows = []
    for i, h in enumerate(horizons):
        idx = i + 1
        b = float(res.params[idx])
        s = float(res.bse[idx])
        rows.append({"h": h, "delta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)
