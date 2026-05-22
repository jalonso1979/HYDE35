"""Pooled rolling-window OLS with country FE and HAC SEs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def pooled_rolling_elasticity(
    df: pd.DataFrame,
    y: str,
    x: str,
    window: int = 40,
    nw_lag: int | None = None,
    min_obs: int = 50,
) -> pd.DataFrame:
    """Rolling window y_ct = alpha_c + beta_w * x_ct + eps_ct.

    Returns one row per window center: center_year, beta, se, ci_low, ci_high, n.
    """
    if nw_lag is None:
        nw_lag = window // 4
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    years = df["year"].to_numpy()
    rows = []
    half = window // 2
    for w in range(years.min() + half, years.max() - half + 1):
        mask = (years >= w - half) & (years < w + half)
        sub = df.loc[mask, ["iso3", y, x]].dropna()
        if len(sub) < min_obs or sub["iso3"].nunique() < 2:
            rows.append({"center_year": w, "beta": np.nan, "se": np.nan,
                          "ci_low": np.nan, "ci_high": np.nan, "n": len(sub)})
            continue
        dums = pd.get_dummies(sub["iso3"], drop_first=True, dtype=float)
        X = sm.add_constant(pd.concat([sub[[x]].astype(float), dums], axis=1))
        res = sm.OLS(sub[y].astype(float).to_numpy(), X.to_numpy()).fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lag}
        )
        b = float(res.params[1])
        s = float(res.bse[1])
        rows.append({"center_year": w, "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s, "n": len(sub)})
    return pd.DataFrame(rows)
