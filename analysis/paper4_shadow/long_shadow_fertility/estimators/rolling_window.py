"""Rolling-window OLS elasticity estimator.

For a 40-year window centered at year w, regress y_t = alpha + beta * x_t + eps_t
on observations t in [w - window/2, w + window/2). Returns one row per window
center with beta, Newey-West SE (lag = window // 4), 95% CI, and N.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def rolling_elasticity(
    df: pd.DataFrame,
    y: str,
    x: str,
    window: int = 40,
    nw_lag: int | None = None,
    min_obs: int = 25,
) -> pd.DataFrame:
    """Rolling-window OLS of y on x with Newey-West SEs.

    Parameters
    ----------
    df : DataFrame with columns ('year', y, x). Year-indexed time series.
    window : window width in years.
    nw_lag : Newey-West truncation lag; default = window // 4.
    min_obs : minimum non-NA observations in window to estimate.

    Returns
    -------
    DataFrame with columns: center_year, beta, se, ci_low, ci_high, n.
    """
    if nw_lag is None:
        nw_lag = window // 4
    df = df.sort_values("year").reset_index(drop=True)
    years = df["year"].to_numpy()
    rows = []
    half = window // 2
    for w in range(years.min() + half, years.max() - half + 1):
        mask = (years >= w - half) & (years < w + half)
        sub = df.loc[mask, [y, x]].dropna()
        if len(sub) < min_obs:
            rows.append({"center_year": w, "beta": np.nan, "se": np.nan,
                          "ci_low": np.nan, "ci_high": np.nan, "n": len(sub)})
            continue
        X = sm.add_constant(sub[x].to_numpy())
        res = sm.OLS(sub[y].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag})
        b = float(res.params[1])
        s = float(res.bse[1])
        rows.append({"center_year": w, "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s, "n": len(sub)})
    return pd.DataFrame(rows)
