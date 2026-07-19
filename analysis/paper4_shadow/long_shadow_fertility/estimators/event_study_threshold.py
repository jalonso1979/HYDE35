"""Estimator helpers for country event-study threshold figure (Phase 10 / D3a).

Functions
---------
crossing_year   : First year a country's log real wage exceeds a threshold c.
rolling_elasticity : Rolling-window OLS slope (beta + se) of y on x.
"""
import pandas as pd
import numpy as np


def crossing_year(df_country, z="log_real_wage", c=9.97, year_col="year"):
    """Find the first year in which z strictly exceeds c.

    Parameters
    ----------
    df_country : pd.DataFrame
    z          : str   column name for the running variable (default 'log_real_wage')
    c          : float threshold (default 9.97 — Phase 9 pooled estimate)
    year_col   : str   column name for years (default 'year')

    Returns
    -------
    int or None   First year where z > c; None if z never exceeds c.
    """
    valid = df_country.dropna(subset=[z])
    above = valid[valid[z] > c]
    if above.empty:
        return None
    return int(above[year_col].min())


def rolling_elasticity(df_country, y, x, window=15, year_col="year"):
    """Rolling-window OLS slope of y on x over a centred window of `window` years.

    For each distinct year t in the data, includes all observations in the
    half-open interval [t - window//2,  t + window//2].  Windows with fewer
    than 5 valid observations are skipped.

    Parameters
    ----------
    df_country : pd.DataFrame
    y          : str   outcome column
    x          : str   regressor column
    window     : int   full-width window in years (default 15)
    year_col   : str   column name for years (default 'year')

    Returns
    -------
    pd.DataFrame with columns [year_col, 'beta', 'se']
        beta — OLS slope of y on x (with intercept)
        se   — standard error of beta
    """
    sub = df_country.dropna(subset=[y, x, year_col]).sort_values(year_col).copy()
    half = window // 2
    rows = []

    for year in sub[year_col].unique():
        w = sub[
            (sub[year_col] >= year - half) & (sub[year_col] <= year + half)
        ]
        if len(w) < 5:
            continue

        X = np.column_stack([np.ones(len(w)), w[x].to_numpy(dtype=float)])
        Y = w[y].to_numpy(dtype=float)

        try:
            b, *_ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ b
            dof = len(w) - 2
            if dof <= 0:
                continue
            s2 = float((resid @ resid) / dof)
            var_b = s2 * np.linalg.inv(X.T @ X)[1, 1]
            rows.append(
                {
                    year_col: int(year),
                    "beta": float(b[1]),
                    "se": float(np.sqrt(max(var_b, 0.0))),
                }
            )
        except np.linalg.LinAlgError:
            continue

    return pd.DataFrame(rows)
