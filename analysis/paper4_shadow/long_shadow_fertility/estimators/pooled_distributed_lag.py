"""Pooled distributed-lag OLS with country FE + year FE + cluster-robust SE."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _add_lags(df: pd.DataFrame, var: str, lags: int, unit_col: str) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values([unit_col, "year"]).copy()
    lag_cols: list[str] = []
    for k in range(lags + 1):
        col = f"{var}_lag{k}"
        df[col] = df.groupby(unit_col)[var].shift(k)
        lag_cols.append(col)
    return df, lag_cols


def fit_pooled_distributed_lag(
    df: pd.DataFrame,
    y: str,
    x: str,
    lags: int = 3,
    unit_col: str = "iso3",
    controls: Sequence[str] | None = None,
    year_fe: bool = True,
) -> pd.DataFrame:
    df_lag, lag_cols = _add_lags(df, x, lags, unit_col)
    ctrl_cols = list(controls or [])
    keep = [y, unit_col, "year"] + lag_cols + ctrl_cols
    sub = df_lag[keep].dropna()
    if sub[unit_col].nunique() < 2:
        raise ValueError("Need >= 2 units")
    unit_dums = pd.get_dummies(sub[unit_col], drop_first=True, dtype=float)
    if year_fe:
        year_dums = pd.get_dummies(sub["year"].astype(int), drop_first=True, dtype=float)
        year_dums.columns = [f"y_{c}" for c in year_dums.columns]
    else:
        year_dums = pd.DataFrame(index=sub.index)
    X = sm.add_constant(pd.concat([
        sub[lag_cols].astype(float),
        unit_dums,
        year_dums,
        sub[ctrl_cols].astype(float) if ctrl_cols else pd.DataFrame(index=sub.index),
    ], axis=1))
    cluster = sub[unit_col].astype("category").cat.codes.to_numpy()
    res = sm.OLS(sub[y].astype(float).to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )
    rows = []
    for k, col in enumerate(lag_cols):
        idx = 1 + k
        b = float(res.params[idx])
        s = float(res.bse[idx])
        rows.append({"lag": k, "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    lag_idx = np.arange(1, 1 + len(lag_cols))
    cum_b = float(res.params[lag_idx].sum())
    cov_block = res.cov_params()[np.ix_(lag_idx, lag_idx)]
    cum_se = float(np.sqrt(np.ones(len(lag_idx)) @ cov_block @ np.ones(len(lag_idx))))
    rows.append({"lag": "cumulative", "beta": cum_b, "se": cum_se,
                  "ci_low": cum_b - 1.96 * cum_se, "ci_high": cum_b + 1.96 * cum_se})
    return pd.DataFrame(rows)
