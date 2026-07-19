"""Pooled DL with two regressors (level + volatility) + FE + cluster SE."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _lag(df: pd.DataFrame, var: str, lags: int, unit_col: str) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values([unit_col, "year"]).copy()
    cols = []
    for k in range(lags + 1):
        c = f"{var}_lag{k}"
        df[c] = df.groupby(unit_col)[var].shift(k)
        cols.append(c)
    return df, cols


def fit_pooled_volatility_dl(
    df: pd.DataFrame,
    y: str,
    x_level: str,
    x_vol: str,
    lags: int = 3,
    unit_col: str = "iso3",
    controls: Sequence[str] | None = None,
    year_fe: bool = True,
) -> pd.DataFrame:
    df_lag, lvl_cols = _lag(df, x_level, lags, unit_col)
    df_lag, vol_cols = _lag(df_lag, x_vol, lags, unit_col)
    ctrl_cols = list(controls or [])
    keep = [y, unit_col, "year"] + lvl_cols + vol_cols + ctrl_cols
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
        sub[lvl_cols + vol_cols].astype(float),
        unit_dums, year_dums,
        sub[ctrl_cols].astype(float) if ctrl_cols else pd.DataFrame(index=sub.index),
    ], axis=1))
    cluster = sub[unit_col].astype("category").cat.codes.to_numpy()
    res = sm.OLS(sub[y].astype(float).to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )
    rows = []
    base = 1
    for k in range(lags + 1):
        b = float(res.params[base + k]); s = float(res.bse[base + k])
        rows.append({"regressor": "level", "lag": k, "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    base = 1 + (lags + 1)
    for k in range(lags + 1):
        b = float(res.params[base + k]); s = float(res.bse[base + k])
        rows.append({"regressor": "vol", "lag": k, "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    return pd.DataFrame(rows)
