"""ERA5 climate shock construction for Paper 3."""
import numpy as np
import pandas as pd


def compute_anomalies(df: pd.DataFrame, variable: str, rolling_window: int = 30) -> pd.DataFrame:
    df = df.sort_values("year").copy()
    rolling_mean = df[variable].rolling(window=rolling_window, center=True, min_periods=rolling_window // 2).mean()
    df[f"{variable}_anomaly"] = df[variable] - rolling_mean
    return df


def compute_volatility(df: pd.DataFrame, variable: str, rolling_window: int = 10) -> pd.DataFrame:
    df = df.sort_values("year").copy()
    df[f"{variable}_volatility"] = df[variable].rolling(window=rolling_window, center=True, min_periods=rolling_window // 2).std()
    return df


def build_climate_shock_panel(era5_panel: pd.DataFrame, climate_vars: list[str], entity_col: str = "region", rolling_window: int = 30) -> pd.DataFrame:
    parts = []
    for entity, grp in era5_panel.groupby(entity_col):
        df = grp.copy()
        for var in climate_vars:
            df = compute_anomalies(df, var, rolling_window=rolling_window)
            df = compute_volatility(df, var, rolling_window=min(10, rolling_window // 3))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
