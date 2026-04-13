"""Seasonality computation for Paper 4.

Two climate measures:
1. Intra-annual seasonality (sigma_s): max - min of monthly values within a
   year. The "Matranga channel" driving agricultural system selection.
2. Historical seasonality proxy: rolling SD of annual temperature from
   PAGES 2k reconstructions (monthly data unavailable for historical periods).
"""
from __future__ import annotations

import pandas as pd


def compute_intra_annual_seasonality_era5(
    monthly: pd.DataFrame,
    entity_col: str = "region",
) -> pd.DataFrame:
    """Compute intra-annual seasonality from monthly ERA5 data.

    Parameters
    ----------
    monthly : DataFrame
        Must contain [entity_col, year, month, temperature_c].
        Optionally contains precipitation_mm.
    entity_col : str
        Column identifying spatial units (default "region").

    Returns
    -------
    DataFrame with [entity_col, year, seasonality_temp, seasonality_precip*].
    """
    group = [entity_col, "year"]
    agg = {"temperature_c": ["max", "min"]}
    has_precip = "precipitation_mm" in monthly.columns
    if has_precip:
        agg["precipitation_mm"] = ["max", "min"]

    stats = monthly.groupby(group).agg(agg)
    result = pd.DataFrame({
        entity_col: stats.index.get_level_values(0),
        "year": stats.index.get_level_values(1),
        "seasonality_temp": (
            stats[("temperature_c", "max")].values
            - stats[("temperature_c", "min")].values
        ),
    })
    if has_precip:
        result["seasonality_precip"] = (
            stats[("precipitation_mm", "max")].values
            - stats[("precipitation_mm", "min")].values
        )
    return result.reset_index(drop=True)


def compute_historical_seasonality_proxy(
    annual: pd.DataFrame,
    window: int = 50,
    entity_col: str = "region",
) -> pd.DataFrame:
    """Compute historical seasonality proxy as rolling SD of annual temperature.

    Parameters
    ----------
    annual : DataFrame
        Must contain [entity_col, year, temperature_c].
    window : int
        Rolling window size in years.
    entity_col : str
        Column identifying spatial units.

    Returns
    -------
    DataFrame with [entity_col, year, seasonality_proxy].
    """
    annual = annual.sort_values([entity_col, "year"]).copy()
    annual["seasonality_proxy"] = (
        annual.groupby(entity_col)["temperature_c"]
        .transform(lambda s: s.rolling(window, min_periods=window).std())
    )
    return annual[[entity_col, "year", "seasonality_proxy"]].copy()


def build_seasonality_panel(
    annual_climate: pd.DataFrame,
    era5_monthly: pd.DataFrame | None = None,
    entity_col: str = "region",
    historical_window: int = 50,
    historical_period: tuple[int, int] = (1, 1850),
) -> pd.DataFrame:
    """Build cross-sectional seasonality endowments (one row per region).

    Computes long-run mean of the historical proxy within the specified
    historical period. Optionally merges ERA5 direct seasonality.

    Parameters
    ----------
    annual_climate : DataFrame
        Annual data with [entity_col, year, temperature_c].
    era5_monthly : DataFrame or None
        Monthly ERA5 data; if provided, adds mean seasonality_temp column.
    entity_col : str
        Column identifying spatial units.
    historical_window : int
        Rolling window for historical proxy.
    historical_period : tuple
        (start_year, end_year) for averaging the proxy.

    Returns
    -------
    DataFrame with one row per region.
    """
    proxy = compute_historical_seasonality_proxy(
        annual_climate, window=historical_window, entity_col=entity_col,
    )
    start, end = historical_period
    mask = (proxy["year"] >= start) & (proxy["year"] <= end)
    panel = (
        proxy.loc[mask]
        .groupby(entity_col)["seasonality_proxy"]
        .mean()
        .reset_index()
        .rename(columns={"seasonality_proxy": "hist_seasonality_proxy"})
    )

    if era5_monthly is not None:
        era5_seas = compute_intra_annual_seasonality_era5(
            era5_monthly, entity_col=entity_col,
        )
        era5_mean = (
            era5_seas.groupby(entity_col)["seasonality_temp"]
            .mean()
            .reset_index()
            .rename(columns={"seasonality_temp": "era5_seasonality_temp"})
        )
        panel = panel.merge(era5_mean, on=entity_col, how="outer")

    return panel
