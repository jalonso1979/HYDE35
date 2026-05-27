"""within_season_variance.py

Compute within-growing-season realized standard deviation of a climate variable.
This is the headline uncertainty proxy for Phase 10 Pillar C.

Paleo-data-density independent: uses only the values within the season window,
no ensemble spread required.
"""

import pandas as pd


def add_within_season_sd(df, var, season_months=(4, 9), unit_col="iso3",
                         year_col="year", month_col="month"):
    """Compute country-year sample SD of `var` over the months in season_months
    (inclusive).

    Parameters
    ----------
    df : pd.DataFrame
        Long monthly panel with columns [unit_col, year_col, month_col, var].
    var : str
        Name of the column to compute the SD on (e.g. 't_anom').
    season_months : (int, int)
        Inclusive month range. Default (4, 9) = April through September.
    unit_col : str
        Column identifying the cross-sectional unit (default "iso3").
    year_col : str
        Column identifying the year (default "year").
    month_col : str
        Column identifying the month (default "month").

    Returns
    -------
    pd.DataFrame
        Country-year panel with columns [unit_col, year_col, '<var>_within_season_sd'].
        Sample SD (ddof=1). Years where fewer than 2 months are present yield NaN.
    """
    lo, hi = season_months
    mask = (df[month_col] >= lo) & (df[month_col] <= hi)
    season = df.loc[mask, [unit_col, year_col, var]]
    out = (season.groupby([unit_col, year_col])[var]
                  .std(ddof=1)
                  .reset_index()
                  .rename(columns={var: f"{var}_within_season_sd"}))
    return out
