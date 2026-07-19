"""ModE-RA ensemble standard deviation aggregation.

Aggregates ensstd_t and ensstd_p (temperature and precipitation ensemble spread
from ModE-RA paleo reanalysis) to growing-season country-year means.

These serve as a comparator to within-season realized SD: when the two measures
diverge, it indicates whether climate uncertainty effects on fertility are real
or a paleo data-density artifact.
"""

import pandas as pd


def aggregate_ensstd_to_country_year(df, season_months=(4, 9),
                                     unit_col="iso3", year_col="year",
                                     month_col="month"):
    """Compute growing-season mean of `ensstd_t` and `ensstd_p` per country-year.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly long panel with columns [unit_col, year_col, month_col,
        ensstd_t, ensstd_p].
    season_months : (int, int)
        Inclusive month range. Default (4, 9) = April through September.
    unit_col : str
        Column name for the geographic unit (default "iso3").
    year_col : str
        Column name for the year (default "year").
    month_col : str
        Column name for the month (default "month").

    Returns
    -------
    pd.DataFrame
        Country-year panel with columns [unit_col, year_col,
        ensstd_t_growing, ensstd_p_growing].
        Simple within-year average over months in season.
    """
    lo, hi = season_months
    season = df[(df[month_col] >= lo) & (df[month_col] <= hi)]
    out = (season.groupby([unit_col, year_col])[["ensstd_t", "ensstd_p"]]
                  .mean()
                  .rename(columns={"ensstd_t": "ensstd_t_growing",
                                   "ensstd_p": "ensstd_p_growing"})
                  .reset_index())
    return out
