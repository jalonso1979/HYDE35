import pandas as pd
import numpy as np
from analysis.paper4_shadow.long_shadow_fertility.data.within_season_variance import add_within_season_sd


def test_within_season_sd_basic():
    """For a synthetic country where t_anom = month for months 4..9, the SD
    should match the sample SD of [4,5,6,7,8,9]."""
    rows = []
    for iso3 in ["AAA", "BBB"]:
        for year in [1900, 1901]:
            for month in range(1, 13):
                rows.append(dict(iso3=iso3, year=year, month=month,
                                 t_anom=float(month + (1 if iso3 == "BBB" else 0))))
    df = pd.DataFrame(rows)
    out = add_within_season_sd(df, var="t_anom", season_months=(4, 9))
    expected = float(np.std([4, 5, 6, 7, 8, 9], ddof=1))
    assert set(out.columns) >= {"iso3", "year", "t_anom_within_season_sd"}
    row = out[(out.iso3 == "AAA") & (out.year == 1900)].iloc[0]
    assert abs(row["t_anom_within_season_sd"] - expected) < 1e-9
    # Country BBB is shifted by +1 in every month, so SD should be identical
    row_b = out[(out.iso3 == "BBB") & (out.year == 1900)].iloc[0]
    assert abs(row_b["t_anom_within_season_sd"] - expected) < 1e-9


def test_within_season_sd_handles_missing_months():
    """If a year has < 2 months in season, SD should be NaN."""
    rows = [
        dict(iso3="AAA", year=1900, month=4, t_anom=5.0),
        dict(iso3="AAA", year=1900, month=1, t_anom=1.0),  # outside season
    ]
    df = pd.DataFrame(rows)
    out = add_within_season_sd(df, var="t_anom", season_months=(4, 9))
    row = out[(out.iso3 == "AAA") & (out.year == 1900)].iloc[0]
    assert pd.isna(row["t_anom_within_season_sd"])
