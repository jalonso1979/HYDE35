import pandas as pd
import numpy as np
from long_shadow_fertility.data.modera_ensstd import aggregate_ensstd_to_country_year


def test_growing_season_mean():
    rows = []
    for iso3 in ["AAA"]:
        for year in [1900]:
            for month in range(1, 13):
                rows.append(dict(iso3=iso3, year=year, month=month,
                                 ensstd_t=0.5 + 0.1 * month,
                                 ensstd_p=1.0))
    df = pd.DataFrame(rows)
    out = aggregate_ensstd_to_country_year(df, season_months=(4, 9))
    expected_t = 0.5 + 0.1 * float(np.mean([4, 5, 6, 7, 8, 9]))
    row = out.iloc[0]
    assert abs(row["ensstd_t_growing"] - expected_t) < 1e-9
    assert abs(row["ensstd_p_growing"] - 1.0) < 1e-9
    assert set(out.columns) >= {"iso3", "year", "ensstd_t_growing", "ensstd_p_growing"}


def test_multiple_countries_and_years():
    rows = []
    for iso3 in ["AAA", "BBB"]:
        for year in [1900, 1901]:
            for month in range(4, 10):
                rows.append(dict(iso3=iso3, year=year, month=month,
                                 ensstd_t=2.0, ensstd_p=3.0))
    df = pd.DataFrame(rows)
    out = aggregate_ensstd_to_country_year(df)
    assert len(out) == 4  # 2 countries x 2 years
    assert (out["ensstd_t_growing"] == 2.0).all()
    assert (out["ensstd_p_growing"] == 3.0).all()
