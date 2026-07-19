import pandas as pd
import numpy as np
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)


def test_columns():
    df = build_real_wage_panel_v2()
    for c in ("iso3", "year", "log_real_wage", "source", "harmonized"):
        assert c in df.columns


def test_no_unit_discontinuity_at_1914():
    """After harmonization, Allen-era and Maddison-era log values should be
    in the same scale (no >2 log-unit jump at the 1913/1914 boundary)."""
    df = build_real_wage_panel_v2().set_index(["iso3", "year"])
    for iso in ("GBR", "FRA"):
        if (iso, 1913) in df.index and (iso, 1914) in df.index:
            jump = abs(df.loc[(iso, 1913), "log_real_wage"] - df.loc[(iso, 1914), "log_real_wage"])
            assert jump < 2.0, f"{iso}: unit discontinuity at 1914 = {jump:.2f} log units"


def test_harmonized_flag():
    """Pre-1914 rows should be flagged harmonized=True (rescaled from Allen);
    post-1913 rows from Maddison are harmonized=False (already in target scale)."""
    df = build_real_wage_panel_v2()
    pre = df.loc[df["year"] < 1914]
    post = df.loc[df["year"] >= 1914]
    assert pre["harmonized"].any()
    assert (post["harmonized"] == False).all()
