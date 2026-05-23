"""Tests for teleconnection panel builder (Phase 4 Task 2, BLOCKABLE).

Tests are skipped if the parquet output does not exist (NOAA NCEI may 403).
"""
import pytest
from pathlib import Path

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
           "teleconnection_panel.parquet")


@pytest.mark.skipif(not OUT.exists(), reason="Teleconnection data not fetched (Task 2 BLOCKED)")
def test_columns():
    import pandas as pd
    df = pd.read_parquet(OUT)
    for c in ("year", "nao", "amo", "enso"):
        assert c in df.columns


@pytest.mark.skipif(not OUT.exists(), reason="Teleconnection data not fetched (Task 2 BLOCKED)")
def test_year_coverage():
    import pandas as pd
    df = pd.read_parquet(OUT)
    assert df["year"].min() <= 1700
    assert df["year"].max() >= 1990
