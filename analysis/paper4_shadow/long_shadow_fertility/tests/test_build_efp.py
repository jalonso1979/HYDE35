import pytest
from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.data.build_efp import (
    build_efp,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
            "efp_province_decade.parquet")
CACHE = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/_efp_raw")


@pytest.mark.skipif(not (CACHE.exists() and any(CACHE.glob("*.csv"))),
                    reason="EFP source not cached locally (Task 13 expected BLOCKED)")
def test_efp_loads():
    df = build_efp()
    assert "province" in df.columns
    assert "decade" in df.columns
    assert "If" in df.columns
    assert df["decade"].min() <= 1840
    assert df["decade"].max() >= 1960
