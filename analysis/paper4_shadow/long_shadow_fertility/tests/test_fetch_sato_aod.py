"""Phase 7 Pillar C1: Sato AOD fetcher with BLOCKED-graceful failure."""
from pathlib import Path
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.fetch_sato_aod import (
    fetch_sato_aod,
    BLOCKED,
)


def test_fetch_returns_dataframe_or_blocked(tmp_path):
    """Live fetch attempt; either returns DataFrame or BLOCKED sentinel."""
    result = fetch_sato_aod(cache_dir=tmp_path, raise_on_failure=False)
    assert result is BLOCKED or isinstance(result, pd.DataFrame)
    if isinstance(result, pd.DataFrame):
        assert {"year", "aod_max"}.issubset(result.columns)
        assert (result["aod_max"] >= 0).all()
        assert result["year"].min() >= 1850
        assert result["year"].max() >= 1900


def test_fetch_with_bad_url_returns_blocked(tmp_path, monkeypatch):
    """When URL is unreachable, fetcher returns BLOCKED sentinel."""
    import analysis.paper4_shadow.long_shadow_fertility.data.fetch_sato_aod as mod
    monkeypatch.setattr(mod, "SATO_URL", "https://nonexistent.invalid/tau.txt")
    result = mod.fetch_sato_aod(cache_dir=tmp_path, raise_on_failure=False)
    assert result is BLOCKED
