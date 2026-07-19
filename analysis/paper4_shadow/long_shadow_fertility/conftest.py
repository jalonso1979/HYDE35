"""Shared pytest fixtures for long_shadow_fertility tests."""
from pathlib import Path
import pytest

ANALYSIS_ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis")

@pytest.fixture(scope="session")
def data_dir() -> Path:
    return ANALYSIS_ROOT / "data" / "long_shadow_fertility"

@pytest.fixture(scope="session")
def figure_dir() -> Path:
    return ANALYSIS_ROOT / "figures" / "long_shadow_fertility"
