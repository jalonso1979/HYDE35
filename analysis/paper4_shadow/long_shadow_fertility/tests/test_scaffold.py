"""Smoke test: subpackage imports cleanly and output directories exist."""
from pathlib import Path

def test_subpackage_imports():
    import analysis.paper4_shadow.long_shadow_fertility as lsf
    assert lsf.__doc__ is not None

def test_output_directories_exist():
    root = Path("/Volumes/BIGDATA/HYDE35/analysis")
    assert (root / "data" / "long_shadow_fertility").exists()
    assert (root / "figures" / "long_shadow_fertility").exists()
