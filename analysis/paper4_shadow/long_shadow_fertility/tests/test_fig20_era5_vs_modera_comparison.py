"""Phase 8 Pillar B4c+d: Fig 20 ERA5 vs ModE-RA comparison."""
import pytest


def test_make_fig20_produces_pdf(tmp_path, monkeypatch):
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.figures import (
        fig20_era5_vs_modera_comparison as mod,
    )
    from pathlib import Path
    if not Path(mod.ERA5_PATH).exists():
        pytest.skip("ERA5 country panel not built")
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, stats = mod.make_fig20()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert "corr_T" in stats and "corr_P" in stats
    assert -1.0 <= stats["corr_T"] <= 1.0
    assert -1.0 <= stats["corr_P"] <= 1.0
