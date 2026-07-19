"""Phase 9 Pillar A2: Fig 20v2 ERA5 vs ModE-RA matched-aggregation comparison."""
import pytest


def test_make_fig20v2_produces_pdf(tmp_path, monkeypatch):
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.figures import (
        fig20v2_era5_vs_modera_matched as mod,
    )
    from pathlib import Path
    if not Path(mod.ERA5_V2_PATH).exists():
        pytest.skip("ERA5 v2 panel not built")
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, stats = mod.make_fig20v2()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert "corr_T" in stats and "corr_P" in stats
    assert -1.0 <= stats["corr_T"] <= 1.0
