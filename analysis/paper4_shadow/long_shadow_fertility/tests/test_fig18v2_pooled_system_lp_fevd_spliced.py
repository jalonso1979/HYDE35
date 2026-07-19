"""Phase 9 Pillar A4: Fig 18v2 LP-FEVD robustness with spliced climate."""
import pytest


def test_make_fig18v2_produces_pdf(tmp_path, monkeypatch):
    pytest.importorskip("xarray")
    from analysis.paper4_shadow.long_shadow_fertility.figures import (
        fig18v2_pooled_system_lp_fevd_spliced as mod,
    )
    from pathlib import Path
    spliced_path = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/country_climate_spliced.parquet")
    if not spliced_path.exists():
        pytest.skip("Spliced panel not built")
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, summary = mod.make_fig18v2()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert "fevd_F_at_h15" in summary
    assert "T" in summary["fevd_F_at_h15"]
