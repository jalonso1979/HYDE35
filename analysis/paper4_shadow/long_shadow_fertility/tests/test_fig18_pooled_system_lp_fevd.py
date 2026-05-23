"""Phase 7 Pillar E2: Fig 18 pooled system LP-FEVD."""


def test_make_fig18_produces_pdf(tmp_path, monkeypatch):
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig18_pooled_system_lp_fevd as mod
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, summary = mod.make_fig18()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert "fevd_F" in summary
    assert "irf_F_T" in summary
    assert len(summary["irf_F_T"]) >= 1
