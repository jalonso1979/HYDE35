"""Phase 7 Pillar B Fig 17 — France dept DL IRF figure."""


def test_make_fig17_produces_pdf(tmp_path, monkeypatch):
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig17_france_dept_dl_irf as mod
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, summary = mod.make_fig17()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert "spec1" in summary and "spec2" in summary
    assert len(summary["spec1"]["beta"]) >= 1
