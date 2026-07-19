"""Phase 7 Pillar B6 Fig 19 — France dept system LP IRF."""


def test_make_fig19_produces_pdf(tmp_path, monkeypatch):
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig19_france_dept_system_lp as mod
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, lp_df = mod.make_fig19()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert "h" in lp_df.columns and "beta" in lp_df.columns
    assert len(lp_df) >= 1
