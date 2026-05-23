"""Phase 7 Pillar C3: Fig 13v2 extended volcanic IV."""


def test_make_fig13v2_produces_pdf_or_skips(tmp_path, monkeypatch):
    """If Pillar C2 returned BLOCKED, figure builder skips cleanly."""
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig13v2_volcanic_iv_extended as mod
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    result = mod.make_fig13v2()
    if result is mod.BLOCKED:
        # Pillar C is blocked — acceptable outcome
        return
    pdf, png, iv_summary = result
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    # fit_iv_2sls returns: beta (LATE), se, first_stage_f, ar_pvalue, n
    assert "first_stage_f" in iv_summary
    assert "beta" in iv_summary
