"""Phase 8 Pillar B1: Fig 16 with dynamic phase data."""


def test_make_fig16_produces_pdf(tmp_path, monkeypatch):
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig16_phase_progression as mod
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    pdf, png, data = mod.make_fig16()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert len(data) == 8  # 5 historical + 1 P5 cum + 1 P5 indirect + 1 P7 W-share
    p7_row = data[-1]
    assert "P7" in p7_row[0]
    assert 0.0 <= p7_row[1] <= 1.0  # FEVD share bounded


def test_p5_indirect_is_live_value():
    """P5 indirect bar reflects current fig11v3 output (~-0.174 post-Phase-7)."""
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig16_phase_progression as mod
    data = mod._collect_phase_data()
    p5_indirect = [r for r in data if "Indirect" in r[0]][0]
    assert -0.25 < p5_indirect[1] < -0.10
