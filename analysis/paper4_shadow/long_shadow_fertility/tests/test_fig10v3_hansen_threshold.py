"""Phase 9 Pillar B2: Fig 10v3 Hansen threshold regression figure."""


def test_make_fig10v3_produces_pdf(tmp_path, monkeypatch):
    from analysis.paper4_shadow.long_shadow_fertility.figures import fig10v3_hansen_threshold as mod
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    monkeypatch.setattr(mod, "N_BOOT", 50)  # speed up test
    pdf, png, res = mod.make_fig10v3()
    assert pdf.exists() and pdf.suffix == ".pdf"
    for k in ("beta_M", "beta_T", "c_hat", "sup_wald_pvalue"):
        assert k in res
