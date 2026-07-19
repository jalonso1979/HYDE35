"""Phase 7 Pillar C3 + Phase 10 B5b: Fig 13v2 extended volcanic IV with AR CI."""
import numpy as np


def test_make_fig13v2_produces_pdf_or_skips(tmp_path, monkeypatch):
    """If Pillar C2 returned BLOCKED, figure builder skips cleanly.

    Phase 10 B5b: also checks that ar_ci_lo/ar_ci_hi keys are present in
    iv_summary. Under weak identification (F < 1), the AR CI may be unbounded
    (nan, nan) — that is a valid outcome.
    """
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
    # Phase 10: AR CI keys must be present
    assert "ar_ci_lo" in iv_summary, "ar_ci_lo missing from iv_summary"
    assert "ar_ci_hi" in iv_summary, "ar_ci_hi missing from iv_summary"
    # Under weak IV (F < 1), AR CI is unbounded (nan) — that is acceptable
    lo = iv_summary["ar_ci_lo"]
    hi = iv_summary["ar_ci_hi"]
    if np.isfinite(lo) and np.isfinite(hi):
        # If bounded, it must contain the 2SLS point estimate
        assert lo <= iv_summary["beta"] <= hi, (
            f"AR CI [{lo:.3f}, {hi:.3f}] does not contain 2SLS β={iv_summary['beta']:.3f}"
        )
