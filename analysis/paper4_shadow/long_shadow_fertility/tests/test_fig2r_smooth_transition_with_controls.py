from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig2r_smooth_transition_with_controls import (
    make_fig2r,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig2r_smooth_transition_with_controls_england.pdf")


def test_makes_pdf():
    pdf, png, fit = make_fig2r()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_M", "beta_T", "c", "theta"):
        assert k in fit
