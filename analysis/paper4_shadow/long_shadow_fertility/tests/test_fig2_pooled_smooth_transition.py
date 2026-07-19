from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig2_pooled_smooth_transition import (
    make_fig2_pooled,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig2_pooled_smooth_transition.pdf")


def test_makes_pdf():
    pdf, png, fit = make_fig2_pooled()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_M", "beta_T", "c", "theta", "n_countries"):
        assert k in fit
