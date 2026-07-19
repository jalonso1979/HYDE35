from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig1r_rolling_with_controls import (
    make_fig1r,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig1r_rolling_with_controls_england.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig1r()
    assert pdf.exists() and pdf == OUT
    assert {"center_year", "beta", "ci_low", "ci_high"}.issubset(est.columns)
