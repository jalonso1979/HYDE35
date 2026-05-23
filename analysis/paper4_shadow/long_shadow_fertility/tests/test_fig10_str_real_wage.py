from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig10_str_real_wage import (
    make_fig10_str_wage,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig10_str_real_wage.pdf")


def test_makes_pdf():
    pdf, png, fit = make_fig10_str_wage()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_M", "beta_T", "c", "theta", "n_countries"):
        assert k in fit
