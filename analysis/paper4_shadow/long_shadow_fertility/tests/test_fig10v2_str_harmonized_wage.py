from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig10v2_str_harmonized_wage import make_fig10v2

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig10v2_str_harmonized_wage.pdf")


def test_makes_pdf():
    pdf, png, fit = make_fig10v2()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_M", "beta_T", "c", "theta", "n_countries"):
        assert k in fit
