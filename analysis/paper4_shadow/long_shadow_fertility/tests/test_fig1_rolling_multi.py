from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig1_rolling_multi import make_fig1_multi

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig1_rolling_multi_country.pdf")


def test_makes_pdf():
    pdf, png, est_per_country = make_fig1_multi()
    assert pdf.exists()
    assert pdf == OUT
    assert set(est_per_country.keys()) == {"GBR", "FRA", "ITA", "SWE"}
