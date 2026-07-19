from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig1v3_rolling_7country import make_fig1v3

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig1v3_rolling_7country.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig1v3()
    assert pdf.exists() and pdf == OUT
    assert set(est.keys()) == {
        "GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP",
        "NOR", "DNK", "FIN", "ISL", "CHE",
    }
