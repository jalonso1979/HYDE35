from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig8_volatility_treatment import (
    make_fig8_vol,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig8_volatility_treatment.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig8_vol()
    assert pdf.exists() and pdf == OUT
    assert set(est.keys()) == {"GBR", "FRA", "ITA", "SWE"}
