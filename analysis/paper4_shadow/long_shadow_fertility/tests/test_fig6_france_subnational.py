from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig6_france_subnational import (
    make_fig6_france_subnational,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig6_france_subnational.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig6_france_subnational()
    assert pdf.exists() and pdf == OUT
