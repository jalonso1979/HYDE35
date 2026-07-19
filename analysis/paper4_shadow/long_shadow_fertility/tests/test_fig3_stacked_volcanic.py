from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig3_stacked_volcanic import (
    make_fig3_stacked,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig3_stacked_volcanic.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig3_stacked()
    assert pdf.exists() and pdf == OUT
    for v in ("Tambora", "Krakatoa", "Pinatubo"):
        assert v in est
