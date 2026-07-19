from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig3r_volcanic_with_controls import (
    make_fig3r,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig3r_volcanic_with_controls_england.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig3r()
    assert pdf.exists() and pdf == OUT
    for k in ("Huaynaputina", "Tambora", "Pinatubo"):
        assert k in est
