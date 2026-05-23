from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig16_phase_progression import make_fig16

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig16_phase_progression.pdf")


def test_makes_pdf():
    pdf, png = make_fig16()
    assert pdf.exists() and pdf == OUT
