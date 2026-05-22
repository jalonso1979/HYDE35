from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig4_country_decade_heatmap import (
    make_fig4_heatmap,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig4_country_decade_heatmap.pdf")


def test_makes_pdf():
    pdf, png, mat = make_fig4_heatmap()
    assert pdf.exists() and pdf == OUT
    assert mat.shape[0] >= 4
