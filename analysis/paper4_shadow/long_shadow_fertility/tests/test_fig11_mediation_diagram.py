from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig11_mediation_diagram import (
    make_fig11_mediation,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig11_mediation_diagram.pdf")


def test_makes_pdf():
    pdf, png, res = make_fig11_mediation()
    assert pdf.exists() and pdf == OUT
    for k in ("direct", "indirect", "total", "phi", "delta"):
        assert k in res
