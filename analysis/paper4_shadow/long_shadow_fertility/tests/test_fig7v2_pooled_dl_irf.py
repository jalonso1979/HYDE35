from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig7v2_pooled_dl_irf import (
    make_fig7v2,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig7v2_pooled_dl_irf.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig7v2()
    assert pdf.exists() and pdf == OUT
    for c in ("lag", "beta", "se"):
        assert c in est.columns
