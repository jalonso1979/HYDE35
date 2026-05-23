from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig7v3_pooled_dl_irf_v3 import make_fig7v3

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig7v3_pooled_dl_irf_v3.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig7v3()
    assert pdf.exists() and pdf == OUT
    for c in ("lag", "beta", "se"):
        assert c in est.columns
