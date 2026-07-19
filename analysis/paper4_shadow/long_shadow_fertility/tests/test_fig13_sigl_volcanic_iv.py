from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig13_sigl_volcanic_iv import (
    make_fig13,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig13_sigl_volcanic_iv.pdf")


def test_makes_pdf():
    pdf, png, res = make_fig13()
    assert pdf.exists() and pdf == OUT
    for k in ("beta", "se", "first_stage_f", "ar_pvalue"):
        assert k in res
