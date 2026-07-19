from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig15_triple_sur import make_fig15

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig15_triple_sur.pdf")


def test_makes_pdf():
    pdf, png, res = make_fig15()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_y1", "beta_y2", "beta_y3", "wald_eq12_pvalue"):
        assert k in res
