from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig9_joint_fertility_mortality import (
    make_fig9_joint,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig9_joint_fertility_mortality.pdf")


def test_makes_pdf():
    pdf, png, res = make_fig9_joint()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_y1", "beta_y2", "wald_eq_pvalue"):
        assert k in res
