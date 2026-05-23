from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig8v2_pooled_vol_dl import (
    make_fig8v2,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig8v2_pooled_vol_dl.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig8v2()
    assert pdf.exists() and pdf == OUT
    assert {"level", "vol"}.issubset(set(est["regressor"].unique()))
