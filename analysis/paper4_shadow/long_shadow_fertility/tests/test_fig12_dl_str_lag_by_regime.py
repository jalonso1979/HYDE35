from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig12_dl_str_lag_by_regime import (
    make_fig12_dl_regime,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig12_dl_str_lag_by_regime.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig12_dl_regime()
    assert pdf.exists() and pdf == OUT
    for k in ("Malthus", "Modern"):
        assert k in est
