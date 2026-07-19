from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig3_volcanic_eventstudy import (
    make_fig3,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig3_volcanic_eventstudy_england.pdf")


def test_makes_pdf():
    pdf, png, est = make_fig3()
    assert pdf.exists() and pdf == OUT
    # est is dict by eruption_name -> DataFrame
    for k in ("Huaynaputina", "Tambora", "Pinatubo"):
        assert k in est
        assert {"h", "delta", "ci_low", "ci_high"}.issubset(est[k].columns)
