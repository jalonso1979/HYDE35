import pytest
from pathlib import Path

TELE = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
             "teleconnection_panel.parquet")
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig14_teleconnection_iv.pdf")


@pytest.mark.skipif(not TELE.exists(), reason="Teleconnection data unavailable (Task 2 blocked)")
def test_makes_pdf():
    from analysis.paper4_shadow.long_shadow_fertility.figures.fig14_teleconnection_iv import (
        make_fig14,
    )
    pdf, png, res = make_fig14()
    assert pdf.exists() and pdf == OUT
