import pytest
from pathlib import Path

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig5_efp_cross_section.pdf")
EFP_PARQUET = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
                    "efp_province_decade.parquet")


@pytest.mark.skipif(not EFP_PARQUET.exists(), reason="EFP data unavailable (Task 13 blocked)")
def test_makes_pdf():
    from analysis.paper4_shadow.long_shadow_fertility.figures.fig5_efp_cross_section import (
        make_fig5_efp,
    )
    pdf, png, fit = make_fig5_efp()
    assert pdf.exists() and pdf == OUT
