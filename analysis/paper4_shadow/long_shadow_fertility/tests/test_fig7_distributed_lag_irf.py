from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig7_distributed_lag_irf import (
    make_fig7_dl_irf,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig7_distributed_lag_irf.pdf")


def test_makes_pdf():
    pdf, png, est_per_country = make_fig7_dl_irf()
    assert pdf.exists() and pdf == OUT
    assert set(est_per_country.keys()) == {"GBR", "FRA", "ITA", "SWE"}
