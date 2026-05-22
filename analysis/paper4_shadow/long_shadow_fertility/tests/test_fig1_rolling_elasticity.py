from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig1_rolling_elasticity import (
    make_fig1,
)

OUT_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def test_makes_pdf_and_png():
    pdf, png = make_fig1()
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert png.exists() and png.suffix == ".png"
    assert pdf.parent == OUT_DIR


def test_returns_estimates_dataframe():
    pdf, png, est = make_fig1(return_estimates=True)
    assert {"center_year", "beta", "ci_low", "ci_high"}.issubset(est.columns)
    assert len(est) > 100  # at least 100 windows over ~480 years
