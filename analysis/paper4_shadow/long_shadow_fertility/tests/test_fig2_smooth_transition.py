from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig2_smooth_transition import (
    make_fig2,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig2_smooth_transition_england.pdf")


def test_makes_pdf():
    pdf, png, fit = make_fig2()
    assert pdf.exists() and pdf == OUT
    for k in ("beta_M", "beta_T", "c"):
        assert k in fit
    # Fig 2 is now a two-panel figure (STR on left, within-era OLS on right).
    for k in ("beta_M_within_era", "beta_T_within_era", "n_malthus", "n_modern"):
        assert k in fit
