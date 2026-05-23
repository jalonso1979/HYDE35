from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig11v2_mediation_cluster_bootstrap import (
    make_fig11v2,
)

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig11v2_mediation_cluster_bootstrap.pdf")


def test_makes_pdf():
    pdf, png, res = make_fig11v2()
    assert pdf.exists() and pdf == OUT
    for k in ("direct", "indirect", "total", "indirect_cluster_se"):
        assert k in res
