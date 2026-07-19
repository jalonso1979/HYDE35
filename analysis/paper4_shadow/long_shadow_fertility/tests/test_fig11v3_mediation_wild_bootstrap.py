from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.figures.fig11v3_mediation_wild_bootstrap import make_fig11v3

OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/"
            "fig11v3_mediation_wild_bootstrap.pdf")


def test_makes_pdf():
    pdf, png, res = make_fig11v3()
    assert pdf.exists() and pdf == OUT
    for k in ("direct", "indirect", "total", "indirect_wild_cluster_se", "n_clusters"):
        assert k in res
