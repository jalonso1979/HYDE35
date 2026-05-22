from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.run_phase1_england import run_all

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")


def test_run_all_emits_three_figures():
    artifacts = run_all()
    assert "fig1" in artifacts and artifacts["fig1"].exists()
    assert "fig2" in artifacts and artifacts["fig2"].exists()
    assert "fig3" in artifacts and artifacts["fig3"].exists()
