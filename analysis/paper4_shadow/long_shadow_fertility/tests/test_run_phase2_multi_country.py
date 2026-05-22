from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.run_phase2_multi_country import run_all_phase2


def test_run_all_emits_phase2_figs():
    artifacts = run_all_phase2()
    for k in ("fig1", "fig2", "fig3", "fig4", "fig6", "fig1r", "fig2r", "fig3r"):
        assert k in artifacts and artifacts[k].exists()
