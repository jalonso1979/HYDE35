from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.run_phase3_lags_mortality_wages import run_all_phase3


def test_run_all_emits_six_figures():
    artifacts = run_all_phase3()
    for k in ("fig7", "fig8", "fig9", "fig10", "fig11", "fig12"):
        assert k in artifacts and artifacts[k].exists()
