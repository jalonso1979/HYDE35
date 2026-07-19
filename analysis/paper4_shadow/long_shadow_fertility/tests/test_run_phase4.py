from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.run_phase4_pooled_iv import run_all_phase4


def test_run_all_emits_core_figures():
    artifacts = run_all_phase4()
    for k in ("fig7v2", "fig8v2", "fig11v2", "fig13"):
        assert k in artifacts and artifacts[k].exists()
