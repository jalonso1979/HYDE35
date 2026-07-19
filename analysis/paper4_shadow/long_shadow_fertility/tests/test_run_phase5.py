from pathlib import Path
from analysis.paper4_shadow.long_shadow_fertility.run_phase5_gapfix_expansion_sur import (
    run_all_phase5,
)


def test_run_all_emits_six_figures():
    artifacts = run_all_phase5()
    for k in ("fig1v3", "fig7v3", "fig10v2", "fig11v3", "fig15", "fig16"):
        assert k in artifacts and artifacts[k].exists()
