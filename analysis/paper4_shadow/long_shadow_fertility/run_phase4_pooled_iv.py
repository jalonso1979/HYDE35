"""End-to-end Phase 4: build Sigl + teleconnection (best-effort), run 5 figures."""
from __future__ import annotations
from pathlib import Path

from analysis.paper4_shadow.long_shadow_fertility.data.build_sigl_volcanic_panel import (
    build_sigl_volcanic_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.figures.fig7v2_pooled_dl_irf import make_fig7v2
from analysis.paper4_shadow.long_shadow_fertility.figures.fig8v2_pooled_vol_dl import make_fig8v2
from analysis.paper4_shadow.long_shadow_fertility.figures.fig11v2_mediation_cluster_bootstrap import make_fig11v2
from analysis.paper4_shadow.long_shadow_fertility.figures.fig13_sigl_volcanic_iv import make_fig13


def run_all_phase4() -> dict[str, Path]:
    build_sigl_volcanic_panel(write=True)
    try:
        from analysis.paper4_shadow.long_shadow_fertility.data.build_teleconnection_panel import (
            build_teleconnection_panel,
        )
        build_teleconnection_panel(write=True)
    except Exception as exc:
        print(f"[teleconnection] skipped: {exc}")

    artifacts: dict[str, Path] = {}
    artifacts["fig7v2"], _, _ = make_fig7v2()
    artifacts["fig8v2"], _, _ = make_fig8v2()
    artifacts["fig11v2"], _, _ = make_fig11v2()
    artifacts["fig13"], _, _ = make_fig13()
    try:
        from analysis.paper4_shadow.long_shadow_fertility.figures.fig14_teleconnection_iv import (
            make_fig14,
        )
        artifacts["fig14"], _, _ = make_fig14()
    except Exception as exc:
        print(f"[fig14 teleconnection] skipped: {exc}")
    return artifacts


if __name__ == "__main__":
    artifacts = run_all_phase4()
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
