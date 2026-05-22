"""End-to-end Phase 1 pipeline for the Long Shadow on Fertility — England.

Steps
-----
1. Build fertility, climate, GDP panels.
2. Assemble unified panel.
3. Produce Fig 1 (rolling-window), Fig 2 (smooth transition + within-era), Fig 3 (volcanic).
"""
from __future__ import annotations
from pathlib import Path

from analysis.paper4_shadow.long_shadow_fertility.data.build_england_fertility_annual import (
    build_england_fertility_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_maddison_england import (
    build_maddison_england,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_england_climate_annual import (
    build_england_climate_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel import (
    assemble_england_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.figures.fig1_rolling_elasticity import make_fig1
from analysis.paper4_shadow.long_shadow_fertility.figures.fig2_smooth_transition import make_fig2
from analysis.paper4_shadow.long_shadow_fertility.figures.fig3_volcanic_eventstudy import make_fig3


def run_all() -> dict[str, Path]:
    build_england_fertility_annual(write=True)
    build_maddison_england(write=True)
    build_england_climate_annual(write=True)
    assemble_england_panel(write=True)
    f1_pdf, _ = make_fig1()
    f2_pdf, _, _ = make_fig2()
    f3_pdf, _, _ = make_fig3()
    return {"fig1": f1_pdf, "fig2": f2_pdf, "fig3": f3_pdf}


if __name__ == "__main__":
    artifacts = run_all()
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
