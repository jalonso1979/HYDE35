"""End-to-end Phase 2 pipeline. EFP (Fig 5) is best-effort and skipped if data absent."""
from __future__ import annotations
from pathlib import Path

from analysis.paper4_shadow.long_shadow_fertility.data.build_maddison_multicountry import build_maddison_multicountry
from analysis.paper4_shadow.long_shadow_fertility.data.build_france_annual import build_france_annual
from analysis.paper4_shadow.long_shadow_fertility.data.build_italy_annual import build_italy_annual
from analysis.paper4_shadow.long_shadow_fertility.data.build_sweden_annual import build_sweden_annual
from analysis.paper4_shadow.long_shadow_fertility.data.build_france_dept_annual import build_france_dept_annual
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_climate_annual import build_country_climate_annual
from analysis.paper4_shadow.long_shadow_fertility.data.build_war_panel import build_war_panel
from analysis.paper4_shadow.long_shadow_fertility.data.build_pandemic_panel import build_pandemic_panel
from analysis.paper4_shadow.long_shadow_fertility.data.build_emdat_panel import build_emdat_panel
from analysis.paper4_shadow.long_shadow_fertility.data.build_climate_extremes import build_climate_extremes
from analysis.paper4_shadow.long_shadow_fertility.data.build_controls_panel import build_controls_panel
from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import assemble_panel_multi

from analysis.paper4_shadow.long_shadow_fertility.figures.fig1_rolling_multi import make_fig1_multi
from analysis.paper4_shadow.long_shadow_fertility.figures.fig2_pooled_smooth_transition import make_fig2_pooled
from analysis.paper4_shadow.long_shadow_fertility.figures.fig3_stacked_volcanic import make_fig3_stacked
from analysis.paper4_shadow.long_shadow_fertility.figures.fig4_country_decade_heatmap import make_fig4_heatmap
from analysis.paper4_shadow.long_shadow_fertility.figures.fig6_france_subnational import make_fig6_france_subnational
from analysis.paper4_shadow.long_shadow_fertility.figures.fig1r_rolling_with_controls import make_fig1r
from analysis.paper4_shadow.long_shadow_fertility.figures.fig2r_smooth_transition_with_controls import make_fig2r
from analysis.paper4_shadow.long_shadow_fertility.figures.fig3r_volcanic_with_controls import make_fig3r


def run_all_phase2() -> dict[str, Path]:
    build_maddison_multicountry(write=True)
    build_france_annual(write=True)
    build_italy_annual(write=True)
    build_sweden_annual(write=True)
    build_france_dept_annual(write=True)
    build_country_climate_annual(write=True)
    build_war_panel(write=True)
    build_pandemic_panel(write=True)
    build_emdat_panel(write=True)
    build_climate_extremes(write=True)
    build_controls_panel(write=True)
    assemble_panel_multi(write=True)

    artifacts: dict[str, Path] = {}
    artifacts["fig1"], _, _ = make_fig1_multi()
    artifacts["fig2"], _, _ = make_fig2_pooled()
    artifacts["fig3"], _, _ = make_fig3_stacked()
    artifacts["fig4"], _, _ = make_fig4_heatmap()
    artifacts["fig6"], _, _ = make_fig6_france_subnational()
    artifacts["fig1r"], _, _ = make_fig1r()
    artifacts["fig2r"], _, _ = make_fig2r()
    artifacts["fig3r"], _, _ = make_fig3r()
    try:
        from analysis.paper4_shadow.long_shadow_fertility.figures.fig5_efp_cross_section import make_fig5_efp
        artifacts["fig5"], _, _ = make_fig5_efp()
    except Exception as exc:
        print(f"[fig5 EFP] skipped: {exc}")
    return artifacts


if __name__ == "__main__":
    artifacts = run_all_phase2()
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
