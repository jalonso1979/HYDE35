"""Phase 7 end-to-end driver: regenerate all Phase 7 outputs from scratch."""
from __future__ import annotations
import importlib
import runpy

PIPELINE = [
    # Pillar D: harmonized wage panel + downstream Phase 5 figures
    "analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig7v3_pooled_dl_irf_v3",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig10v2_str_harmonized_wage",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig11v3_mediation_wild_bootstrap",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig15_triple_sur",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig16_phase_progression",
    # Pillar B: France dept DL + LP
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig17_france_dept_dl_irf",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig19_france_dept_system_lp",
    # Pillar C: volcanic splice + extended IV (graceful on BLOCKED)
    "analysis.paper4_shadow.long_shadow_fertility.data.build_volcanic_panel_v2",
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig13v2_volcanic_iv_extended",
    # Pillar E: system LP-FEVD
    "analysis.paper4_shadow.long_shadow_fertility.figures.fig18_pooled_system_lp_fevd",
]


def main():
    for mod_name in PIPELINE:
        print(f"=== {mod_name} ===")
        try:
            runpy.run_module(mod_name, run_name="__main__")
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
