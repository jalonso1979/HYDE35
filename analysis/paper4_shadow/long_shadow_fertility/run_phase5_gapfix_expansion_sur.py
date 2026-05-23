"""End-to-end Phase 5: rebuild all panels with 7 countries + gap-filled England, then run six new figures."""
from __future__ import annotations
from pathlib import Path

from analysis.paper4_shadow.long_shadow_fertility.data.build_england_fertility_annual import (
    build_england_fertility_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_bel_fertility_annual import (
    build_bel_fertility_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_nld_fertility_annual import (
    build_nld_fertility_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_esp_fertility_annual import (
    build_esp_fertility_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_climate_annual import (
    build_country_climate_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_maddison_multicountry import (
    build_maddison_multicountry,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)

from analysis.paper4_shadow.long_shadow_fertility.figures.fig1v3_rolling_7country import make_fig1v3
from analysis.paper4_shadow.long_shadow_fertility.figures.fig7v3_pooled_dl_irf_v3 import make_fig7v3
from analysis.paper4_shadow.long_shadow_fertility.figures.fig10v2_str_harmonized_wage import make_fig10v2
from analysis.paper4_shadow.long_shadow_fertility.figures.fig11v3_mediation_wild_bootstrap import make_fig11v3
from analysis.paper4_shadow.long_shadow_fertility.figures.fig15_triple_sur import make_fig15
from analysis.paper4_shadow.long_shadow_fertility.figures.fig16_phase_progression import make_fig16


def run_all_phase5() -> dict[str, Path]:
    build_england_fertility_annual(write=True)
    build_bel_fertility_annual(write=True)
    build_nld_fertility_annual(write=True)
    build_esp_fertility_annual(write=True)
    build_country_mortality_annual(write=True)
    build_country_climate_annual(write=True)
    build_maddison_multicountry(write=True)
    build_real_wage_panel_v2(write=True)
    assemble_panel_multi(write=True)

    artifacts: dict[str, Path] = {}
    artifacts["fig1v3"], _, _ = make_fig1v3()
    artifacts["fig7v3"], _, _ = make_fig7v3()
    artifacts["fig10v2"], _, _ = make_fig10v2()
    artifacts["fig11v3"], _, _ = make_fig11v3()
    artifacts["fig15"], _, _ = make_fig15()
    artifacts["fig16"], _ = make_fig16()
    return artifacts


if __name__ == "__main__":
    artifacts = run_all_phase5()
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
