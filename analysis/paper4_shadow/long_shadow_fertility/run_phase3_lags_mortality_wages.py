"""End-to-end Phase 3: build mortality + wages, run six new figures."""
from __future__ import annotations
from pathlib import Path

from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel import (
    build_real_wage_panel,
)

from analysis.paper4_shadow.long_shadow_fertility.figures.fig7_distributed_lag_irf import make_fig7_dl_irf
from analysis.paper4_shadow.long_shadow_fertility.figures.fig8_volatility_treatment import make_fig8_vol
from analysis.paper4_shadow.long_shadow_fertility.figures.fig9_joint_fertility_mortality import make_fig9_joint
from analysis.paper4_shadow.long_shadow_fertility.figures.fig10_str_real_wage import make_fig10_str_wage
from analysis.paper4_shadow.long_shadow_fertility.figures.fig11_mediation_diagram import make_fig11_mediation
from analysis.paper4_shadow.long_shadow_fertility.figures.fig12_dl_str_lag_by_regime import make_fig12_dl_regime


def run_all_phase3() -> dict[str, Path]:
    build_country_mortality_annual(write=True)
    build_real_wage_panel(write=True)
    artifacts: dict[str, Path] = {}
    artifacts["fig7"], _, _ = make_fig7_dl_irf()
    artifacts["fig8"], _, _ = make_fig8_vol()
    artifacts["fig9"], _, _ = make_fig9_joint()
    artifacts["fig10"], _, _ = make_fig10_str_wage()
    artifacts["fig11"], _, _ = make_fig11_mediation()
    artifacts["fig12"], _, _ = make_fig12_dl_regime()
    return artifacts


if __name__ == "__main__":
    artifacts = run_all_phase3()
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
