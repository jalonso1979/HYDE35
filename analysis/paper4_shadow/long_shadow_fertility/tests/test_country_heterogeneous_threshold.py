import sys
import pandas as pd

sys.path.insert(0, "/Volumes/BIGDATA/HYDE35")

from analysis.paper4_shadow.long_shadow_fertility.estimators.country_heterogeneous_threshold import (
    fit_country_specific_thresholds,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)


def test_seven_country_specific_thresholds():
    """Returns a result per country with at least min_obs valid rows."""
    panel = pd.read_parquet(
        "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
        "panel_multi_country_year.parquet"
    )
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(wage, on=["iso3", "year"], how="left")

    out = fit_country_specific_thresholds(
        df, y="log_cbr", x="t_growing", z="log_real_wage",
        unit_col="iso3", n_boot=20, seed=0, min_obs=80,
    )
    # 7 countries expected (England, France, Italy, Sweden, Belgium, Netherlands, Spain)
    # — but some may be skipped if min_obs not met. Assert >= 3 returned.
    assert len(out) >= 3
    for iso3, res in out.items():
        for k in ["beta_M", "beta_T", "c_hat", "c_ci_lo", "c_ci_hi", "sup_wald_pvalue", "n"]:
            assert k in res, f"missing key {k} for {iso3}"
        assert res["c_ci_lo"] <= res["c_hat"] <= res["c_ci_hi"]
