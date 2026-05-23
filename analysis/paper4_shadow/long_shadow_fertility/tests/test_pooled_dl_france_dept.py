"""Phase 7 Pillar B: pooled DL on France dept-year panel."""
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_dl_france_dept import (
    fit_dept_distributed_lag,
)


def _synth_dept_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    year_fe = {y: rng.normal(0, 0.05) for y in range(1851, 1898)}
    depts = [f"{i:02d}" for i in range(1, 91)]
    dept_fe = {d: rng.normal(0, 0.4) for d in depts}
    for d in depts:
        t_prev = 0.0
        for y in range(1851, 1898):
            t = rng.normal(0, 1)
            p = rng.normal(0, 1)
            log_cbr = (
                dept_fe[d] + year_fe[y]
                + (-0.04) * t + (-0.02) * t_prev
                + 0.01 * p + rng.normal(0, 0.05)
            )
            rows.append({"dep": d, "year": y, "log_cbr": log_cbr, "t_growing": t, "p_growing": p})
            t_prev = t
    return pd.DataFrame(rows)


def test_returns_irf_dataframe():
    df = _synth_dept_panel()
    out = fit_dept_distributed_lag(df, y="log_cbr", x="t_growing", lags=2,
                                    controls=["p_growing"], year_fe=False)
    for c in ("lag", "beta", "se", "ci_low", "ci_high"):
        assert c in out.columns
    assert (out["lag"] == "cumulative").any()


def test_recovers_negative_climate_coefficient():
    df = _synth_dept_panel()
    out = fit_dept_distributed_lag(df, y="log_cbr", x="t_growing", lags=2,
                                    controls=["p_growing"], year_fe=False)
    b0 = out.loc[out["lag"] == 0, "beta"].iloc[0]
    assert -0.06 < b0 < -0.02


def test_year_fe_changes_coefficient():
    df = _synth_dept_panel()
    out_no_yfe = fit_dept_distributed_lag(df, y="log_cbr", x="t_growing", lags=2,
                                            controls=["p_growing"], year_fe=False)
    out_with_yfe = fit_dept_distributed_lag(df, y="log_cbr", x="t_growing", lags=2,
                                              controls=["p_growing"], year_fe=True)
    b0_no = out_no_yfe.loc[out_no_yfe["lag"] == 0, "beta"].iloc[0]
    b0_yes = out_with_yfe.loc[out_with_yfe["lag"] == 0, "beta"].iloc[0]
    assert abs(b0_no - b0_yes) > 1e-6


def test_se_positive_at_every_horizon():
    df = _synth_dept_panel()
    out = fit_dept_distributed_lag(df, y="log_cbr", x="t_growing", lags=3,
                                    controls=["p_growing"], year_fe=True)
    assert (out["se"] > 0).all()
