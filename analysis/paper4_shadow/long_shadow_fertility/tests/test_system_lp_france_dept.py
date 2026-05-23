"""Phase 7 Pillar B5: France dept single-equation LP with year FE."""
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_france_dept import (
    fit_dept_local_projection,
)


def _synth_dept_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    year_fe = {y: rng.normal(0, 0.05) for y in range(1851, 1898)}
    depts = [f"{i:02d}" for i in range(1, 91)]
    dept_fe = {d: rng.normal(0, 0.4) for d in depts}
    for d in depts:
        for y in range(1851, 1898):
            t = rng.normal(0, 1)
            log_cbr = dept_fe[d] + year_fe[y] + (-0.03) * t + rng.normal(0, 0.05)
            rows.append({"dep": d, "year": y, "log_cbr": log_cbr, "t_growing": t})
    return pd.DataFrame(rows)


def test_lp_returns_horizon_betas():
    df = _synth_dept_panel()
    out = fit_dept_local_projection(df, y="log_cbr", shock="t_growing", horizons=range(0, 4))
    assert "h" in out.columns and "beta" in out.columns and "se" in out.columns
    assert len(out) == 4
    assert (out["h"] == [0, 1, 2, 3]).all()


def test_lp_recovers_h0_coefficient():
    df = _synth_dept_panel()
    out = fit_dept_local_projection(df, y="log_cbr", shock="t_growing", horizons=range(0, 3))
    b0 = out.loc[out["h"] == 0, "beta"].iloc[0]
    assert -0.05 < b0 < -0.01


def test_lp_se_positive():
    df = _synth_dept_panel()
    out = fit_dept_local_projection(df, y="log_cbr", shock="t_growing", horizons=range(0, 4))
    assert (out["se"] > 0).all()
