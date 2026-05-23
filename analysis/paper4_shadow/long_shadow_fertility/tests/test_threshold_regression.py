"""Phase 9 Pillar B1: Hansen 1996/2000 threshold regression."""
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression import (
    fit_threshold_regression,
)


def _synth(seed: int, c_true: float, beta_M: float, beta_T: float, n_per_cluster: int = 150):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(7):
        alpha = rng.normal(0, 0.5)
        for t in range(n_per_cluster):
            z = rng.uniform(8.0, 12.0)
            x = rng.normal(0, 1)
            beta = beta_M if z <= c_true else beta_T
            y = alpha + beta * x + rng.normal(0, 0.1)
            rows.append({"iso3": f"C{c}", "year": 1900 + t, "y": y, "x": x, "z": z})
    return pd.DataFrame(rows)


def test_recovers_known_threshold():
    df = _synth(seed=0, c_true=10.0, beta_M=-0.3, beta_T=0.1, n_per_cluster=150)
    res = fit_threshold_regression(df, y="y", x="x", z="z", n_boot=100, seed=0)
    assert abs(res["c_hat"] - 10.0) < 0.5, f"c_hat={res['c_hat']:.3f}, expected ~10.0"
    assert abs(res["beta_M"] - (-0.3)) < 0.08
    assert abs(res["beta_T"] - 0.1) < 0.08


def test_threshold_with_threshold_low_pvalue():
    df = _synth(seed=1, c_true=10.0, beta_M=-0.5, beta_T=0.5, n_per_cluster=150)
    res = fit_threshold_regression(df, y="y", x="x", z="z", n_boot=200, seed=1)
    assert res["sup_wald_pvalue"] < 0.10


def test_no_threshold_high_pvalue():
    df = _synth(seed=2, c_true=10.0, beta_M=0.2, beta_T=0.2, n_per_cluster=100)
    res = fit_threshold_regression(df, y="y", x="x", z="z", n_boot=200, seed=2)
    assert res["sup_wald_pvalue"] > 0.10


def test_returns_required_keys():
    df = _synth(seed=3, c_true=10.0, beta_M=-0.3, beta_T=0.1, n_per_cluster=80)
    res = fit_threshold_regression(df, y="y", x="x", z="z", n_boot=50, seed=3)
    for k in ("beta_M", "beta_T", "c_hat", "c_ci_lo", "c_ci_hi", "sup_wald_pvalue", "n"):
        assert k in res
