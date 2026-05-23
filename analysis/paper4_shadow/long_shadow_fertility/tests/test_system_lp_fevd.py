"""Phase 7 Pillar E1: system LP-FEVD on multi-country panel."""
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)


def _synth_panel_t_to_w_to_f(seed: int = 0, n_countries: int = 7, t_max: int = 200) -> pd.DataFrame:
    """Synthetic DGP where T -> W -> F (no direct T -> F path), M independent."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_countries):
        alpha = rng.normal(0, 0.5)
        for t in range(t_max):
            T = rng.normal(0, 1)
            W = 0.5 * T + rng.normal(0, 0.3)
            M = rng.normal(0, 0.5)
            F = alpha + 0.4 * W + rng.normal(0, 0.2)
            rows.append({"iso3": f"C{c}", "year": 1800 + t,
                          "T": T, "W": W, "M": M, "F": F})
    return pd.DataFrame(rows)


def test_returns_irf_and_fevd():
    df = _synth_panel_t_to_w_to_f()
    res = fit_system_lp_fevd(df, variables=["T", "W", "M", "F"], horizons=range(0, 5))
    assert "irf" in res and "fevd" in res
    assert ("F", "T") in res["irf"]
    assert len(res["irf"][("F", "T")]) == 5


def test_fevd_shares_sum_to_one():
    df = _synth_panel_t_to_w_to_f()
    res = fit_system_lp_fevd(df, variables=["T", "W", "M", "F"], horizons=range(0, 5))
    fevd = res["fevd"]
    for out in ("T", "W", "M", "F"):
        for h in range(5):
            share_sum = sum(fevd[out][s][h] for s in range(4))
            assert abs(share_sum - 1.0) < 1e-6


def test_cholesky_ordering_changes_fevd():
    """Swap T's POSITION (not just later variables) to test Cholesky sensitivity."""
    df = _synth_panel_t_to_w_to_f()
    base = fit_system_lp_fevd(df, variables=["T", "W", "M", "F"], horizons=range(0, 5))
    swap = fit_system_lp_fevd(df, variables=["W", "T", "M", "F"], horizons=range(0, 5))
    # In base ordering, T is shock index 0 in F's FEVD
    # In swap ordering, T is shock index 1 (W is now first)
    base_T_share_in_F = base["fevd"]["F"][0][4]   # T-shock contribution to F
    swap_T_share_in_F = swap["fevd"]["F"][1][4]   # T-shock now at index 1
    # Different orderings of (T, W) produce materially different T-shares
    assert abs(base_T_share_in_F - swap_T_share_in_F) > 0.05


def test_synthetic_recovery_attributes_to_wage_channel():
    """In a T -> W -> F DGP with no direct T -> F path:
    - The wage variable W must absorb non-trivial F-variance under Cholesky [T,W,M,F].
    - The mortality M (independent of T) should explain very little F-variance.
    - F's own-innovation share is the residual portion.

    Note: under Cholesky with T first, the T structural shock subsumes the
    entire indirect T -> W -> F chain, so T's apparent share can be large (~40%)
    even with no direct path. This is correct Cholesky semantics — the dynamic
    decomposition into 'direct' vs 'wage-mediated' requires the ordering swap
    in Fig 18's Panel D (comparison of [T,W,M,F] vs [T,F,W,M]) rather than
    this synthetic-recovery test."""
    df = _synth_panel_t_to_w_to_f(t_max=400)
    res = fit_system_lp_fevd(df, variables=["T", "W", "M", "F"], horizons=range(0, 8))
    fevd_F = res["fevd"]["F"]
    w_share_long = fevd_F[1][7]
    m_share_long = fevd_F[2][7]
    # Wage shock should contribute non-trivially to F (the true transmission)
    assert w_share_long > 0.05
    # Mortality (independent of T in DGP) should contribute very little
    assert m_share_long < 0.10
