"""Tests for Wright (2003) Anderson-Rubin identification-robust CI."""
import numpy as np
import pytest
from analysis.paper4_shadow.long_shadow_fertility.estimators.ar_inference import (
    anderson_rubin_ci,
)


def test_ar_ci_covers_truth_under_strong_iv():
    """With a strong first stage, the AR CI should cover the true beta and be tight."""
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    x = 2.0 * z + rng.normal(size=n)          # strong first stage
    y = 0.5 * x + rng.normal(size=n)          # true beta = 0.5
    lo, hi = anderson_rubin_ci(y, x, z, alpha=0.05)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= 0.5 <= hi
    assert hi - lo < 1.0                       # tight under strong IV


def test_ar_ci_widens_under_weak_iv():
    """With a very weak first stage, the AR CI should be wide (or unbounded)."""
    rng = np.random.default_rng(1)
    n = 500
    z = rng.normal(size=n)
    x = 0.05 * z + rng.normal(size=n)          # weak first stage
    y = 0.5 * x + rng.normal(size=n)
    lo, hi = anderson_rubin_ci(y, x, z, alpha=0.05)
    # AR CI under weak IV should still cover the truth but be much wider
    if np.isfinite(lo) and np.isfinite(hi):
        assert hi - lo > 3.0
    # else allow unbounded — that's also a valid AR outcome
