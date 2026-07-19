"""Wright (2003) Anderson-Rubin identification-robust confidence interval for IV.

Reference: Anderson & Rubin (1949); Wright (2003) 'Weak Instruments, Weak
Identification, and Weak Instruments and Weak Identification in Estimating the
Returns to Schooling'.

The AR CI is obtained by inverting the AR test: for each candidate beta on a
grid, form the residual r = y - beta*x and test whether z is uncorrelated with
r. The CI is the set of betas for which the null is not rejected.

This one-instrument, one-endogenous-regressor case is exact even under
arbitrarily weak identification.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def anderson_rubin_ci(
    y,
    x,
    z,
    alpha: float = 0.05,
    grid_n: int = 2001,
    beta_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Wright (2003) Anderson-Rubin identification-robust CI for IV with one
    endogenous regressor and one instrument.

    Procedure: invert the AR test. At each candidate beta on a grid, regress
    (y - beta * x) on z and test whether the z-coefficient is zero. The
    confidence interval is the set of betas for which the test does NOT
    reject at level alpha.

    Parameters
    ----------
    y : array-like, length n
        Outcome.
    x : array-like, length n
        Endogenous regressor.
    z : array-like, length n
        Instrument.
    alpha : float
        Test level. CI is (1 - alpha) confidence.
    grid_n : int
        Number of beta values on the search grid.
    beta_range : (float, float) or None
        Search range. If None, uses simple-IV estimate ± 5.

    Returns
    -------
    (lo, hi) : tuple of floats
        AR CI. May be (-inf, +inf) under very weak identification. Returns
        (nan, nan) if grid is entirely rejected (set is empty).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(y)

    if beta_range is None:
        zx = float(np.dot(z, x))
        if abs(zx) < 1e-10:
            beta_range = (-10.0, 10.0)
        else:
            b0 = float(np.dot(z, y) / zx)
            beta_range = (b0 - 5.0, b0 + 5.0)

    grid = np.linspace(beta_range[0], beta_range[1], grid_n)
    z_dem = z - z.mean()
    # Residual dof: n minus the intercept and the z slope partialled out (n - 2).
    crit = stats.f.ppf(1.0 - alpha, 1, n - 2)

    accept = []
    for b in grid:
        r = y - b * x
        r_dem = r - r.mean()
        # Numerator: projection of r onto z (partitioned out of mean)
        num_e = (np.dot(z_dem, r_dem)) ** 2 / np.dot(z_dem, z_dem)
        # SSE after removing the z contribution
        sse = float(np.dot(r_dem, r_dem) - num_e)
        if sse <= 0:
            f = np.inf
        else:
            f = num_e / (sse / (n - 2))
        if f <= crit:
            accept.append(float(b))

    if not accept:
        return (float("nan"), float("nan"))
    return (min(accept), max(accept))
