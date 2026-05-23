"""Bivariate SUR (seemingly-unrelated regressions) for jointly modeling
fertility and mortality responses to the same climate shock.

Two-step FGLS:
1. Per-equation OLS -> residuals.
2. Residual covariance Sigma.
3. Joint GLS on stacked design with Sigma (x) I_n weighting.

Memory note: the joint GLS step builds an explicit 2n x 2n weight matrix W
via np.eye(n). This is O(n^2) memory. For tests (n=800) this is fine; for the
full Phase 3 panel (n~3000) consider refactoring to closed-form per-block
computation using kron(sinv, I_n) and exploiting block structure
(A = sum_{ij} sinv[i,j] X.T X stacked appropriately).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _build_design(df: pd.DataFrame, x: str, unit_col: str, controls: Sequence[str]):
    dums = pd.get_dummies(df[unit_col], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[[x]].astype(float), dums, df[list(controls)].astype(float)], axis=1))
    return X.to_numpy()


def fit_bivariate_sur(
    df: pd.DataFrame,
    y1: str,
    y2: str,
    x: str,
    unit_col: str = "iso3",
    controls: Sequence[str] | None = None,
) -> dict:
    controls = list(controls or [])
    sub = df.dropna(subset=[y1, y2, x] + controls + [unit_col]).copy()
    X = _build_design(sub, x, unit_col, controls)
    y1v = sub[y1].astype(float).to_numpy()
    y2v = sub[y2].astype(float).to_numpy()
    n, k = X.shape

    res1 = sm.OLS(y1v, X).fit()
    res2 = sm.OLS(y2v, X).fit()
    e1 = y1v - res1.fittedvalues
    e2 = y2v - res2.fittedvalues
    sigma = np.array([[float(np.dot(e1, e1) / n), float(np.dot(e1, e2) / n)],
                       [float(np.dot(e1, e2) / n), float(np.dot(e2, e2) / n)]])

    sinv = np.linalg.inv(sigma)
    Z = np.zeros((2 * n, 2 * k))
    Z[:n, :k] = X; Z[n:, k:] = X
    yz = np.concatenate([y1v, y2v])
    W = np.zeros((2 * n, 2 * n))
    W[:n, :n] = sinv[0, 0] * np.eye(n); W[:n, n:] = sinv[0, 1] * np.eye(n)
    W[n:, :n] = sinv[1, 0] * np.eye(n); W[n:, n:] = sinv[1, 1] * np.eye(n)
    A = Z.T @ W @ Z
    b = Z.T @ W @ yz
    B_hat = np.linalg.solve(A, b)
    cov_B = np.linalg.inv(A)

    beta_y1 = float(B_hat[1])
    beta_y2 = float(B_hat[k + 1])
    se_y1 = float(np.sqrt(cov_B[1, 1]))
    se_y2 = float(np.sqrt(cov_B[k + 1, k + 1]))

    R = np.zeros((1, 2 * k))
    R[0, 1] = 1.0; R[0, k + 1] = -1.0
    diff = float((R @ B_hat)[0])
    var_diff = float((R @ cov_B @ R.T)[0, 0])
    wald = diff ** 2 / var_diff
    from scipy.stats import chi2
    pval = float(1 - chi2.cdf(wald, df=1))

    return {
        "beta_y1": beta_y1, "beta_y2": beta_y2,
        "se_y1": se_y1, "se_y2": se_y2,
        "wald_eq_stat": float(wald), "wald_eq_pvalue": pval,
        "n": int(n), "sigma": sigma,
    }
