"""3-equation SUR. Two-step FGLS:
1. Per-equation OLS -> residuals.
2. 3x3 residual covariance Sigma.
3. Joint GLS on stacked design with (Sigma (x) I_n) weighting.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


def _build_design(df: pd.DataFrame, x: str, unit_col: str, controls: Sequence[str]):
    dums = pd.get_dummies(df[unit_col], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[[x]].astype(float), dums,
                                     df[list(controls)].astype(float)], axis=1))
    return X.to_numpy()


def fit_triple_sur(
    df: pd.DataFrame,
    ys: list[str],
    x: str,
    unit_col: str = "iso3",
    controls: Sequence[str] | None = None,
) -> dict:
    assert len(ys) == 3, "Triple SUR requires exactly 3 dependent variables"
    controls = list(controls or [])
    sub = df.dropna(subset=ys + [x] + controls + [unit_col]).copy()
    X = _build_design(sub, x, unit_col, controls)
    Y = [sub[y].astype(float).to_numpy() for y in ys]
    n, k = X.shape

    res_each = [sm.OLS(Y[i], X).fit() for i in range(3)]
    resid = [Y[i] - res_each[i].fittedvalues for i in range(3)]
    sigma = np.array([[float(np.dot(resid[i], resid[j]) / n) for j in range(3)] for i in range(3)])

    sinv = np.linalg.inv(sigma)
    Z = np.zeros((3 * n, 3 * k))
    for i in range(3):
        Z[i * n:(i + 1) * n, i * k:(i + 1) * k] = X
    yz = np.concatenate(Y)
    W = np.zeros((3 * n, 3 * n))
    for i in range(3):
        for j in range(3):
            W[i * n:(i + 1) * n, j * n:(j + 1) * n] = sinv[i, j] * np.eye(n)
    A = Z.T @ W @ Z
    b = Z.T @ W @ yz
    B_hat = np.linalg.solve(A, b)
    cov_B = np.linalg.inv(A)

    betas = [float(B_hat[i * k + 1]) for i in range(3)]
    ses = [float(np.sqrt(cov_B[i * k + 1, i * k + 1])) for i in range(3)]

    def _wald(i: int, j: int) -> tuple[float, float]:
        R = np.zeros((1, 3 * k))
        R[0, i * k + 1] = 1.0
        R[0, j * k + 1] = -1.0
        diff = float((R @ B_hat)[0])
        v = float((R @ cov_B @ R.T)[0, 0])
        w = diff ** 2 / v
        return w, float(1 - chi2.cdf(w, df=1))

    w12, p12 = _wald(0, 1)
    w13, p13 = _wald(0, 2)
    w23, p23 = _wald(1, 2)

    return {
        "beta_y1": betas[0], "beta_y2": betas[1], "beta_y3": betas[2],
        "se_y1": ses[0], "se_y2": ses[1], "se_y3": ses[2],
        "wald_eq12_stat": w12, "wald_eq12_pvalue": p12,
        "wald_eq13_stat": w13, "wald_eq13_pvalue": p13,
        "wald_eq23_stat": w23, "wald_eq23_pvalue": p23,
        "n": int(n), "sigma": sigma,
    }
