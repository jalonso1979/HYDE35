"""Smooth-transition regression with logistic transition function.

Spec model (§5b):
    y_t = alpha + [beta_M * (1 - G(z_t)) + beta_T * G(z_t)] * x_t + gamma * z_t + eps_t
    G(z) = 1 / (1 + exp(-theta * (z - c)))

Estimation: scipy.optimize.least_squares; bootstrap SEs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def _residuals(params, x, z, y):
    alpha, beta_m, beta_t, c, theta, gamma = params
    g = 1.0 / (1.0 + np.exp(-theta * (z - c)))
    beta = beta_m * (1.0 - g) + beta_t * g
    return alpha + beta * x + gamma * z - y


def fit_smooth_transition(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    sub = df[[y, x, z]].dropna().to_numpy()
    yv, xv, zv = sub[:, 0], sub[:, 1], sub[:, 2]
    # init: beta_M from x effect when z low; beta_T when z high
    lo, hi = np.percentile(zv, [25, 75])
    beta_m0 = np.polyfit(xv[zv < lo], yv[zv < lo], 1)[0] if (zv < lo).sum() > 3 else 0.0
    beta_t0 = np.polyfit(xv[zv > hi], yv[zv > hi], 1)[0] if (zv > hi).sum() > 3 else 0.0
    p0 = np.array([float(yv.mean()), float(beta_m0), float(beta_t0),
                    float(np.median(zv)), 1.0, 0.0])
    bounds_lo = np.array([-np.inf, -np.inf, -np.inf, zv.min(), 0.05, -np.inf])
    bounds_hi = np.array([ np.inf,  np.inf,  np.inf, zv.max(), 20.0,  np.inf])
    res = least_squares(_residuals, p0, args=(xv, zv, yv), bounds=(bounds_lo, bounds_hi))
    alpha, beta_m, beta_t, c, theta, gamma = res.x

    # Bootstrap SE for beta_M and beta_T
    rng = np.random.default_rng(seed)
    boot_m, boot_t = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), size=len(yv))
        try:
            r = least_squares(_residuals, res.x, args=(xv[idx], zv[idx], yv[idx]),
                                bounds=(bounds_lo, bounds_hi))
            boot_m.append(r.x[1]); boot_t.append(r.x[2])
        except Exception:
            continue
    return {
        "beta_M": float(beta_m), "beta_T": float(beta_t),
        "c": float(c), "theta": float(theta),
        "alpha": float(alpha), "gamma": float(gamma),
        "beta_M_se": float(np.std(boot_m)) if boot_m else float("nan"),
        "beta_T_se": float(np.std(boot_t)) if boot_t else float("nan"),
        "n": int(len(yv)),
    }
