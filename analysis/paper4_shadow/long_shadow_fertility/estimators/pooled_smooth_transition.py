"""Pooled smooth-transition: shared slopes beta_M, beta_T; country intercepts alpha_c."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def _residuals(params, x, z, y, country_idx, n_countries, theta_max=5.0):
    alphas = params[:n_countries]
    beta_m, beta_t, c, theta, gamma = params[n_countries:n_countries + 5]
    g = 1.0 / (1.0 + np.exp(-theta * (z - c)))
    beta = beta_m * (1.0 - g) + beta_t * g
    return alphas[country_idx] + beta * x + gamma * z - y


def fit_pooled_smooth_transition(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str,
    theta_max: float = 5.0,
    n_boot: int = 100,
    seed: int = 0,
) -> dict:
    sub = df[["iso3", y, x, z]].dropna()
    countries = sorted(sub["iso3"].unique())
    n_c = len(countries)
    country_idx = sub["iso3"].map({c: i for i, c in enumerate(countries)}).to_numpy()
    yv = sub[y].to_numpy()
    xv = sub[x].to_numpy()
    zv = sub[z].to_numpy()

    lo, hi = np.percentile(zv, [25, 75])
    bm0 = np.polyfit(xv[zv < lo], yv[zv < lo], 1)[0] if (zv < lo).sum() > 3 else 0.0
    bt0 = np.polyfit(xv[zv > hi], yv[zv > hi], 1)[0] if (zv > hi).sum() > 3 else 0.0
    p0 = np.concatenate([
        np.full(n_c, float(yv.mean())),
        np.array([float(bm0), float(bt0), float(np.median(zv)), 1.0, 0.0]),
    ])
    bounds_lo = np.concatenate([
        np.full(n_c, -np.inf),
        np.array([-np.inf, -np.inf, zv.min(), 0.05, -np.inf]),
    ])
    bounds_hi = np.concatenate([
        np.full(n_c, np.inf),
        np.array([np.inf, np.inf, zv.max(), float(theta_max), np.inf]),
    ])

    res = least_squares(
        _residuals, p0,
        args=(xv, zv, yv, country_idx, n_c, theta_max),
        bounds=(bounds_lo, bounds_hi),
    )
    beta_m, beta_t, c, theta, gamma = res.x[n_c:n_c + 5]
    alphas = res.x[:n_c]

    rng = np.random.default_rng(seed)
    boot_m, boot_t = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), size=len(yv))
        try:
            r = least_squares(
                _residuals, res.x,
                args=(xv[idx], zv[idx], yv[idx], country_idx[idx], n_c, theta_max),
                bounds=(bounds_lo, bounds_hi),
            )
            boot_m.append(r.x[n_c]); boot_t.append(r.x[n_c + 1])
        except Exception:
            continue

    return {
        "beta_M": float(beta_m), "beta_T": float(beta_t),
        "c": float(c), "theta": float(theta), "gamma": float(gamma),
        "alphas": {c_: float(a) for c_, a in zip(countries, alphas)},
        "beta_M_se": float(np.std(boot_m)) if boot_m else float("nan"),
        "beta_T_se": float(np.std(boot_t)) if boot_t else float("nan"),
        "n": int(len(yv)),
        "n_countries": n_c,
    }
