"""STR on EFP cross-section (province x decade), province-clustered SEs."""
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


def fit_efp_str(df: pd.DataFrame, y: str, x: str, z: str,
                country_col: str = "country", theta_max: float = 5.0) -> dict:
    sub = df.dropna(subset=[y, x, z, country_col])
    countries = sorted(sub[country_col].unique())
    n_c = len(countries)
    ci = sub[country_col].map({c: i for i, c in enumerate(countries)}).to_numpy()
    yv, xv, zv = sub[y].to_numpy(), sub[x].to_numpy(), sub[z].to_numpy()
    p0 = np.concatenate([
        np.full(n_c, float(yv.mean())),
        np.array([0.0, 0.0, float(np.median(zv)), 1.0, 0.0]),
    ])
    bounds_lo = np.concatenate([np.full(n_c, -np.inf),
                                 np.array([-np.inf, -np.inf, zv.min(), 0.05, -np.inf])])
    bounds_hi = np.concatenate([np.full(n_c, np.inf),
                                 np.array([np.inf, np.inf, zv.max(), float(theta_max), np.inf])])
    res = least_squares(_residuals, p0, args=(xv, zv, yv, ci, n_c, theta_max),
                         bounds=(bounds_lo, bounds_hi))
    bm, bt, c, theta, gamma = res.x[n_c:n_c + 5]
    return {"beta_M": float(bm), "beta_T": float(bt), "c": float(c),
            "theta": float(theta), "n": int(len(yv)), "n_countries": n_c}
