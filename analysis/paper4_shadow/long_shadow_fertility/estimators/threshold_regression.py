"""Hansen 1996/2000 threshold regression for panel data.

Model:
    y_{c,t} = alpha_c + beta_M * x_{c,t} * 1{z_{c,t} <= c} + beta_T * x_{c,t} * 1{z_{c,t} > c} + eps

Estimation:
1. Grid search over candidate thresholds in [trim_low, trim_high] quantiles of z.
2. For each candidate c: fit OLS with country FE; record RSS + Wald stat for beta_M=beta_T.
3. Select c* minimizing RSS.
4. Sup-Wald test (Hansen 1996): wild cluster bootstrap p-value.
5. LR-based CI for c (Hansen 2000): invert the LR test on the grid (95% cutoff = 7.35).

Returns: dict with beta_M, beta_T, c_hat, c_ci_lo, c_ci_hi, sup_wald_pvalue, n, lr_path.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit_at_c(df: pd.DataFrame, y: str, x: str, z: str, c: float,
                unit_col: str) -> tuple[float, float, float, float]:
    """Return (beta_M, beta_T, RSS, Wald_stat) at threshold c."""
    df = df.dropna(subset=[y, x, z, unit_col]).copy()
    df["below"] = (df[z] <= c).astype(float)
    df["x_M"] = df[x] * df["below"]
    df["x_T"] = df[x] * (1.0 - df["below"])
    dums = pd.get_dummies(df[unit_col], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[["x_M", "x_T"]].astype(float), dums], axis=1))
    res = sm.OLS(df[y].astype(float).to_numpy(), X.to_numpy()).fit()
    beta_M = float(res.params[1])
    beta_T = float(res.params[2])
    rss = float((res.resid ** 2).sum())
    cov = res.cov_params()
    R = np.zeros(len(res.params))
    R[1] = 1.0
    R[2] = -1.0
    diff = beta_M - beta_T
    v = float(R @ cov @ R)
    wald = diff ** 2 / v if v > 0 else 0.0
    return beta_M, beta_T, rss, wald


def fit_threshold_regression(
    df: pd.DataFrame,
    y: str, x: str, z: str,
    unit_col: str = "iso3",
    n_boot: int = 500,
    trim: tuple[float, float] = (0.15, 0.85),
    seed: int = 0,
) -> dict:
    df = df.dropna(subset=[y, x, z, unit_col]).copy()
    rng = np.random.default_rng(seed)
    z_vals = df[z].to_numpy()
    q_lo, q_hi = np.quantile(z_vals, trim[0]), np.quantile(z_vals, trim[1])
    grid = np.linspace(q_lo, q_hi, 50)

    results = [(c, *_fit_at_c(df, y, x, z, c, unit_col)) for c in grid]
    # tuple layout: (c, beta_M, beta_T, rss, wald)
    best = min(results, key=lambda r: r[3])  # MIN RSS
    c_hat, beta_M_hat, beta_T_hat, rss_hat, wald_hat = best

    # Sup-Wald wild cluster bootstrap
    null_supwalds = []
    units = df[unit_col].unique()
    for _ in range(n_boot):
        weights = rng.choice([-1.0, 1.0], size=len(units))
        w_map = dict(zip(units, weights))
        boot = df.copy()
        boot[y] = boot[y] * boot[unit_col].map(w_map)
        boot_walds = []
        for c in grid:
            _, _, _, w = _fit_at_c(boot, y, x, z, c, unit_col)
            boot_walds.append(w)
        null_supwalds.append(max(boot_walds))
    p = float(np.mean([s >= wald_hat for s in null_supwalds]))

    # LR-based CI for c (Hansen 2000)
    sigma2 = rss_hat / len(df)
    lr_path = [(c, (rss - rss_hat) / sigma2) for c, _, _, rss, _ in results]
    ci_threshold = 7.35
    in_ci = [c for c, lr in lr_path if lr <= ci_threshold]
    if in_ci:
        c_ci_lo = float(min(in_ci))
        c_ci_hi = float(max(in_ci))
    else:
        c_ci_lo, c_ci_hi = float(c_hat), float(c_hat)

    return {
        "beta_M": beta_M_hat, "beta_T": beta_T_hat,
        "c_hat": float(c_hat),
        "c_ci_lo": c_ci_lo, "c_ci_hi": c_ci_hi,
        "sup_wald_pvalue": p,
        "n": int(len(df)),
        "lr_path": lr_path,
    }
