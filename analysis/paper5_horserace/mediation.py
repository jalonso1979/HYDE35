# analysis/paper5_horserace/mediation.py
"""Bootstrap mediation-share estimator.

For (outcome y, substrate s, mediator P, controls X):
  beta_hat   = OLS coef on s in: y ~ s + X
  beta_tilde = OLS coef on s in: y ~ s + X + P
  mediation_share = 1 - beta_tilde / beta_hat

Non-parametric percentile bootstrap over countries.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _ols_coef(df: pd.DataFrame, y_col: str, target: str, controls: list[str]) -> float:
    X = sm.add_constant(df[[target, *controls]])
    res = sm.OLS(df[y_col], X).fit()
    return float(res.params[target])


def mediation_share_with_ci(
    df: pd.DataFrame,
    y_col: str,
    substrate: str,
    pathway_dummies: Sequence[str],
    controls: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Return mediation share point estimate + percentile bootstrap CI."""
    needed = [y_col, substrate, *pathway_dummies, *controls]
    df = df.dropna(subset=needed).copy()
    rng = np.random.default_rng(seed)

    def _share(d: pd.DataFrame) -> float:
        beta_hat = _ols_coef(d, y_col, substrate, list(controls))
        beta_tilde = _ols_coef(d, y_col, substrate, list(controls) + list(pathway_dummies))
        if beta_hat == 0:
            return np.nan
        return 1.0 - beta_tilde / beta_hat

    point = _share(df)
    boot = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot.append(_share(df.iloc[idx]))
    boot = np.array(boot)
    boot = boot[~np.isnan(boot)]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    return {
        "mediation_share": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_obs": n,
        "n_boot_valid": len(boot),
    }
