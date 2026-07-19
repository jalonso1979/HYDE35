# analysis/paper5_horserace/mediation.py
"""Bootstrap mediation-share estimators.

Two definitions of mediation share are supported:

  1. β-attenuation (single-substrate, original):
       beta_hat   = OLS coef on s in: y ~ s + X
       beta_tilde = OLS coef on s in: y ~ s + X + P
       mediation_share = 1 - beta_tilde / beta_hat

  2. Partial-R² (supports single column or vector substrate):
       ΔR²_S    = R²(y ~ S + X)     - R²(y ~ X)
       ΔR²_S|P  = R²(y ~ S + X + P) - R²(y ~ X + P)
       mediation_share = 1 - ΔR²_S|P / ΔR²_S

For a single-column substrate the two definitions track each other but are
not identical. The partial-R² form is the natural extension when the
"substrate" is a coalition (e.g., the 4-element climate bundle).

Non-parametric percentile bootstrap over countries.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

SubstrateSpec = Union[str, Sequence[str]]


def stable_seed(*parts) -> int:
    """Deterministic 31-bit RNG seed from arbitrary key parts.

    Uses hashlib rather than the built-in ``hash``, which is salted per process
    via ``PYTHONHASHSEED`` and therefore makes bootstrap draws non-reproducible
    run-to-run.
    """
    import hashlib
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest, 16) % (2**31)


def _expand(spec: SubstrateSpec) -> list[str]:
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def _ols_coef(df: pd.DataFrame, y_col: str, target: str, controls: list[str]) -> float:
    X = sm.add_constant(df[[target, *controls]])
    res = sm.OLS(df[y_col], X).fit()
    return float(res.params[target])


def _ols_r2(df: pd.DataFrame, y_col: str, regressors: list[str]) -> float:
    if not regressors:
        return 0.0
    X = sm.add_constant(df[regressors])
    res = sm.OLS(df[y_col], X).fit()
    return float(res.rsquared)


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
    """β-attenuation mediation share + percentile bootstrap CI. Single-column substrate."""
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


def mediation_share_partial_r2_with_ci(
    df: pd.DataFrame,
    y_col: str,
    substrate: SubstrateSpec,
    pathway_dummies: Sequence[str],
    controls: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Partial-R² mediation share + percentile bootstrap CI.

    Substrate may be a single column name or a sequence of column names
    (coalition / bundle). The mediation share is:
        1 - (R²(y~S+X+P) - R²(y~X+P)) / (R²(y~S+X) - R²(y~X))

    Robustness: when the un-mediated partial R² of S is essentially zero
    (< 1e-4), the ratio is unstable; we return NaN for that bootstrap draw.
    """
    s_cols = _expand(substrate)
    needed = [y_col, *s_cols, *pathway_dummies, *controls]
    df = df.dropna(subset=needed).copy()
    rng = np.random.default_rng(seed)

    def _share(d: pd.DataFrame) -> float:
        r2_x = _ols_r2(d, y_col, list(controls))
        r2_sx = _ols_r2(d, y_col, list(controls) + s_cols)
        r2_xp = _ols_r2(d, y_col, list(controls) + list(pathway_dummies))
        r2_sxp = _ols_r2(d, y_col, list(controls) + s_cols + list(pathway_dummies))
        delta_un = r2_sx - r2_x
        delta_med = r2_sxp - r2_xp
        if delta_un < 1e-4:
            return np.nan
        return 1.0 - delta_med / delta_un

    point = _share(df)
    boot = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot.append(_share(df.iloc[idx]))
    boot = np.array(boot)
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        lo, hi = np.nan, np.nan
    else:
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    return {
        "mediation_share": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_obs": n,
        "n_boot_valid": len(boot),
    }
