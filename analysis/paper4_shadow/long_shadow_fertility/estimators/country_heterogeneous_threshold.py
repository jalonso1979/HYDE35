"""Country-heterogeneous Hansen threshold estimator.

Wraps fit_threshold_regression to estimate a Hansen threshold
independently for each country (unit) in the panel. Use to test
whether the pooled threshold (Pillar B) is stable across countries.

Returns a dict mapping country code -> per-country result dict with
the same keys as fit_threshold_regression:
    beta_M, beta_T, c_hat, c_ci_lo, c_ci_hi, sup_wald_pvalue, n, lr_path.

Countries with fewer than min_obs valid rows are skipped. Countries
that fail estimation (e.g. insufficient z variation) return a dict with
an "error" key instead.
"""
from __future__ import annotations

import pandas as pd

from .threshold_regression import fit_threshold_regression


def fit_country_specific_thresholds(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str,
    unit_col: str = "iso3",
    n_boot: int = 500,
    trim: tuple[float, float] = (0.15, 0.85),
    seed: int = 0,
    min_obs: int = 80,
) -> dict[str, dict]:
    """For each unit (country), estimate a Hansen threshold independently.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with columns y, x, z, and unit_col.
    y : str
        Outcome variable (e.g. "log_cbr").
    x : str
        Regressor (e.g. "t_growing").
    z : str
        Threshold variable (e.g. "log_real_wage").
    unit_col : str
        Column identifying the cross-sectional unit (default "iso3").
    n_boot : int
        Wild cluster bootstrap replications for the sup-Wald p-value.
    trim : tuple[float, float]
        Quantile bounds for the threshold grid search.
    seed : int
        Random seed for reproducibility.
    min_obs : int
        Minimum valid (non-NaN) rows required to estimate a country.
        Countries below this cutoff are silently skipped.

    Returns
    -------
    dict mapping each country code (str) to a result dict with keys:
        beta_M, beta_T, c_hat, c_ci_lo, c_ci_hi, sup_wald_pvalue, n, lr_path
    or {"error": <message>} if estimation failed for that country.
    """
    out: dict[str, dict] = {}
    sub_all = df.dropna(subset=[y, x, z, unit_col])

    for iso3, sub in sub_all.groupby(unit_col):
        if len(sub) < min_obs:
            continue
        try:
            out[str(iso3)] = fit_threshold_regression(
                sub,
                y=y,
                x=x,
                z=z,
                unit_col=unit_col,
                n_boot=n_boot,
                trim=trim,
                seed=seed,
            )
        except Exception as e:  # noqa: BLE001
            # Record failure and continue rather than crashing the whole run.
            out[str(iso3)] = {"error": str(e)}

    return out
