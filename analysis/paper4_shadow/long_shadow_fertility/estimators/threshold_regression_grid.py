"""Phase 10 Pillar B1: Hansen threshold regression grid for multiple development proxies.

Wraps fit_threshold_regression to iterate over candidate threshold variables (z),
fitting each on its own non-null subset of the panel and collecting results.

Usage:
    from long_shadow_fertility.estimators.threshold_regression_grid import fit_threshold_grid

    results = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=["log_real_wage", "log_cdr", "log_gdppc", "log_tfr"],
        n_boot=500, seed=0,
    )
    # results["log_real_wage"]["c_hat"] -> threshold estimate

Note: candidates whose column is absent from df are silently skipped (not in result).
"""
from __future__ import annotations

import warnings
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression import (
    fit_threshold_regression,
)


def fit_threshold_grid(
    df: pd.DataFrame,
    y: str,
    x: str,
    z_candidates: list[str],
    unit_col: str = "iso3",
    n_boot: int = 500,
    trim: tuple[float, float] = (0.15, 0.85),
    seed: int = 0,
) -> dict[str, dict]:
    """Run Hansen threshold regression for each candidate z in z_candidates.

    Each candidate z is fit on its own non-null subset of [y, x, z, unit_col].
    Candidates not present as columns in df are silently skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data frame (long format, with unit_col identifying the cross-sectional unit).
    y : str
        Outcome variable column name.
    x : str
        Regressor column name (the slope that switches regimes).
    z_candidates : list[str]
        Column names to try as the threshold variable.
    unit_col : str
        Column identifying the panel unit (default "iso3").
    n_boot : int
        Number of wild cluster bootstrap replications for the sup-Wald p-value.
    trim : tuple[float, float]
        Quantile trim for the threshold grid search.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict mapping z_name -> the dict returned by fit_threshold_regression
    (keys: beta_M, beta_T, c_hat, c_ci_lo, c_ci_hi, sup_wald_pvalue, n, lr_path).
    """
    results: dict[str, dict] = {}
    for z in z_candidates:
        if z not in df.columns:
            warnings.warn(
                f"fit_threshold_grid: column '{z}' not found in df — skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        sub = df.dropna(subset=[y, x, z, unit_col]).copy()
        if len(sub) < 30:
            warnings.warn(
                f"fit_threshold_grid: only {len(sub)} rows after dropping NaN for z='{z}'"
                " — skipping (need ≥30).",
                UserWarning,
                stacklevel=2,
            )
            continue
        results[z] = fit_threshold_regression(
            sub, y=y, x=x, z=z,
            unit_col=unit_col,
            n_boot=n_boot,
            trim=trim,
            seed=seed,
        )
    return results
