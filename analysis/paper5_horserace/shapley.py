"""Shapley R² decomposition for variance attribution across substrates.

The Shapley value of substrate s, given controls X, is the average over
all orderings of substrates of the marginal R² contribution of adding s.
Equivalent closed form:
    phi_s = sum_{S subset of substrates without s} (|S|!*(k-|S|-1)!/k!) * [R²(S+{s}) - R²(S)]
where k is the number of substrates.

The decomposition has the property that sum_s phi_s == R²(full) - R²(baseline).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
from math import factorial
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _r2(df: pd.DataFrame, y_col: str, regressors: list[str]) -> float:
    if not regressors:
        return 0.0
    X = sm.add_constant(df[regressors])
    res = sm.OLS(df[y_col], X).fit()
    return float(res.rsquared)


def shapley_r2_decomposition(
    df: pd.DataFrame,
    y_col: str,
    substrates: Sequence[str],
    controls: Sequence[str],
) -> dict:
    """Compute Shapley R² decomposition over substrates, conditioning on controls.

    Returns a dict with keys:
      - shapley: dict[substrate -> shapley value]
      - full_model_r2: R² of regressing y on all substrates + controls
      - baseline_r2: R² of regressing y on controls alone
      - n_obs: number of observations after listwise deletion
    """
    needed = list({y_col, *substrates, *controls})
    df = df.dropna(subset=needed).copy()
    k = len(substrates)
    baseline_regressors = list(controls)
    baseline_r2 = _r2(df, y_col, baseline_regressors)

    # Compute R²(S + controls) for every subset S of substrates
    subset_r2 = {}
    for subset in _powerset(substrates):
        regressors = baseline_regressors + list(subset)
        subset_r2[subset] = _r2(df, y_col, regressors)

    full_model_r2 = subset_r2[tuple(substrates)]

    shapley = {s: 0.0 for s in substrates}
    for s in substrates:
        others = [t for t in substrates if t != s]
        for subset in _powerset(others):
            without = subset
            with_s = tuple(sorted(set(subset) | {s}, key=substrates.index))
            # Recompute the sorted key so subset_r2 lookups match
            without_key = tuple(t for t in substrates if t in without)
            with_key = tuple(t for t in substrates if t in set(without) | {s})
            marginal = subset_r2[with_key] - subset_r2[without_key]
            weight = factorial(len(subset)) * factorial(k - len(subset) - 1) / factorial(k)
            shapley[s] += weight * marginal

    return {
        "shapley": shapley,
        "full_model_r2": full_model_r2,
        "baseline_r2": baseline_r2,
        "n_obs": len(df),
    }
