"""Shapley R² decomposition for variance attribution across substrates.

The Shapley value of substrate s, given controls X, is the average over
all orderings of substrates of the marginal R² contribution of adding s.
Equivalent closed form:
    phi_s = sum_{S subset of substrates without s} (|S|!*(k-|S|-1)!/k!) * [R²(S+{s}) - R²(S)]
where k is the number of substrates.

The decomposition has the property that sum_s phi_s == R²(full) - R²(baseline).

Grouped/coalition substrates: each element of `substrates` may be either a
single column name (str) or a tuple/list of column names (a coalition player).
When a coalition player is in the active subset, all of its variables are
added to the regression together; when out, none are. The decomposition
identity above is preserved.
"""
from __future__ import annotations

from itertools import chain, combinations
from math import factorial
from typing import Sequence, Union

import pandas as pd
import statsmodels.api as sm

SubstrateSpec = Union[str, Sequence[str]]


def _expand(spec: SubstrateSpec) -> list[str]:
    """Normalise a substrate spec to a list of column names."""
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _r2(df: pd.DataFrame, y_col: str, regressors: list[str]) -> float:
    if not regressors:
        return 0.0
    X = sm.add_constant(df[regressors])
    res = sm.OLS(df[y_col], X).fit()
    return float(res.rsquared)


def _spec_key(spec: SubstrateSpec) -> str:
    """Stable, human-readable key for a substrate spec."""
    if isinstance(spec, str):
        return spec
    return "+".join(spec)


def shapley_r2_decomposition(
    df: pd.DataFrame,
    y_col: str,
    substrates: Sequence[SubstrateSpec],
    controls: Sequence[str],
) -> dict:
    """Compute Shapley R² decomposition over substrates, conditioning on controls.

    Each element of `substrates` is either a column name or a sequence of
    column names (a coalition player added/removed together).

    Returns a dict with keys:
      - shapley: dict[substrate_key -> shapley value], where substrate_key is
        the original string for individual substrates, or "col1+col2+..."
        for coalition players.
      - full_model_r2: R² of regressing y on all substrate columns + controls
      - baseline_r2: R² of regressing y on controls alone
      - n_obs: number of observations after listwise deletion
    """
    expanded = [_expand(s) for s in substrates]
    keys = [_spec_key(s) for s in substrates]
    all_substrate_cols = [c for cols in expanded for c in cols]
    needed = list({y_col, *all_substrate_cols, *controls})
    df = df.dropna(subset=needed).copy()

    k = len(substrates)
    baseline_regressors = list(controls)
    baseline_r2 = _r2(df, y_col, baseline_regressors)

    # Enumerate subsets of substrate-indices and compute R²(controls + union of subset cols)
    indices = tuple(range(k))
    subset_r2: dict[tuple, float] = {}
    for subset in _powerset(indices):
        cols = [c for i in subset for c in expanded[i]]
        subset_r2[subset] = _r2(df, y_col, baseline_regressors + cols)

    full_model_r2 = subset_r2[indices]

    shapley = {key: 0.0 for key in keys}
    for i, key in enumerate(keys):
        others = tuple(j for j in indices if j != i)
        for subset in _powerset(others):
            without_key = subset
            with_key = tuple(sorted(set(subset) | {i}))
            marginal = subset_r2[with_key] - subset_r2[without_key]
            weight = factorial(len(subset)) * factorial(k - len(subset) - 1) / factorial(k)
            shapley[key] += weight * marginal

    return {
        "shapley": shapley,
        "full_model_r2": full_model_r2,
        "baseline_r2": baseline_r2,
        "n_obs": len(df),
    }
