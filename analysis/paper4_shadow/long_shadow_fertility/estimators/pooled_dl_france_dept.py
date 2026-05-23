"""Pooled distributed-lag OLS on the France dept-year panel.

Dept + (optional) year fixed effects, SE clustered by dept. Identical math
to `pooled_distributed_lag.fit_pooled_distributed_lag` but with `unit_col`
defaulting to `dep` (departement code) for clarity at call sites.
"""
from __future__ import annotations
from typing import Sequence
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.estimators.pooled_distributed_lag import (
    fit_pooled_distributed_lag,
)


def fit_dept_distributed_lag(
    df: pd.DataFrame,
    y: str,
    x: str,
    lags: int = 5,
    controls: Sequence[str] | None = None,
    year_fe: bool = False,
) -> pd.DataFrame:
    """Pooled DL on (dep, year) panel; thin wrapper around the multi-country fitter.

    Parameters
    ----------
    df : panel with columns `dep`, `year`, plus `y`, `x`, and `controls`.
    year_fe : if True, absorbs all national-level shocks (national wages,
        national mortality, national policy). Spec 1 in the design = False;
        Spec 2 = True.
    """
    return fit_pooled_distributed_lag(
        df, y=y, x=x, lags=lags, unit_col="dep",
        controls=controls, year_fe=year_fe,
    )
