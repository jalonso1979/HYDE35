"""Scenario aggregation and uncertainty quantification."""
import numpy as np
import pandas as pd


def _std_ddof0(x: pd.Series) -> float:
    """Population standard deviation (ddof=0), matching HYDE convention."""
    return float(np.std(x.to_numpy(dtype="float64"), ddof=0))


def compute_scenario_stats(
    scenario_panel: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, std, and standard error across scenarios."""
    if group_cols is None:
        group_cols = [c for c in scenario_panel.columns if c not in ("scenario", "value")]
    agg = scenario_panel.groupby(group_cols, as_index=False)["value"].agg(
        mean="mean",
        std=_std_ddof0,
    )
    n_scenarios = scenario_panel["scenario"].nunique()
    agg["se"] = agg["std"] / np.sqrt(n_scenarios)
    return agg


def compute_manski_bounds(
    coefficients: dict[str, float],
) -> tuple[float, float]:
    """Compute Manski-style bounds from coefficients estimated across scenarios."""
    vals = list(coefficients.values())
    return min(vals), max(vals)
