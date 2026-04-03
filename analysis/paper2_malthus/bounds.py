"""Partial identification bounds using HYDE scenario variation (Paper 2)."""
import pandas as pd
from analysis.shared.config import SCENARIO_LABELS
from analysis.shared.uncertainty import compute_manski_bounds
from analysis.paper2_malthus.regressions import run_fe_regression


def estimate_bounds(
    scenario_panels: dict[str, pd.DataFrame],
    dep_var: str = "pop_growth_rate",
    indep_vars: list[str] | None = None,
    key_var: str = "popdens_lag",
    entity_col: str = "country",
) -> dict:
    if indep_vars is None:
        indep_vars = [key_var, "land_labor_ratio_lag"]
    scenario_coefs = {}
    scenario_results = {}
    for label, panel in scenario_panels.items():
        reg = run_fe_regression(panel, dep_var, indep_vars, entity_col)
        scenario_coefs[label] = reg["params"][key_var]
        scenario_results[label] = {
            "coef": reg["params"][key_var],
            "pvalue": reg["pvalues"][key_var],
            "nobs": reg["nobs"],
        }
    lb, ub = compute_manski_bounds(scenario_coefs)
    return {"bounds": (lb, ub), "scenario_results": scenario_results}


def bounds_summary_table(bounds_result: dict) -> pd.DataFrame:
    rows = []
    for label, res in bounds_result["scenario_results"].items():
        rows.append({"scenario": label, **res})
    lb, ub = bounds_result["bounds"]
    rows.append({"scenario": "BOUNDS", "coef": f"[{lb:.6f}, {ub:.6f}]", "pvalue": None, "nobs": None})
    return pd.DataFrame(rows)
