"""Counterfactual simulations for Paper 3."""
import numpy as np
import pandas as pd


def simulate_counterfactual(irf_coefficients: np.ndarray, actual_shocks: np.ndarray, counterfactual_shocks: np.ndarray, baseline_response: np.ndarray) -> np.ndarray:
    T = len(baseline_response)
    H = len(irf_coefficients)
    cf_response = baseline_response.copy()
    for t in range(T):
        cumulative_diff = 0.0
        for h in range(min(H, t + 1)):
            shock_diff = counterfactual_shocks[t - h] - actual_shocks[t - h]
            cumulative_diff += irf_coefficients[h] * shock_diff
        cf_response[t] = baseline_response[t] + cumulative_diff
    return cf_response


def run_counterfactual_experiment(irf_df: pd.DataFrame, climate_panel: pd.DataFrame, response_panel: pd.DataFrame, source_entity: str, target_entity: str, shock_var: str, response_var: str, entity_col: str = "region") -> pd.DataFrame:
    irf_coefs = irf_df.sort_values("horizon")["coefficient"].values
    target_climate = climate_panel[climate_panel[entity_col] == target_entity].sort_values("year")
    source_climate = climate_panel[climate_panel[entity_col] == source_entity].sort_values("year")
    target_response = response_panel[response_panel[entity_col] == target_entity].sort_values("year")
    common_years = sorted(set(target_climate["year"]) & set(source_climate["year"]) & set(target_response["year"]))
    actual_shocks = target_climate.set_index("year").loc[common_years, shock_var].values
    cf_shocks = source_climate.set_index("year").loc[common_years, shock_var].values
    baseline = target_response.set_index("year").loc[common_years, response_var].values
    cf_path = simulate_counterfactual(irf_coefs, actual_shocks, cf_shocks, baseline)
    return pd.DataFrame({"year": common_years, "actual": baseline, "counterfactual": cf_path})
