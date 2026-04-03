"""Core Malthusian regressions and rolling window estimation for Paper 2."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

def run_fe_regression(panel: pd.DataFrame, dep_var: str, indep_vars: list[str], entity_col: str = "country") -> dict:
    df = panel.dropna(subset=[dep_var] + indep_vars).copy()
    for col in [dep_var] + indep_vars:
        entity_means = df.groupby(entity_col)[col].transform("mean")
        df[f"{col}_dm"] = df[col] - entity_means
    y = df[f"{dep_var}_dm"]
    X = df[[f"{v}_dm" for v in indep_vars]]
    X = sm.add_constant(X)
    model = OLS(y, X).fit()
    params = {}
    pvalues = {}
    for orig, dm in zip(indep_vars, [f"{v}_dm" for v in indep_vars]):
        params[orig] = model.params[dm]
        pvalues[orig] = model.pvalues[dm]
    return {"params": params, "pvalues": pvalues, "rsquared": model.rsquared, "nobs": int(model.nobs), "summary": model.summary()}

def run_rolling_window(panel: pd.DataFrame, dep_var: str, key_var: str, control_vars: list[str], entity_col: str = "country", window_years: int = 400, step_years: int = 100) -> pd.DataFrame:
    years = sorted(panel["year"].unique())
    min_year, max_year = min(years), max(years)
    results = []
    start = min_year
    while start + window_years <= max_year + step_years:
        end = start + window_years
        center = start + window_years // 2
        window_df = panel[(panel["year"] >= start) & (panel["year"] < end)]
        if len(window_df) < 10:
            start += step_years
            continue
        all_vars = [key_var] + control_vars
        try:
            reg = run_fe_regression(window_df, dep_var, all_vars, entity_col)
            results.append({"center_year": center, "coefficient": reg["params"][key_var], "pvalue": reg["pvalues"][key_var], "nobs": reg["nobs"]})
        except Exception:
            pass
        start += step_years
    return pd.DataFrame(results)
