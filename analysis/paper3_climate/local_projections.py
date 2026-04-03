"""Local projection (Jordà) impulse response functions for Paper 3."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_local_projection(
    panel: pd.DataFrame,
    shock_var: str,
    response_var: str,
    entity_col: str = "region",
    max_horizon: int = 10,
    control_vars: list[str] | None = None,
    n_lags: int = 2,
) -> pd.DataFrame:
    if control_vars is None:
        control_vars = []
    panel = panel.sort_values([entity_col, "year"]).copy()
    results = []
    for h in range(max_horizon + 1):
        rows = []
        for entity, grp in panel.groupby(entity_col):
            grp = grp.sort_values("year").reset_index(drop=True)
            for i in range(n_lags, len(grp) - h):
                t_row = grp.iloc[i]
                th_row = grp.iloc[i + h]
                if i == 0:
                    continue
                y_prev = grp.iloc[i - 1][response_var]
                y_future = th_row[response_var]
                dy = y_future - y_prev
                row = {"dy": dy, "shock": t_row[shock_var], "entity": entity}
                for lag in range(1, n_lags + 1):
                    if i - lag >= 0:
                        row[f"y_lag{lag}"] = grp.iloc[i - lag][response_var]
                    else:
                        row[f"y_lag{lag}"] = np.nan
                for cv in control_vars:
                    row[cv] = t_row[cv]
                rows.append(row)
        hdf = pd.DataFrame(rows).dropna()
        if len(hdf) < 10:
            continue
        for col in ["dy", "shock"] + [f"y_lag{l}" for l in range(1, n_lags + 1)] + control_vars:
            if col in hdf.columns:
                hdf[col] = hdf[col] - hdf.groupby("entity")[col].transform("mean")
        y = hdf["dy"]
        X_cols = ["shock"] + [f"y_lag{l}" for l in range(1, n_lags + 1)] + control_vars
        X_cols = [c for c in X_cols if c in hdf.columns]
        X = sm.add_constant(hdf[X_cols])
        model = sm.OLS(y, X).fit(cov_type="HC1")
        coef = model.params["shock"]
        se = model.bse["shock"]
        results.append({
            "horizon": h,
            "coefficient": coef,
            "std_err": se,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "pvalue": model.pvalues["shock"],
        })
    return pd.DataFrame(results)
