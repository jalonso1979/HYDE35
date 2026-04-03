"""Regime-switching and threshold models for Paper 3."""
import numpy as np
import pandas as pd
from analysis.paper3_climate.local_projections import run_local_projection


def split_sample_by_regime(panel: pd.DataFrame, regime_var: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = panel[panel[regime_var] < threshold].copy()
    post = panel[panel[regime_var] >= threshold].copy()
    return pre, post


def interaction_local_projection(panel: pd.DataFrame, shock_var: str, response_var: str, regime_var: str, entity_col: str = "region", max_horizon: int = 10) -> pd.DataFrame:
    import statsmodels.api as sm
    panel = panel.sort_values([entity_col, "year"]).copy()
    panel["shock_x_regime"] = panel[shock_var] * panel[regime_var]
    results = []
    for h in range(max_horizon + 1):
        rows = []
        for entity, grp in panel.groupby(entity_col):
            grp = grp.sort_values("year").reset_index(drop=True)
            for i in range(1, len(grp) - h):
                y_prev = grp.iloc[i - 1][response_var]
                y_future = grp.iloc[i + h][response_var]
                t_row = grp.iloc[i]
                rows.append({"dy": y_future - y_prev, "shock": t_row[shock_var], "regime": t_row[regime_var], "shock_x_regime": t_row["shock_x_regime"], "entity": entity})
        hdf = pd.DataFrame(rows).dropna()
        if len(hdf) < 10:
            continue
        for col in ["dy", "shock", "regime", "shock_x_regime"]:
            hdf[col] = hdf[col] - hdf.groupby("entity")[col].transform("mean")
        y = hdf["dy"]
        X = sm.add_constant(hdf[["shock", "regime", "shock_x_regime"]])
        model = sm.OLS(y, X).fit(cov_type="HC1")
        results.append({"horizon": h, "coef_shock": model.params.get("shock", np.nan), "coef_interaction": model.params.get("shock_x_regime", np.nan), "pval_shock": model.pvalues.get("shock", np.nan), "pval_interaction": model.pvalues.get("shock_x_regime", np.nan)})
    return pd.DataFrame(results)
