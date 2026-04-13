"""Multinomial logit for pathway prediction (Exercise 1).

Does climate seasonality predict which agricultural pathway a country ends up in?
"""

import pandas as pd
import statsmodels.api as sm


def run_pathway_multinomial(
    df,
    pathway_col="pathway",
    climate_vars=None,
    control_vars=None,
):
    """Run statsmodels MNLogit predicting pathway from climate variables.

    Parameters
    ----------
    df : DataFrame
        Must contain ``pathway_col`` and all variables listed in
        ``climate_vars`` and ``control_vars``.
    pathway_col : str
        Column with integer pathway assignments.
    climate_vars : list[str] or None
        Climate regressors. Defaults to ``["seasonality"]``.
    control_vars : list[str] or None
        Additional control regressors.

    Returns
    -------
    dict with keys: model, summary, predictions, pvalues, params.
    """
    if climate_vars is None:
        climate_vars = ["seasonality"]
    if control_vars is None:
        control_vars = []

    all_vars = climate_vars + control_vars
    X = sm.add_constant(df[all_vars].astype(float))
    y = df[pathway_col].astype(int)

    mnlogit = sm.MNLogit(y, X)
    try:
        model = mnlogit.fit(disp=False, method="newton", maxiter=200)
    except Exception:
        # Fall back to bfgs when Newton fails (e.g. near-perfect separation)
        model = mnlogit.fit(disp=False, method="bfgs", maxiter=200)

    predictions = model.predict(X)

    return {
        "model": model,
        "summary": model.summary(),
        "predictions": predictions,
        "pvalues": model.pvalues,
        "params": model.params,
    }


def compute_marginal_effects(model, df, climate_vars):
    """Compute marginal effects for climate variables.

    Parameters
    ----------
    model : MNLogitResults
        Fitted multinomial logit model.
    df : DataFrame
        Data used for estimation (needed for fallback).
    climate_vars : list[str]
        Variables for which to report marginal effects.

    Returns
    -------
    DataFrame with columns: variable, pathway, marginal_effect.
    """
    try:
        me = model.get_margeff(at="overall", method="dydx")
        # me.margeff shape: (n_vars, n_outcomes).
        # We report only non-base pathways to match params layout.
        records = []
        param_names = [n for n in model.model.exog_names if n != "const"]
        # Non-base pathway indices (skip first / base category)
        n_outcomes = me.margeff.shape[1]
        for i, var in enumerate(param_names):
            if var not in climate_vars:
                continue
            for j in range(1, n_outcomes):
                records.append({
                    "variable": var,
                    "pathway": j,
                    "marginal_effect": me.margeff[i, j],
                })
        return pd.DataFrame(records)
    except Exception:
        # Fallback: extract from params directly.
        # params columns correspond to non-base pathways.
        params = model.params
        records = []
        for var in climate_vars:
            if var in params.index:
                for col in params.columns:
                    records.append({
                        "variable": var,
                        "pathway": col,
                        "marginal_effect": params.loc[var, col],
                    })
        return pd.DataFrame(records)
