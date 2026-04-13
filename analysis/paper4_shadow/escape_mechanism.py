"""Escape mechanism interaction regressions for Exercise 5.

Tests whether intensification or urbanization mediates the decline in
climate sensitivity after the Green Revolution:

    growth_it = alpha_i + beta1 * shock_t + beta2 * shock_t * h_it + eps

where h = intensification_index or urban_share.  beta2 < 0 means higher h
reduces climate sensitivity (i.e. the negative shock effect shrinks).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _demean_by_entity(df, cols, entity_col):
    """Subtract entity-level means (within transformation for FE)."""
    out = df.copy()
    means = df.groupby(entity_col)[cols].transform("mean")
    out[cols] = df[cols] - means
    return out


def _run_single_interaction(panel, shock_var, outcome_var, mediator_var, entity_col):
    """Run interaction regression for a single mediator with entity FE.

    Model (demeaned):
        outcome ~ shock + mediator + shock*mediator

    Returns dict with coefficients and diagnostics, or None if <50 obs.
    """
    df = panel[[entity_col, outcome_var, shock_var, mediator_var]].dropna()
    if len(df) < 50:
        return None

    interaction_col = f"{shock_var}_x_{mediator_var}"
    df[interaction_col] = df[shock_var] * df[mediator_var]

    reg_cols = [outcome_var, shock_var, mediator_var, interaction_col]
    df = _demean_by_entity(df, reg_cols, entity_col)

    X = sm.add_constant(df[[shock_var, mediator_var, interaction_col]])
    y = df[outcome_var]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel.loc[df.index, entity_col]},
    )

    return {
        "shock_coef": model.params[shock_var],
        "shock_pval": model.pvalues[shock_var],
        "mediator_coef": model.params[mediator_var],
        "mediator_pval": model.pvalues[mediator_var],
        "interaction_coef": model.params[interaction_col],
        "interaction_pval": model.pvalues[interaction_col],
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "model": model,
    }


def _run_joint_interaction(panel, shock_var, outcome_var, mediators, entity_col):
    """Run interaction regression with all mediators simultaneously.

    Model (demeaned):
        outcome ~ shock + med1 + shock*med1 + med2 + shock*med2 + ...

    Returns dict with {med}_interaction_coef and {med}_interaction_pval
    for each mediator, plus common diagnostics.
    """
    keep_cols = [entity_col, outcome_var, shock_var] + list(mediators)
    df = panel[keep_cols].dropna()
    if len(df) < 50:
        return None

    interaction_cols = []
    for med in mediators:
        icol = f"{shock_var}_x_{med}"
        df[icol] = df[shock_var] * df[med]
        interaction_cols.append(icol)

    reg_cols = [outcome_var, shock_var] + list(mediators) + interaction_cols
    df = _demean_by_entity(df, reg_cols, entity_col)

    X = sm.add_constant(
        df[[shock_var] + list(mediators) + interaction_cols]
    )
    y = df[outcome_var]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel.loc[df.index, entity_col]},
    )

    result = {
        "shock_coef": model.params[shock_var],
        "shock_pval": model.pvalues[shock_var],
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "model": model,
    }
    for med, icol in zip(mediators, interaction_cols):
        result[f"{med}_interaction_coef"] = model.params[icol]
        result[f"{med}_interaction_pval"] = model.pvalues[icol]

    return result


def run_escape_interactions(
    panel,
    shock_var="temp_anomaly",
    outcome_var="pop_growth",
    mediators=None,
    entity_col="country_id",
):
    """Run escape-mechanism interaction regressions for each mediator and jointly.

    Parameters
    ----------
    panel : DataFrame
        Panel with entity_col, year, shock_var, outcome_var, and mediator columns.
    shock_var : str
        Climate shock variable (default ``temp_anomaly``).
    outcome_var : str
        Outcome variable (default ``pop_growth``).
    mediators : list[str] or None
        Mediator columns. Defaults to ``["urban_share", "intensification_index"]``.
    entity_col : str
        Entity identifier for fixed effects and clustering.

    Returns
    -------
    dict
        Mapping mediator_name -> results_dict, plus ``"joint"`` -> joint results.
    """
    if mediators is None:
        mediators = ["urban_share", "intensification_index"]

    results = {}
    for med in mediators:
        results[med] = _run_single_interaction(
            panel, shock_var, outcome_var, med, entity_col
        )

    results["joint"] = _run_joint_interaction(
        panel, shock_var, outcome_var, mediators, entity_col
    )

    return results


def run_rolling_with_mediator(
    panel,
    shock_var,
    outcome_var,
    mediator_var,
    entity_col,
    window=20,
    step=5,
):
    """Run interaction regression in rolling windows.

    Parameters
    ----------
    panel : DataFrame
        Must contain a ``year`` column.
    window : int
        Window width in years.
    step : int
        Step size in years between windows.

    Returns
    -------
    DataFrame
        Columns: center_year, shock_coef, interaction_coef, interaction_pval, nobs.
    """
    years = sorted(panel["year"].unique())
    min_year, max_year = years[0], years[-1]

    records = []
    start = min_year
    while start + window - 1 <= max_year:
        end = start + window - 1
        subset = panel[(panel["year"] >= start) & (panel["year"] <= end)]
        result = _run_single_interaction(
            subset.reset_index(drop=True),
            shock_var,
            outcome_var,
            mediator_var,
            entity_col,
        )
        if result is not None:
            records.append(
                {
                    "center_year": (start + end) / 2,
                    "shock_coef": result["shock_coef"],
                    "interaction_coef": result["interaction_coef"],
                    "interaction_pval": result["interaction_pval"],
                    "nobs": result["nobs"],
                }
            )
        start += step

    return pd.DataFrame(records)
