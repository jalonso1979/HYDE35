"""Dual-channel separation regressions for Exercise 2.

Tests that intra-annual seasonality and inter-annual volatility operate
through separate causal channels:
  - Stage 1: Seasonality predicts pathway assignment
  - Stage 2: Volatility predicts Malthusian intensity within pathway
"""

import statsmodels.api as sm


def _run_ols(df, dep_col, seasonality_col, volatility_col, control_vars):
    """Run OLS with HC1 standard errors and return standardised result dict."""
    regressors = [seasonality_col, volatility_col]
    if control_vars:
        regressors = regressors + list(control_vars)

    X = sm.add_constant(df[regressors])
    y = df[dep_col]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    return {
        "seasonality_coef": model.params[seasonality_col],
        "seasonality_pval": model.pvalues[seasonality_col],
        "volatility_coef": model.params[volatility_col],
        "volatility_pval": model.pvalues[volatility_col],
        "rsquared": model.rsquared,
        "nobs": int(model.nobs),
        "model": model,
    }


def run_dual_channel_pathway(
    df,
    pathway_col="pathway",
    seasonality_col="seasonality",
    volatility_col="volatility",
    control_vars=None,
):
    """OLS: pathway ~ seasonality + volatility + controls (HC1 SEs).

    Stage 1 of the dual-channel test. Seasonality should predict pathway
    assignment while volatility should not.
    """
    return _run_ols(df, pathway_col, seasonality_col, volatility_col, control_vars)


def run_dual_channel_malthusian(
    df,
    beta_col="malthusian_beta",
    seasonality_col="seasonality",
    volatility_col="volatility",
    control_vars=None,
):
    """OLS: malthusian_beta ~ seasonality + volatility + controls (HC1 SEs).

    Stage 2 of the dual-channel test. Volatility should predict Malthusian
    intensity while seasonality should not.
    """
    return _run_ols(df, beta_col, seasonality_col, volatility_col, control_vars)
