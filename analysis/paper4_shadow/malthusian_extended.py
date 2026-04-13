"""Extended Malthusian regressions stratified by pathway cluster (Exercise 3)."""

import pandas as pd

from analysis.paper2_malthus.regressions import run_fe_regression, run_rolling_window


def run_malthusian_by_pathway(
    panel: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    dep_var: str = "pop_growth",
    density_var: str = "log_density",
    volatility_var: str = "temp_volatility",
    land_labor_var: str = "log_land_labor",
    entity_col: str = "country",
    pathway_col: str = "cluster",
) -> pd.DataFrame:
    """Run fixed-effects Malthusian regression separately for each pathway cluster.

    Parameters
    ----------
    panel : DataFrame
        Panel data with columns for dep_var, density_var, and entity_col.
    pathway_assignments : DataFrame
        Must contain entity_col and pathway_col columns.
    dep_var, density_var, volatility_var, land_labor_var : str
        Column names for the dependent variable, density, volatility, and
        land-labor ratio regressors.
    entity_col : str
        Entity identifier for fixed effects.
    pathway_col : str
        Column in pathway_assignments identifying the cluster/pathway.

    Returns
    -------
    DataFrame with one row per pathway: pathway, beta_density, pval_density,
    nobs, rsquared, and optionally gamma_volatility / pval_volatility.
    """
    merged = panel.merge(
        pathway_assignments[[entity_col, pathway_col]].drop_duplicates(),
        on=entity_col,
        how="inner",
    )

    rows = []
    for pathway, grp in merged.groupby(pathway_col):
        # Build regressor list: always include density
        indep_vars = [density_var]

        # Check whether volatility column is usable
        has_volatility = (
            volatility_var in grp.columns
            and grp[volatility_var].notna().sum() > 10
        )
        if has_volatility:
            indep_vars.append(volatility_var)

        # Add land-labor ratio if present
        if land_labor_var in grp.columns:
            indep_vars.append(land_labor_var)

        try:
            reg = run_fe_regression(grp, dep_var, indep_vars, entity_col)
        except Exception:
            continue

        row = {
            "pathway": pathway,
            "beta_density": reg["params"][density_var],
            "pval_density": reg["pvalues"][density_var],
            "nobs": reg["nobs"],
            "rsquared": reg["rsquared"],
        }
        if has_volatility:
            row["gamma_volatility"] = reg["params"][volatility_var]
            row["pval_volatility"] = reg["pvalues"][volatility_var]

        rows.append(row)

    return pd.DataFrame(rows)


def run_rolling_by_pathway(
    panel: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    dep_var: str = "pop_growth",
    key_var: str = "log_density",
    control_vars: list[str] | None = None,
    entity_col: str = "country",
    pathway_col: str = "cluster",
    window_years: int = 400,
    step_years: int = 100,
) -> pd.DataFrame:
    """Run rolling-window regressions separately for each pathway cluster.

    Parameters
    ----------
    panel : DataFrame
        Panel data with year column and regression variables.
    pathway_assignments : DataFrame
        Must contain entity_col and pathway_col columns.
    dep_var, key_var : str
        Dependent variable and key regressor for the rolling window.
    control_vars : list[str] or None
        Additional control variables. Defaults to ["log_land_labor"].
    entity_col : str
        Entity identifier for fixed effects.
    pathway_col : str
        Column identifying the cluster/pathway.
    window_years, step_years : int
        Rolling window width and step size in years.

    Returns
    -------
    Concatenated DataFrame from run_rolling_window with an added "pathway" column.
    """
    if control_vars is None:
        control_vars = ["log_land_labor"]

    merged = panel.merge(
        pathway_assignments[[entity_col, pathway_col]].drop_duplicates(),
        on=entity_col,
        how="inner",
    )

    frames = []
    for pathway, grp in merged.groupby(pathway_col):
        result = run_rolling_window(
            grp, dep_var, key_var, control_vars, entity_col,
            window_years, step_years,
        )
        if not result.empty:
            result = result.copy()
            result["pathway"] = pathway
            frames.append(result)

    if not frames:
        return pd.DataFrame(columns=["center_year", "coefficient", "pvalue", "nobs", "pathway"])

    return pd.concat(frames, ignore_index=True)
