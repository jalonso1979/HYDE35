"""Pathway-stratified Jordà local-projection IRFs for Exercise 4."""

import pandas as pd

from analysis.paper3_climate.local_projections import run_local_projection


def run_irfs_by_pathway(
    panel: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    shock_var: str,
    response_var: str,
    entity_col: str = "country_id",
    pathway_col: str = "cluster",
    max_horizon: int = 10,
    n_lags: int = 2,
) -> dict[str, pd.DataFrame]:
    """Run local projections separately for each agricultural pathway cluster.

    Parameters
    ----------
    panel : DataFrame
        Panel data with entity, year, shock, and response columns.
    pathway_assignments : DataFrame
        Mapping from entity to pathway cluster.  May use ``"country"``
        instead of *entity_col* — the function handles the rename.
    shock_var, response_var : str
        Column names for the shock and outcome variable.
    entity_col : str
        Entity identifier in *panel*.
    pathway_col : str
        Column in *pathway_assignments* that holds the cluster label.
    max_horizon, n_lags : int
        Passed through to :func:`run_local_projection`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from pathway label to IRF DataFrame.
    """
    # --- harmonise entity column name in pathway_assignments ---------------
    pa = pathway_assignments.copy()
    if entity_col not in pa.columns and "country" in pa.columns:
        pa = pa.rename(columns={"country": entity_col})

    # --- merge -------------------------------------------------------------
    merged = panel.merge(pa[[entity_col, pathway_col]], on=entity_col, how="inner")

    irf_dict: dict[str, pd.DataFrame] = {}

    for pathway, grp in merged.groupby(pathway_col):
        n_entities = grp[entity_col].nunique()
        n_valid = grp[[shock_var, response_var]].dropna().shape[0]

        if n_entities < 3 or n_valid < 30:
            continue

        irf = run_local_projection(
            grp,
            shock_var=shock_var,
            response_var=response_var,
            entity_col=entity_col,
            max_horizon=max_horizon,
            n_lags=n_lags,
        )
        irf["pathway"] = pathway
        irf_dict[str(pathway)] = irf

    return irf_dict


def compare_pathway_irfs(
    irf_dict: dict[str, pd.DataFrame],
    horizon: int = 5,
) -> pd.DataFrame:
    """Compare IRF coefficients across pathways at a single horizon.

    Parameters
    ----------
    irf_dict : dict
        Output of :func:`run_irfs_by_pathway`.
    horizon : int
        The horizon step to compare.

    Returns
    -------
    pd.DataFrame
        One row per pathway with coefficient, std_err, pvalue,
        ci_lower, ci_upper.
    """
    rows = []
    for pathway, irf_df in irf_dict.items():
        row = irf_df.loc[irf_df["horizon"] == horizon].squeeze()
        if row.empty:
            continue
        rows.append(
            {
                "pathway": pathway,
                "coefficient": row["coefficient"],
                "std_err": row["std_err"],
                "pvalue": row["pvalue"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
            }
        )
    return pd.DataFrame(rows)
