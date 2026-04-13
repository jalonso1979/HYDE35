"""Long shadow cross-section regressions for Exercise 6.

Links pre-industrial (0-1850 CE) seasonality/volatility endowments to
modern (2015-2025) structural outcomes via one-row-per-country OLS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_long_shadow_cross_section(
    modern_panel: pd.DataFrame,
    seasonality_endowments: pd.DataFrame,
    pathway_assignments: pd.DataFrame,
    entity_col: str = "country_id",
    early_period: tuple[int, int] = (1950, 1960),
    late_period: tuple[int, int] = (2015, 2025),
) -> pd.DataFrame:
    """Build one-row-per-country cross-section for long-shadow regressions.

    Parameters
    ----------
    modern_panel : DataFrame
        Panel with *entity_col*, ``year``, ``temperature_c``, ``crop_share``,
        ``irrigation_share``, ``urban_share``, ``density``, ``pop``, and
        ``era5_region``.
    seasonality_endowments : DataFrame
        One row per region with at least ``era5_region`` and a seasonality
        column (e.g. ``seasonality_historical``).
    pathway_assignments : DataFrame
        Mapping from entity to pathway cluster.  May use ``"country"``
        instead of *entity_col* and ``"pathway"`` instead of ``"cluster"``.
    entity_col : str
        Entity identifier column in *modern_panel*.
    early_period, late_period : tuple[int, int]
        Inclusive (start, end) year bounds for early and late aggregation.

    Returns
    -------
    pd.DataFrame
        One row per entity with early/late means, changes, seasonality
        endowments, and pathway assignment.
    """
    var_map = {
        "temperature_c": "t",
        "crop_share": "cs",
        "irrigation_share": "ir",
        "urban_share": "ur",
        "density": "d",
        "pop": "p",
    }

    # --- helper: aggregate period means ------------------------------------
    def _period_means(df: pd.DataFrame, period: tuple[int, int], suffix: str):
        mask = df["year"].between(period[0], period[1])
        agg = (
            df.loc[mask]
            .groupby(entity_col)[list(var_map.keys())]
            .mean()
        )
        rename = {v: f"{short}{suffix}" for v, short in var_map.items()}
        return agg.rename(columns=rename).reset_index()

    early = _period_means(modern_panel, early_period, "0")
    late = _period_means(modern_panel, late_period, "1")

    xs = early.merge(late, on=entity_col, how="inner")

    # --- compute changes ---------------------------------------------------
    xs["warming"] = xs["t1"] - xs["t0"]
    xs["d_crop"] = xs["cs1"] - xs["cs0"]
    xs["d_irrig"] = xs["ir1"] - xs["ir0"]
    xs["d_urban"] = xs["ur1"] - xs["ur0"]

    p0 = xs["p0"].clip(lower=1)
    p1 = xs["p1"].clip(lower=1)
    xs["pop_gr"] = np.log(p1) - np.log(p0)

    # --- merge era5_region from modern_panel (first per entity) ------------
    region_map = (
        modern_panel
        .dropna(subset=["era5_region"])
        .groupby(entity_col)["era5_region"]
        .first()
        .reset_index()
    )
    xs = xs.merge(region_map, on=entity_col, how="left")

    # --- merge seasonality endowments on region ----------------------------
    xs = xs.merge(seasonality_endowments, on="era5_region", how="left")

    # --- merge pathway assignments -----------------------------------------
    pa = pathway_assignments.copy()
    if entity_col not in pa.columns and "country" in pa.columns:
        pa = pa.rename(columns={"country": entity_col})
    if "cluster" not in pa.columns and "pathway" in pa.columns:
        pa = pa.rename(columns={"pathway": "cluster"})

    pa_cols = [entity_col] + [c for c in pa.columns if c != entity_col]
    xs = xs.merge(pa[pa_cols], on=entity_col, how="left")

    return xs


def run_long_shadow_regressions(
    xs: pd.DataFrame,
    outcomes: list[str] | None = None,
    seasonality_col: str = "seasonality_historical",
    pathway_col: str = "cluster",
    include_pathway_fe: bool = False,
) -> pd.DataFrame:
    """Run cross-sectional OLS linking seasonality endowment to outcomes.

    Parameters
    ----------
    xs : DataFrame
        Cross-section from :func:`build_long_shadow_cross_section`.
    outcomes : list[str], optional
        Dependent variables.  Defaults to
        ``["d_crop", "d_irrig", "d_urban", "pop_gr"]``.
    seasonality_col : str
        Column with the historical seasonality endowment.
    pathway_col : str
        Column with pathway cluster labels (used for fixed effects).
    include_pathway_fe : bool
        If True, also run each regression with pathway dummies.

    Returns
    -------
    pd.DataFrame
        One row per regression with ``outcome``, ``pathway_fe``,
        ``seasonality_coef``, ``seasonality_pval``, ``rsquared``, ``nobs``.
    """
    if outcomes is None:
        outcomes = ["d_crop", "d_irrig", "d_urban", "pop_gr"]

    rows: list[dict] = []

    for outcome in outcomes:
        for use_fe in ([False, True] if include_pathway_fe else [False]):
            sub = xs[[seasonality_col, outcome]].copy()
            if use_fe:
                sub[pathway_col] = xs[pathway_col]

            sub = sub.dropna()
            if sub.shape[0] < 3:
                continue

            regressors = [seasonality_col]
            if use_fe:
                dummies = pd.get_dummies(
                    sub[pathway_col], prefix="pw", drop_first=True, dtype=float
                )
                sub = pd.concat([sub, dummies], axis=1)
                regressors = regressors + list(dummies.columns)

            X = sm.add_constant(sub[regressors])
            y = sub[outcome]
            model = sm.OLS(y, X).fit(cov_type="HC1")

            rows.append(
                {
                    "outcome": outcome,
                    "pathway_fe": use_fe,
                    "seasonality_coef": model.params[seasonality_col],
                    "seasonality_pval": model.pvalues[seasonality_col],
                    "rsquared": model.rsquared,
                    "nobs": int(model.nobs),
                }
            )

    return pd.DataFrame(rows)
