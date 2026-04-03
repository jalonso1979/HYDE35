"""Structural break detection in the Malthusian coefficient (Paper 2)."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def chow_test(y: np.ndarray, X: np.ndarray, break_idx: int) -> tuple[float, float]:
    n = len(y)
    k = X.shape[1]
    ols_full = sm.OLS(y, X).fit()
    rss_full = ols_full.ssr
    ols_1 = sm.OLS(y[:break_idx], X[:break_idx]).fit()
    ols_2 = sm.OLS(y[break_idx:], X[break_idx:]).fit()
    rss_sub = ols_1.ssr + ols_2.ssr
    f_stat = ((rss_full - rss_sub) / k) / (rss_sub / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
    return f_stat, p_value


def detect_structural_breaks(
    panel: pd.DataFrame,
    entity: str,
    dep_var: str = "pop_growth_rate",
    key_var: str = "popdens_lag",
    entity_col: str = "country",
    min_segment: int = 3,
    significance: float = 0.05,
) -> list[dict]:
    df = panel[panel[entity_col] == entity].sort_values("year").dropna(subset=[dep_var, key_var])
    if len(df) < 2 * min_segment:
        return []
    y = df[dep_var].values
    X = sm.add_constant(df[key_var].values)
    years = df["year"].values
    breaks = []
    for i in range(min_segment, len(df) - min_segment):
        f_stat, p_val = chow_test(y, X, i)
        if p_val < significance:
            breaks.append({"break_year": int(years[i]), "f_stat": f_stat, "p_value": p_val})
    return breaks


def detect_breaks_all_entities(
    panel: pd.DataFrame,
    entity_col: str = "country",
    **kwargs,
) -> pd.DataFrame:
    all_breaks = []
    for entity in panel[entity_col].unique():
        breaks = detect_structural_breaks(panel, entity, entity_col=entity_col, **kwargs)
        for b in breaks:
            b[entity_col] = entity
            all_breaks.append(b)
    if not all_breaks:
        return pd.DataFrame(columns=[entity_col, "break_year", "f_stat", "p_value"])
    return pd.DataFrame(all_breaks)
