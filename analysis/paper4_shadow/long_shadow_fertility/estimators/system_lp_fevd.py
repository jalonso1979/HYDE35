"""System local-projection FEVD on a country-year panel.

Approach (mirrors Paper 3 v2 and Paper 4 v2 house tool):
1. Per-variable reduced-form panel regression: each y_i on lags of all variables
   plus country FE. Residuals collected per (c, t).
2. Pooled residual covariance Sigma estimated across (c, t).
3. Cholesky decomposition Sigma = P P'. Structural shocks u_t = P^{-1} eps_t
   identify under the Cholesky ordering (first variable exogenous, last reactive).
4. For each (outcome i, shock j, horizon h): regress y_{i,c,t+h} on u_{j,c,t}
   plus country FE -> Theta_{ij,h} = impulse response.
5. FEVD share at horizon h: sum_{s=0..h} Theta_{ij,s}^2 / sum_{j'} sum_{s} Theta_{ij',s}^2.
"""
from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _reduced_form_residuals(
    df: pd.DataFrame,
    variables: Sequence[str],
    p: int,
    unit_col: str,
) -> pd.DataFrame:
    """Per-variable VAR(p) reduced-form on the panel, returning residual columns."""
    df = df.sort_values([unit_col, "year"]).copy()
    lag_cols: list[str] = []
    for v in variables:
        for k in range(1, p + 1):
            col = f"{v}_lag{k}"
            df[col] = df.groupby(unit_col)[v].shift(k)
            lag_cols.append(col)

    unit_dums = pd.get_dummies(df[unit_col], drop_first=True, dtype=float)
    keep = df.dropna(subset=variables + lag_cols)
    unit_dums = unit_dums.loc[keep.index]

    resid = pd.DataFrame(index=keep.index, columns=variables, dtype=float)
    X = sm.add_constant(pd.concat([keep[lag_cols].astype(float), unit_dums], axis=1))
    X_np = X.to_numpy()
    for v in variables:
        y = keep[v].astype(float).to_numpy()
        res = sm.OLS(y, X_np).fit()
        resid[v] = y - res.fittedvalues
    out = keep[[unit_col, "year"]].copy()
    for v in variables:
        out[f"e_{v}"] = resid[v].to_numpy()
    return out


def fit_system_lp_fevd(
    df: pd.DataFrame,
    variables: Sequence[str],
    horizons: Iterable[int],
    p: int = 2,
    unit_col: str = "iso3",
) -> dict:
    """Estimate Cholesky-ordered system LP-FEVD on a panel.

    Parameters
    ----------
    df : panel with unit_col, year, and one column per variable
    variables : list of variable names in Cholesky order (first = most exogenous)
    horizons : iterable of non-negative ints
    p : number of VAR lags for reduced-form residualization
    """
    variables = list(variables)
    horizons = list(horizons)
    n_vars = len(variables)
    H = len(horizons)

    resid_df = _reduced_form_residuals(df, variables, p=p, unit_col=unit_col)
    eps = resid_df[[f"e_{v}" for v in variables]].to_numpy()
    Sigma = (eps.T @ eps) / len(eps)
    P = np.linalg.cholesky(Sigma)
    Pinv = np.linalg.inv(P)
    u = eps @ Pinv.T

    shock_df = resid_df[[unit_col, "year"]].copy()
    for j, v in enumerate(variables):
        shock_df[f"u_{v}"] = u[:, j]

    df_with_shocks = df.merge(shock_df, on=[unit_col, "year"], how="left").sort_values(
        [unit_col, "year"]
    )

    irf: dict[tuple[str, str], list[float]] = {}
    irf_se: dict[tuple[str, str], list[float]] = {}

    unit_dums_full = pd.get_dummies(df_with_shocks[unit_col], drop_first=True, dtype=float)

    for v_out in variables:
        for v_shock in variables:
            betas: list[float] = []
            ses: list[float] = []
            for h in horizons:
                tmp = df_with_shocks.copy()
                tmp[f"{v_out}_h{h}"] = tmp.groupby(unit_col)[v_out].shift(-h)
                keep_idx = tmp.dropna(subset=[f"{v_out}_h{h}", f"u_{v_shock}"]).index
                if len(keep_idx) < 10:
                    betas.append(np.nan)
                    ses.append(np.nan)
                    continue
                udums = unit_dums_full.loc[keep_idx]
                X = sm.add_constant(pd.concat([
                    tmp.loc[keep_idx, [f"u_{v_shock}"]].astype(float),
                    udums,
                ], axis=1))
                y = tmp.loc[keep_idx, f"{v_out}_h{h}"].astype(float).to_numpy()
                cluster = tmp.loc[keep_idx, unit_col].astype("category").cat.codes.to_numpy()
                res = sm.OLS(y, X.to_numpy()).fit(
                    cov_type="cluster", cov_kwds={"groups": cluster}
                )
                betas.append(float(res.params[1]))
                ses.append(float(res.bse[1]))
            irf[(v_out, v_shock)] = betas
            irf_se[(v_out, v_shock)] = ses

    fevd: dict[str, np.ndarray] = {}
    for v_out in variables:
        contribs = np.zeros((n_vars, H))
        for j, v_shock in enumerate(variables):
            theta = np.array(irf[(v_out, v_shock)], dtype=float)
            theta = np.where(np.isnan(theta), 0.0, theta)
            for h_idx, h in enumerate(horizons):
                contribs[j, h_idx] = (theta[:h_idx + 1] ** 2).sum()
        denom = contribs.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        fevd[v_out] = contribs / denom

    return {
        "irf": irf,
        "irf_se": irf_se,
        "fevd": fevd,
        "Sigma": Sigma,
        "P": P,
        "variables": variables,
        "horizons": horizons,
    }
