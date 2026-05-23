"""Single-equation local projection on the France dept-year panel with year FE.

At each horizon h, estimate:
    y_{d,t+h} = alpha_d + lambda_t + beta_h * shock_{d,t} + eps
with SE clustered by dept. Year FE absorbs all national-level shocks
(national wages, national mortality, national policy), so beta_h
identifies the *direct* climate effect on fertility net of any
nationally-mediated channel.
"""
from __future__ import annotations
from typing import Iterable
import pandas as pd
import statsmodels.api as sm


def fit_dept_local_projection(
    df: pd.DataFrame,
    y: str,
    shock: str,
    horizons: Iterable[int],
    unit_col: str = "dep",
) -> pd.DataFrame:
    """Local projection of y on shock at each horizon h, two-way FE, cluster SE.

    Parameters
    ----------
    df : panel with `unit_col`, `year`, `y`, `shock`
    horizons : iterable of non-negative ints
    """
    df = df.sort_values([unit_col, "year"]).copy()
    rows = []
    for h in horizons:
        sub = df.copy()
        sub[f"{y}_h{h}"] = sub.groupby(unit_col)[y].shift(-h)
        keep = sub.dropna(subset=[f"{y}_h{h}", shock, unit_col, "year"])
        if keep[unit_col].nunique() < 2:
            raise ValueError(f"Insufficient units at h={h}")

        unit_dums = pd.get_dummies(keep[unit_col], drop_first=True, dtype=float)
        year_dums = pd.get_dummies(keep["year"].astype(int), drop_first=True, dtype=float)
        year_dums.columns = [f"y_{c}" for c in year_dums.columns]
        X = sm.add_constant(pd.concat([keep[[shock]].astype(float),
                                         unit_dums, year_dums], axis=1))
        cluster = keep[unit_col].astype("category").cat.codes.to_numpy()
        res = sm.OLS(keep[f"{y}_h{h}"].astype(float).to_numpy(), X.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": cluster}
        )
        b = float(res.params[1])
        s = float(res.bse[1])
        rows.append({"h": int(h), "beta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    return pd.DataFrame(rows)
