"""2SLS IV estimator with weak-IV diagnostics (first-stage F + Anderson-Rubin)."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_iv_2sls(
    df: pd.DataFrame,
    y: str,
    x: str,
    instruments: Sequence[str],
    controls: Sequence[str] | None = None,
    unit_col: str = "iso3",
) -> dict:
    instruments = list(instruments)
    controls = list(controls or [])
    sub = df.dropna(subset=[y, x] + instruments + controls + [unit_col]).copy()
    unit_dums = pd.get_dummies(sub[unit_col], drop_first=True, dtype=float)
    # First stage: x ~ z + W + FE
    W_first = sm.add_constant(pd.concat([
        sub[instruments].astype(float),
        sub[controls].astype(float) if controls else pd.DataFrame(index=sub.index),
        unit_dums,
    ], axis=1))
    fs = sm.OLS(sub[x].astype(float).to_numpy(), W_first.to_numpy()).fit()
    x_hat = fs.fittedvalues
    # First-stage F on instrument coefficients
    f_test = fs.f_test(np.eye(len(W_first.columns))[1:1 + len(instruments)])
    first_stage_f = float(f_test.fvalue)

    # Second stage: y ~ x_hat + W + FE
    sub2 = sub.copy()
    sub2["__x_hat"] = x_hat
    W_second = sm.add_constant(pd.concat([
        sub2[["__x_hat"]].astype(float),
        sub2[controls].astype(float) if controls else pd.DataFrame(index=sub2.index),
        unit_dums.reindex(sub2.index),
    ], axis=1))
    ss = sm.OLS(sub2[y].astype(float).to_numpy(), W_second.to_numpy()).fit(cov_type="HC1")
    beta = float(ss.params[1])
    se = float(ss.bse[1])

    # Anderson-Rubin: F-test of instrument coefficients in reduced-form y ~ z + W
    W_ar = sm.add_constant(pd.concat([
        sub[instruments].astype(float),
        sub[controls].astype(float) if controls else pd.DataFrame(index=sub.index),
        unit_dums,
    ], axis=1))
    ar_res = sm.OLS(sub[y].astype(float).to_numpy(), W_ar.to_numpy()).fit()
    ar_test = ar_res.f_test(np.eye(len(W_ar.columns))[1:1 + len(instruments)])
    ar_pvalue = float(ar_test.pvalue)

    return {
        "beta": beta, "se": se,
        "first_stage_f": first_stage_f,
        "ar_pvalue": ar_pvalue,
        "n": int(len(sub)),
    }
