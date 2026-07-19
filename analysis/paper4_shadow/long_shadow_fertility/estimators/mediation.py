"""Two-stage mediation: x -> m -> y plus direct x -> y, with country FE.

Direct effect beta: coefficient on x in y ~ x + m + FE.
Indirect effect: phi * delta where phi from m ~ x + FE, delta from y ~ x + m + FE.
Bootstrap SEs by row resampling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit_step(df: pd.DataFrame, y: str, regressors: list[str], unit_col: str):
    dums = pd.get_dummies(df[unit_col], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[regressors].astype(float), dums], axis=1))
    res = sm.OLS(df[y].astype(float).to_numpy(), X.to_numpy()).fit()
    coefs = {r: float(res.params[1 + i]) for i, r in enumerate(regressors)}
    return coefs


def fit_mediation(
    df: pd.DataFrame,
    y: str,
    x: str,
    m: str,
    unit_col: str = "iso3",
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    sub = df.dropna(subset=[y, x, m, unit_col])
    n = len(sub)
    step1 = _fit_step(sub, y=m, regressors=[x], unit_col=unit_col)
    phi = step1[x]
    step2 = _fit_step(sub, y=y, regressors=[x, m], unit_col=unit_col)
    direct = step2[x]; delta = step2[m]
    indirect = phi * delta
    total = direct + indirect

    rng = np.random.default_rng(seed)
    boot_direct, boot_indirect = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = sub.iloc[idx]
        try:
            s1 = _fit_step(boot, y=m, regressors=[x], unit_col=unit_col)
            s2 = _fit_step(boot, y=y, regressors=[x, m], unit_col=unit_col)
            boot_direct.append(s2[x])
            boot_indirect.append(s1[x] * s2[m])
        except Exception:
            continue
    return {
        "direct": float(direct), "indirect": float(indirect), "total": float(total),
        "direct_se": float(np.std(boot_direct)) if boot_direct else float("nan"),
        "indirect_se": float(np.std(boot_indirect)) if boot_indirect else float("nan"),
        "phi": float(phi), "delta": float(delta),
        "n": int(n),
    }
