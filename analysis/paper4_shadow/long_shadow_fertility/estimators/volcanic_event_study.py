"""Per-event volcanic event study on a single-country annual time series."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def event_study_single_event(
    df: pd.DataFrame,
    y: str,
    eruption_year: int,
    pre: int = 5,
    post: int = 10,
    reference_h: int = -1,
) -> pd.DataFrame:
    """Estimate event-time dummies h in [-pre, +post] with h=reference_h excluded,
    plus a linear secular trend, on observations in [eruption_year - pre, eruption_year + post].

    The window is restricted to the given pre/post bracket; the dummies span the
    same range minus the reference period. A linear trend is included whenever it
    is linearly independent of the dummies and constant (i.e., whenever some
    event-time horizons are unobserved or duplicated); if the design is exactly
    saturated, the trend is dropped to maintain identification.

    Returns DataFrame with columns: h, delta, se, ci_low, ci_high.
    """
    sub = df[df["year"].between(eruption_year - pre, eruption_year + post)].copy()
    sub = sub.dropna(subset=[y]).sort_values("year").reset_index(drop=True)
    sub["h"] = sub["year"] - eruption_year
    horizons = [h for h in range(-pre, post + 1) if h != reference_h]
    for h in horizons:
        sub[f"D_h{h}"] = (sub["h"] == h).astype(int)
    sub["trend"] = sub["year"] - sub["year"].mean()

    dummy_cols = [f"D_h{h}" for h in horizons]
    X_with_trend = sm.add_constant(sub[dummy_cols + ["trend"]])
    rank_full = np.linalg.matrix_rank(X_with_trend.to_numpy())
    if rank_full < X_with_trend.shape[1]:
        # Saturated design: trend is collinear with const + dummies.
        X = sm.add_constant(sub[dummy_cols])
    else:
        X = X_with_trend

    res = sm.OLS(sub[y].to_numpy(), X.to_numpy()).fit(cov_type="HC1")
    rows = []
    for i, h in enumerate(horizons):
        idx = i + 1  # +1 for const
        b = float(res.params[idx])
        s = float(res.bse[idx])
        rows.append({"h": h, "delta": b, "se": s,
                      "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s})
    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)
