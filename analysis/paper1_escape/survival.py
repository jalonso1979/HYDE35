"""Survival analysis for time-to-Malthusian-exit (Paper 1)."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def build_survival_dataset(
    trajectory_df: pd.DataFrame,
    event_col: str = "intensification_onset_year",
    origin_col: str = "peak_extensification_year",
    entity_col: str = "country",
    max_duration: float = 2000.0,
) -> pd.DataFrame:
    df = trajectory_df[[entity_col, origin_col, event_col]].copy()
    df["event"] = (~df[event_col].isna()).astype(int)
    df["duration"] = np.where(
        df["event"] == 1,
        df[event_col] - df[origin_col],
        max_duration,
    )
    df = df[df[origin_col].notna() & (df["duration"] > 0)]
    return df[[entity_col, "duration", "event"]].reset_index(drop=True)


def fit_cox_model(
    surv_df: pd.DataFrame,
    covariates: list[str],
    duration_col: str = "duration",
    event_col: str = "event",
) -> pd.DataFrame:
    cph = CoxPHFitter()
    fit_cols = [duration_col, event_col] + covariates
    cph.fit(surv_df[fit_cols], duration_col=duration_col, event_col=event_col)
    return cph.summary
