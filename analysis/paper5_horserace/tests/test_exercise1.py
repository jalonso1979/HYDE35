"""Tests for Exercise 1: Shapley decomposition + Table 3 OLS output.

Run:  python -m pytest analysis/paper5_horserace/tests/test_exercise1.py -v
"""
from pathlib import Path

import pandas as pd

PARQ = Path("analysis/data/deep_determinants/exercise1_shapley_results.parquet")
TEX = Path("analysis/figures/paper5_horserace/tab03_full_ols.tex")


def test_parquet_exists():
    assert PARQ.exists(), f"Output parquet not found: {PARQ}"


def test_columns_and_shape():
    df = pd.read_parquet(PARQ)
    expected = {"outcome", "substrate", "shapley_r2", "n_obs",
                "baseline_r2", "full_model_r2"}
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )
    # 6 outcomes × 5 substrates = 30 rows
    assert len(df) == 30, f"Expected 30 rows, got {len(df)}"


def test_shapley_values_nonnegative():
    df = pd.read_parquet(PARQ)
    # In principle Shapley can be slightly negative when adding an
    # uninformative substrate reduces R² for some orderings; require
    # all values > -0.02 as a sanity bound.
    violations = df[df["shapley_r2"] <= -0.02][["outcome", "substrate", "shapley_r2"]]
    assert violations.empty, (
        f"Shapley values below -0.02 found:\n{violations}"
    )


def test_shapley_sum_matches_marginal_r2():
    df = pd.read_parquet(PARQ)
    for outcome, sub in df.groupby("outcome"):
        marginal = sub["full_model_r2"].iloc[0] - sub["baseline_r2"].iloc[0]
        s_sum = sub["shapley_r2"].sum()
        assert abs(marginal - s_sum) < 1e-4, (
            f"Outcome {outcome}: Shapley sum {s_sum:.6f} != marginal R² {marginal:.6f}"
        )


def test_tex_table_exists():
    assert TEX.exists(), f"LaTeX table not found: {TEX}"
