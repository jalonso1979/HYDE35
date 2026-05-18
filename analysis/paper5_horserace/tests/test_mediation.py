# analysis/paper5_horserace/tests/test_mediation.py
import numpy as np
import pandas as pd
import pytest

from analysis.paper5_horserace.mediation import mediation_share_with_ci


def test_pure_mediation_returns_one():
    """If pathway perfectly mediates, share should be close to 1."""
    rng = np.random.default_rng(42)
    n = 300
    s = rng.standard_normal(n)
    pathway = (s + rng.standard_normal(n) * 0.1) > 0  # pathway nearly a function of s
    y = pathway.astype(float) * 2.0 + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"y": y, "s": s, "p": pathway.astype(int)})
    result = mediation_share_with_ci(df, y_col="y", substrate="s",
                                      pathway_dummies=["p"], controls=[],
                                      n_boot=200, seed=42)
    assert result["mediation_share"] > 0.7


def test_no_mediation_returns_zero():
    """If pathway is independent of substrate and outcome, share should be near 0."""
    rng = np.random.default_rng(42)
    n = 300
    s = rng.standard_normal(n)
    pathway = (rng.standard_normal(n) > 0).astype(int)
    y = s * 1.5 + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"y": y, "s": s, "p": pathway})
    result = mediation_share_with_ci(df, y_col="y", substrate="s",
                                      pathway_dummies=["p"], controls=[],
                                      n_boot=200, seed=42)
    assert abs(result["mediation_share"]) < 0.15


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(42)
    n = 200
    s = rng.standard_normal(n)
    p = (s + rng.standard_normal(n)) > 0
    y = p.astype(float) + s * 0.5 + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"y": y, "s": s, "p": p.astype(int)})
    result = mediation_share_with_ci(df, y_col="y", substrate="s",
                                      pathway_dummies=["p"], controls=[],
                                      n_boot=500, seed=42)
    assert result["ci_lower"] <= result["mediation_share"] <= result["ci_upper"]
