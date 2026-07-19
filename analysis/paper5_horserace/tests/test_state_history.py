"""Validation tests for the Putterman-Weil state-history substrate.

Canonical anchor values (from the raw statehist.xls, statehistn05v3 column,
Putterman & Weil 2010 QJE replication, v3 with 0.5 discount rate):

  CHN raw  = 0.9384  (China: longest continuous state history)
  JPN raw  = 0.8841
  EGY raw  = 0.6946
  ISL raw  = 0.4503
  USA raw  = 0.2097  (pre-1500 minimal state organisation)
  AUS raw  = 0.1473  (pre-1500 hunter-gatherer)

Ancestry-adjusted values (computed via World Migration Matrix v1.1):
  USA pwadj ~ 0.647  (European settler ancestry raises score substantially)
  AUS pwadj ~ 0.730  (British settler ancestry)
  CHN pwadj ~ 0.938  (overwhelmingly own ancestry)
"""
from pathlib import Path

import pandas as pd
import pytest

PARQ = Path("analysis/data/deep_determinants/state_history_pw.parquet")


# ---------------------------------------------------------------------------
# Basic structural tests (from plan Task 2 Step 1)
# ---------------------------------------------------------------------------

def test_parquet_exists():
    assert PARQ.exists(), f"state_history_pw.parquet missing at {PARQ}"


def test_columns():
    df = pd.read_parquet(PARQ)
    expected = {"iso3", "state_hist", "state_hist_pwadj", "source"}
    missing = expected - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_coverage():
    df = pd.read_parquet(PARQ)
    assert len(df) >= 140, f"Only {len(df)} countries; need >= 140"
    assert df["iso3"].str.match(r"^[A-Z]{3}$").all(), "iso3 contains non-uppercase-3-letter codes"


def test_value_ranges():
    df = pd.read_parquet(PARQ)
    assert df["state_hist"].between(0, 1).all(), "state_hist has values outside [0,1]"
    pwadj_valid = df["state_hist_pwadj"].dropna()
    assert len(pwadj_valid) >= 100, f"Only {len(pwadj_valid)} non-null pwadj values"
    assert pwadj_valid.between(0, 1).all(), "state_hist_pwadj has values outside [0,1]"


# ---------------------------------------------------------------------------
# Canonical anchor tests — anchored against known PW 2010 values
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(PARQ).set_index("iso3")


def test_anchor_china_high_state_hist(df):
    """China should have the highest (or near-highest) raw state-history score."""
    chn = df.loc["CHN", "state_hist"]
    assert chn > 0.90, f"CHN raw state_hist={chn:.4f}; expected > 0.90 (PW value ~0.938)"


def test_anchor_usa_raw_low_pwadj_high(df):
    """USA pre-1500: low raw score but high ancestry-adjusted score.

    USA raw ~ 0.21 (minimal pre-1500 state organisation in North America).
    USA pwadj ~ 0.65 (most ancestry traces to European source populations
    with much longer state histories: GBR 18.4%, DEU 17.5%, IRL 8.9%, etc.).
    """
    usa_raw = df.loc["USA", "state_hist"]
    usa_adj = df.loc["USA", "state_hist_pwadj"]
    assert usa_raw < 0.30, f"USA raw={usa_raw:.4f}; expected < 0.30 (PW value ~0.210)"
    assert usa_adj > 0.55, f"USA pwadj={usa_adj:.4f}; expected > 0.55 (computed ~0.647)"
    assert usa_adj > usa_raw + 0.30, (
        f"USA pwadj={usa_adj:.4f} should exceed raw={usa_raw:.4f} by > 0.30"
    )


def test_anchor_australia_settler_country(df):
    """Australia (pre-1500 hunter-gatherer) should show a large ancestry uplift."""
    aus_raw = df.loc["AUS", "state_hist"]
    aus_adj = df.loc["AUS", "state_hist_pwadj"]
    assert aus_raw < 0.25, f"AUS raw={aus_raw:.4f}; expected < 0.25 (PW value ~0.147)"
    assert aus_adj > 0.60, f"AUS pwadj={aus_adj:.4f}; expected > 0.60 (computed ~0.730)"


def test_anchor_china_minimal_ancestry_adjustment(df):
    """China's ancestry-adjusted score should barely differ from its raw score.

    China's population is overwhelmingly indigenous; the WMM row for CHN
    is dominated by 'chn' itself, so pwadj ≈ raw.
    """
    chn_raw = df.loc["CHN", "state_hist"]
    chn_adj = df.loc["CHN", "state_hist_pwadj"]
    assert abs(chn_adj - chn_raw) < 0.05, (
        f"CHN raw={chn_raw:.4f}, pwadj={chn_adj:.4f}; "
        f"difference {abs(chn_adj - chn_raw):.4f} should be < 0.05"
    )
