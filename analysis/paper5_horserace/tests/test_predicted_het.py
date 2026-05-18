"""Validation tests for predicted heterozygosity substrate.

Includes canonical-value anchors against Ashraf-Galor (2013) AER
country.dta values, with tolerance ±0.001. These anchor tests would
fail under any first-principles reconstruction (linear-from-Ramachandran
or similar) that does not load the actual replication file.
"""
from pathlib import Path

import pandas as pd

PARQ = Path("analysis/data/deep_determinants/predicted_het_pw_adjusted.parquet")


def test_parquet_exists():
    assert PARQ.exists(), "predicted_het_pw_adjusted.parquet missing"


def test_columns():
    df = pd.read_parquet(PARQ)
    expected = {"iso3", "H_pred", "H_pred_pwadj",
                "mdist_addis", "mdist_addis_pwadj", "source"}
    assert expected.issubset(set(df.columns)), f"missing {expected - set(df.columns)}"


def test_coverage():
    df = pd.read_parquet(PARQ)
    assert len(df) >= 140, f"only {len(df)} countries; need >= 140"
    assert df["iso3"].str.match(r"^[A-Z]{3}$").all()


def test_value_ranges():
    df = pd.read_parquet(PARQ)
    assert df["H_pred"].between(0.55, 0.78).all()
    assert df["H_pred_pwadj"].between(0.55, 0.78).all()
    assert df["mdist_addis"].between(0, 30000).all()


def test_pwadj_differs_from_unadjusted():
    """Ancestry adjustment must move at least 20 countries by >0.005."""
    df = pd.read_parquet(PARQ)
    moved = (df["H_pred_pwadj"] - df["H_pred"]).abs() > 0.005
    assert moved.sum() >= 20, f"only {moved.sum()} countries adjusted"


def test_source_label_honest():
    """Source must indicate the AG 2013 replication archive, not a reconstruction."""
    df = pd.read_parquet(PARQ)
    src = df["source"].iloc[0].lower()
    assert "replication" in src or "country.dta" in src
    assert "first-principles" not in src
    assert "ramachandran" not in src


def test_canonical_anchor_eth():
    """Ethiopia: AG 2013 pdiv = 0.77430, pdiv_aa = 0.77430 (origin point)."""
    df = pd.read_parquet(PARQ).set_index("iso3")
    assert abs(df.loc["ETH", "H_pred"] - 0.77430) < 0.001
    assert abs(df.loc["ETH", "H_pred_pwadj"] - 0.77430) < 0.001
    assert df.loc["ETH", "mdist_addis"] < 1.0


def test_canonical_anchor_jpn():
    """Japan: AG 2013 pdiv = 0.68346, pdiv_aa = 0.68346."""
    df = pd.read_parquet(PARQ).set_index("iso3")
    assert abs(df.loc["JPN", "H_pred"] - 0.68346) < 0.001
    assert abs(df.loc["JPN", "H_pred_pwadj"] - 0.68346) < 0.001


def test_canonical_anchor_usa_ancestry_shift():
    """USA: pdiv = 0.63166 raw but pdiv_aa = 0.72034 after ancestry adjustment
    (the canonical demonstration that the AG ancestry adjustment matters)."""
    df = pd.read_parquet(PARQ).set_index("iso3")
    assert abs(df.loc["USA", "H_pred"] - 0.63166) < 0.001
    assert abs(df.loc["USA", "H_pred_pwadj"] - 0.72034) < 0.001
    assert df.loc["USA", "H_pred_pwadj"] - df.loc["USA", "H_pred"] > 0.05


def test_canonical_anchor_bol_per():
    """Bolivia and Peru: native-American populations move noticeably with adjustment."""
    df = pd.read_parquet(PARQ).set_index("iso3")
    assert abs(df.loc["BOL", "H_pred"] - 0.58997) < 0.001
    assert abs(df.loc["BOL", "H_pred_pwadj"] - 0.62789) < 0.001
    assert abs(df.loc["PER", "H_pred"] - 0.59680) < 0.001
    assert abs(df.loc["PER", "H_pred_pwadj"] - 0.64336) < 0.001
