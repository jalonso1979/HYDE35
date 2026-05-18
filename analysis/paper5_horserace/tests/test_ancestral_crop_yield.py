"""Validation tests for Galor-Özak ancestral crop-yield substrate.

Source: Galor & Özak (2016 AER) "The Agricultural Origins of Time Preference".
Data: Caloric Suitability Index (CSI) — country-level pre-1500 caloric yield
from ozak/Caloric-Suitability-Index (Zenodo 14714917).

Variable used: pre15002AverageCalories0mean
  pre1500 = pre-Columbian crop set
  2       = excludes Asian crop varieties in Africa
  Average = average of single-crop caloric yields across suitable crops
  0       = constructs statistics excluding zero-yield cells
  mean    = mean across grid cells within the country
"""
from pathlib import Path
import pandas as pd

PARQ = Path("analysis/data/deep_determinants/ancestral_crop_yield.parquet")


def test_parquet_exists():
    assert PARQ.exists()


def test_columns():
    df = pd.read_parquet(PARQ)
    expected = {"iso3", "ancestral_yield", "ancestral_yield_log", "source"}
    assert expected.issubset(set(df.columns))


def test_coverage():
    df = pd.read_parquet(PARQ)
    assert len(df) >= 130, f"only {len(df)} countries"
    assert df["iso3"].str.match(r"^[A-Z]{3}$").all()


def test_value_ranges():
    df = pd.read_parquet(PARQ)
    assert (df["ancestral_yield"] >= 0).all()
    nz = df["ancestral_yield"] > 0
    assert df.loc[nz, "ancestral_yield_log"].notna().all()


def test_known_high_yield_countries():
    """France and Italy should be in the top-100 by yield (top ~40% of 252 countries).

    Note: IND (India) is NOT in the top half of the distribution in this dataset because
    many small tropical island territories (Caribbean, Pacific) rank higher due to
    high per-pixel caloric suitability for sugar-cane and rice crops. This is a known
    feature of the CSI aggregation, not a data error.

    France (rank ~51/252) and Italy (rank ~80/252) are solidly in the upper tier
    of large economies with high temperate-zone caloric suitability.
    """
    df = pd.read_parquet(PARQ).sort_values("ancestral_yield", ascending=False)
    top_100 = df.head(100)["iso3"].tolist()
    for iso in ["FRA", "ITA"]:
        assert iso in top_100, f"{iso} not in top-100 ancestral-yield countries (rank check)"


def test_source_label_honest():
    df = pd.read_parquet(PARQ)
    src = df["source"].iloc[0].lower()
    assert "galor" in src or "özak" in src or "ozak" in src
    assert "first-principles" not in src
    assert "reconstruction" not in src


def test_canonical_anchors():
    """Anchor countries within expected ordinal ranking from GÖ 2016.

    Confirmed rankings from the canonical CSI dataset (Zenodo 14714917):
    - FRA (4309 kcal/ha): high temperate yield, rank ~51 of 252
    - ITA (3839 kcal/ha): high temperate yield, rank ~80 of 252
    - IND (2546 kcal/ha): moderate yield, rank ~161 of 252
    - SAU  (558 kcal/ha): arid, rank ~232 of 252
    - ISL   (0 kcal/ha): cold/tundra, rank ~246 of 252

    Ordinal checks:
    - FRA > SAU (high temperate Europe vs. arid Arabian peninsula)
    - IND > ISL (India's agricultural regions vs. subarctic Iceland)
    - ITA > IND (Mediterranean Italy vs. semi-arid parts of India)
    """
    df = pd.read_parquet(PARQ).set_index("iso3")
    assert df.loc["FRA", "ancestral_yield"] > df.loc["SAU", "ancestral_yield"]
    assert df.loc["IND", "ancestral_yield"] > df.loc["ISL", "ancestral_yield"]
    assert df.loc["ITA", "ancestral_yield"] > df.loc["IND", "ancestral_yield"]
