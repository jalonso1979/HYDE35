from pathlib import Path
import pandas as pd

PARQ = Path("analysis/data/deep_determinants/pandemic_intensity_pre1500.parquet")


def test_parquet_exists():
    assert PARQ.exists()


def test_columns():
    df = pd.read_parquet(PARQ)
    expected = {"iso3", "pandemic_intensity", "pandemic_intensity_norm", "n_pandemic_years", "source"}
    assert expected.issubset(set(df.columns))


def test_coverage():
    df = pd.read_parquet(PARQ)
    assert len(df) >= 180
    assert df["iso3"].str.match(r"^[A-Z]{3}$").all()


def test_value_ranges():
    df = pd.read_parquet(PARQ)
    assert (df["pandemic_intensity"] >= 0).all()
    assert df["pandemic_intensity_norm"].between(0, 1).all()
    assert (df["n_pandemic_years"] >= 0).all()


def test_european_iso_highest():
    """ITA, GRC, EGY, TUR should be in top decile of pandemic exposure
    (Justinianic plague's Mediterranean focus)."""
    df = pd.read_parquet(PARQ).sort_values("pandemic_intensity", ascending=False)
    top_decile_count = max(1, len(df) // 10)
    top_decile = df.head(top_decile_count)["iso3"].tolist()
    for iso in ["ITA", "GRC", "EGY", "TUR"]:
        assert iso in top_decile, f"{iso} not in top decile of pandemic exposure"


def test_americas_zero_or_low():
    """Pre-Columbian Americas had no Old-World pandemic exposure pre-1500."""
    df = pd.read_parquet(PARQ)
    for iso in ["MEX", "PER", "BRA", "USA"]:
        row = df[df["iso3"] == iso]
        if len(row) == 0:
            continue
        assert row["pandemic_intensity"].iloc[0] < 0.05
