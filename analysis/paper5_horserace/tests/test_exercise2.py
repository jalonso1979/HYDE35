# analysis/paper5_horserace/tests/test_exercise2.py
from pathlib import Path
import pandas as pd

PARQ = Path("analysis/data/deep_determinants/exercise2_mediation_results.parquet")


def test_parquet_exists():
    assert PARQ.exists()


def test_grid_shape():
    df = pd.read_parquet(PARQ)
    assert len(df) == 16  # 4 outcomes × 4 substrates


def test_columns():
    df = pd.read_parquet(PARQ)
    expected = {"outcome", "substrate", "mediation_share",
                "ci_lower", "ci_upper", "n_obs"}
    assert expected.issubset(set(df.columns))


def test_cis_ordered():
    df = pd.read_parquet(PARQ)
    assert (df["ci_lower"] <= df["ci_upper"]).all()
