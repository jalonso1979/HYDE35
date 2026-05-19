from pathlib import Path
import pandas as pd

PARQ = Path("analysis/data/deep_determinants_horserace.parquet")


def test_parquet_exists():
    assert PARQ.exists()


def test_substrate_columns():
    df = pd.read_parquet(PARQ)
    substrates = {"sigma_v_T_pre1750", "H_pred_pwadj", "ancestral_yield_log",
                  "pandemic_intensity_norm"}
    assert substrates.issubset(set(df.columns))


def test_climate_bundle_columns():
    df = pd.read_parquet(PARQ)
    bundle = {"t_mean_pre1750", "p_mean_pre1750",
              "sigma_v_T_pre1750", "sigma_v_P_pre1750"}
    assert bundle.issubset(set(df.columns))
    # All four climate bundle columns should have at least 185 non-null countries
    for col in bundle:
        n = df[col].notna().sum()
        assert n >= 185, f"{col} has only {n} non-null countries"


def test_density_outcomes():
    df = pd.read_parquet(PARQ)
    density = {"log_popd_1500", "log_popd_2025"}
    assert density.issubset(set(df.columns))
    for col in density:
        n = df[col].notna().sum()
        assert n >= 185, f"{col} has only {n} non-null countries"


def test_outcome_columns():
    df = pd.read_parquet(PARQ)
    outcomes = {"log_pop_growth_1950_2025", "urban_change_1950_2025",
                "log_gdppc_2015", "dt_timing_year",
                "log_popd_1500", "log_popd_2025"}
    assert outcomes.issubset(set(df.columns))


def test_control_columns():
    df = pd.read_parquet(PARQ)
    controls = {"abs_lat", "log_area", "landlocked", "ruggedness_proxy",
                "log_dist_neolithic"}
    assert controls.issubset(set(df.columns))


def test_pathway_dummies():
    df = pd.read_parquet(PARQ)
    pathway_cols = [c for c in df.columns if c.startswith("pathway_")]
    assert len(pathway_cols) >= 4, f"need ≥4 pathway dummies; got {pathway_cols}"


def test_coverage_for_full_horserace():
    """At least 140 countries must have all four substrates non-null."""
    df = pd.read_parquet(PARQ)
    full = df.dropna(subset=["sigma_v_T_pre1750", "H_pred_pwadj",
                              "ancestral_yield_log", "pandemic_intensity_norm"])
    assert len(full) >= 140, f"only {len(full)} countries with all substrates"


def test_one_row_per_iso3():
    df = pd.read_parquet(PARQ)
    assert df["iso3"].is_unique
    assert df["iso3"].str.match(r"^[A-Z]{3}$").all()
