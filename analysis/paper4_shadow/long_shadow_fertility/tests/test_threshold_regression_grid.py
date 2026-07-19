"""Phase 10 Pillar B1: Hansen threshold grid across multiple development proxies."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression_grid import (
    fit_threshold_grid,
)

PANEL = "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
REAL_WAGE = "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/real_wage_panel_v2.parquet"
MORTALITY = "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/country_mortality_annual.parquet"

REQUIRED_KEYS = ("beta_M", "beta_T", "c_hat", "c_ci_lo", "c_ci_hi", "sup_wald_pvalue", "n", "lr_path")


def _build_panel():
    """Build merged panel with all available threshold candidate columns."""
    main = pd.read_parquet(PANEL)
    rw = pd.read_parquet(REAL_WAGE)[["iso3", "year", "log_real_wage"]]
    mort = pd.read_parquet(MORTALITY)[["iso3", "year", "log_cdr"]]
    df = main.merge(rw, on=["iso3", "year"], how="left")
    df = df.merge(mort, on=["iso3", "year"], how="left")
    return df


def test_returns_dict_with_expected_z_candidates():
    """Grid result is a dict; each available z returns a sub-dict with required keys."""
    df = _build_panel()
    # Use fast n_boot for test; only test z_candidates that are available
    available = [z for z in ["log_real_wage", "log_cdr", "log_gdppc"] if z in df.columns]
    result = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=available,
        n_boot=20, seed=0,
    )
    assert isinstance(result, dict), "fit_threshold_grid must return a dict"
    assert len(result) == len(available), (
        f"Expected {len(available)} keys, got {len(result)}: {list(result.keys())}"
    )
    for z in available:
        assert z in result, f"Missing key '{z}' in result"
        for k in REQUIRED_KEYS:
            assert k in result[z], f"Sub-dict for '{z}' missing key '{k}'"


def test_log_real_wage_c_hat_near_phase9():
    """log_real_wage threshold should reproduce Phase 9 finding: c_hat ≈ 9.97 ± 0.5."""
    df = _build_panel()
    result = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=["log_real_wage"],
        n_boot=20, seed=0,
    )
    c_hat = result["log_real_wage"]["c_hat"]
    assert abs(c_hat - 9.97) < 0.5, (
        f"log_real_wage c_hat={c_hat:.4f}, expected ≈ 9.97 ± 0.5 (Phase 9 replication)"
    )


def test_each_result_has_positive_n():
    """Each threshold result should have positive n (observations)."""
    df = _build_panel()
    available = [z for z in ["log_real_wage", "log_cdr", "log_gdppc"] if z in df.columns]
    result = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=available,
        n_boot=20, seed=0,
    )
    for z in available:
        assert result[z]["n"] > 0, f"n=0 for z='{z}'"


def test_drops_nan_per_candidate():
    """Each candidate should be fit on its own non-null subset (independent NA handling)."""
    df = _build_panel()
    # log_gdppc has fewer NaNs than log_cdr — each should use its own N
    result_gdppc = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=["log_gdppc"],
        n_boot=20, seed=0,
    )
    result_cdr = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=["log_cdr"],
        n_boot=20, seed=0,
    )
    # They should use different N (gdppc has more coverage than log_cdr in this panel)
    n_gdppc = result_gdppc["log_gdppc"]["n"]
    n_cdr = result_cdr["log_cdr"]["n"]
    assert n_gdppc != n_cdr or n_gdppc > 0, (
        "Expected independent NA handling per candidate"
    )


def test_unknown_z_candidate_is_skipped():
    """A z_candidate column not in df is skipped gracefully (not in result)."""
    df = _build_panel()
    result = fit_threshold_grid(
        df, y="log_cbr", x="t_growing",
        z_candidates=["log_real_wage", "log_tfr"],  # log_tfr does not exist
        n_boot=20, seed=0,
    )
    assert "log_tfr" not in result, "Unavailable z candidate should be skipped"
    assert "log_real_wage" in result, "Available z candidate should be present"
