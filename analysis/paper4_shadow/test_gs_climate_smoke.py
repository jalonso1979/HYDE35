"""Smoke tests for build_gs_climate.py country-level outputs.

These are the pre-registered edge-case properties the builder must satisfy.
Run after the builder produces output parquets. Exit 0 on success.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

DATA = Path("/Volumes/BIGDATA/HYDE35/analysis/data")
WEIGHTINGS = ["area", "pop", "cropw"]


def assert_cross_section() -> None:
    df = pd.read_parquet(DATA / "country_seasonality_gs_preindustrial.parquet")
    expected_cols = ["iso3", "n_gs_months_cropw", "gs_months_mask_cropw"]
    for w in WEIGHTINGS:
        expected_cols += [f"sigma_v_T_gs_pre1750_{w}",
                          f"sigma_v_P_gs_pre1750_{w}",
                          f"T_gs_mean_pre1750_{w}",
                          f"P_gs_mean_pre1750_{w}",
                          f"sigma_v_T_nongs_pre1750_{w}",
                          f"n_gs_months_{w}"]
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"missing cross-section cols: {missing}"
    assert df["iso3"].is_unique, "iso3 must be unique in cross-section"
    assert len(df) >= 180, f"too few countries: {len(df)}"
    egy = df[df["iso3"] == "EGY"].iloc[0]
    assert egy["n_gs_months_cropw"] <= 6, (
        f"Egypt cropw n_gs_months unexpectedly high: {egy['n_gs_months_cropw']}")
    idn = df[df["iso3"] == "IDN"].iloc[0]
    assert idn["n_gs_months_cropw"] == 12, (
        f"Indonesia cropw n_gs_months not 12: {idn['n_gs_months_cropw']}")
    fra = df[df["iso3"] == "FRA"].iloc[0]
    assert 5 <= fra["n_gs_months_cropw"] <= 9, (
        f"France cropw n_gs_months out of range: {fra['n_gs_months_cropw']}")
    for w in WEIGHTINGS:
        for col in (f"sigma_v_T_gs_pre1750_{w}", f"sigma_v_P_gs_pre1750_{w}"):
            valid = df.dropna(subset=[col])
            assert (valid[col] > 0).all(), f"{col} must be positive when defined"
    print(f"  cross-section: {len(df)} countries, "
          f"{df['n_gs_months_cropw'].notna().sum()} with cropw n_gs defined")


def assert_annual_panel() -> None:
    df = pd.read_parquet(DATA / "country_climate_gs_1421_2025.parquet")
    expected_cols = ["iso3", "year"]
    for w in WEIGHTINGS:
        expected_cols += [f"t_gs_mean_{w}", f"p_gs_mean_{w}",
                          f"t_gs_anom_{w}", f"p_gs_anom_{w}"]
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"missing panel cols: {missing}"
    assert df.groupby(["iso3", "year"]).size().max() == 1, "duplicate iso3-year"
    yrs = sorted(df["year"].unique())
    assert yrs[0] == 1421 and yrs[-1] >= 2008, (
        f"unexpected year range: {yrs[0]}-{yrs[-1]}")
    assert list(yrs) == list(range(yrs[0], yrs[-1] + 1)), (
        f"year range not contiguous: gap detected in 1421-{yrs[-1]}")
    # Panel must carry n_gs_months_* so downstream filters are self-contained.
    for w in WEIGHTINGS:
        assert f"n_gs_months_{w}" in df.columns, (
            f"panel missing n_gs_months_{w} (needed for downstream filters)")
    pre = df[df["year"].between(1421, 1750)]
    anom_mean = pre.groupby("iso3")["t_gs_anom_cropw"].mean().abs()
    # Tolerance 1e-5 (not 1e-6): source data is float32, so roundtrip gives ~2e-6 residuals
    assert (anom_mean.dropna() < 1e-5).all(), (
        f"t_gs_anom_cropw not centered on 1421-1750: max |mean|={anom_mean.max()}")
    print(f"  annual panel: {len(df)} rows, {df['iso3'].nunique()} countries, "
          f"years {yrs[0]}-{yrs[-1]}")


def main() -> None:
    print("Running build_gs_climate smoke tests...")
    assert_cross_section()
    assert_annual_panel()
    print("All smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
