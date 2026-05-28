"""Tests for the SPEI joint climate-stress index (Phase 10 Pillar #5).

Tests verify:
1. SPEI is returned as a zero-mean unit-variance series per country.
2. SPEI detects known historical drought events.
3. Thornthwaite PET helpers produce physically sensible values.
4. Daylight-hour formula gives correct sign (tropical < polar in summer).
5. Absolute reconstruction is internally consistent.
"""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestDaylightHours:
    def test_equator_close_to_12(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _mean_monthly_daylight_hours,
        )
        N = _mean_monthly_daylight_hours(0.0)
        assert N.shape == (12,)
        # Equator: all months close to 12 h
        assert np.all(np.abs(N - 12.0) < 1.0)

    def test_summer_longer_in_north(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _mean_monthly_daylight_hours,
        )
        N_60 = _mean_monthly_daylight_hours(60.0)
        N_30 = _mean_monthly_daylight_hours(30.0)
        # July (index 6) should have more daylight at higher latitude
        assert N_60[6] > N_30[6]

    def test_symmetric_summer_winter(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _mean_monthly_daylight_hours,
        )
        N = _mean_monthly_daylight_hours(50.0)
        # June (idx 5) > December (idx 11)
        assert N[5] > N[11]

    def test_reasonable_range(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _mean_monthly_daylight_hours,
        )
        N = _mean_monthly_daylight_hours(55.0)
        # Daylight hours between 6 and 20 for mid-latitude
        assert np.all(N >= 6.0)
        assert np.all(N <= 20.0)


class TestThornthwaitePET:
    def test_zero_when_cold(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _thornthwaite_monthly_pet,
            _mean_monthly_daylight_hours,
        )
        T_clim = np.array([5.0] * 12)
        # All negative temperatures -> zero PET
        T_abs = np.full((5, 12), -5.0)
        N = _mean_monthly_daylight_hours(50.0)
        pet = _thornthwaite_monthly_pet(T_clim, T_abs, N)
        assert np.all(pet == 0.0)

    def test_positive_when_warm(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _thornthwaite_monthly_pet,
            _mean_monthly_daylight_hours,
        )
        T_clim = np.array([10.0] * 12)
        T_abs = np.full((5, 12), 15.0)
        N = _mean_monthly_daylight_hours(50.0)
        pet = _thornthwaite_monthly_pet(T_clim, T_abs, N)
        assert np.all(pet > 0.0)

    def test_higher_temp_higher_pet(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _thornthwaite_monthly_pet,
            _mean_monthly_daylight_hours,
        )
        T_clim = np.array([10.0] * 12)
        T_warm = np.full((1, 12), 20.0)
        T_cool = np.full((1, 12), 10.0)
        N = _mean_monthly_daylight_hours(50.0)
        pet_warm = _thornthwaite_monthly_pet(T_clim, T_warm, N)
        pet_cool = _thornthwaite_monthly_pet(T_clim, T_cool, N)
        assert np.all(pet_warm > pet_cool)

    def test_shape_preserved(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import (
            _thornthwaite_monthly_pet,
            _mean_monthly_daylight_hours,
        )
        T_clim = np.array([8.0] * 12)
        T_abs = np.random.randn(100, 12) + 10.0
        N = _mean_monthly_daylight_hours(46.0)
        pet = _thornthwaite_monthly_pet(T_clim, T_abs, N)
        assert pet.shape == (100, 12)


# ---------------------------------------------------------------------------
# Integration tests on the full compute_spei function
# ---------------------------------------------------------------------------

class TestComputeSPEI:
    @pytest.fixture(scope="class")
    def spei(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import compute_spei
        return compute_spei()

    def test_columns(self, spei):
        for col in ["iso3", "year", "spei_growing", "spei_winter", "spei_annual"]:
            assert col in spei.columns, f"Missing column: {col}"

    def test_all_countries_present(self, spei):
        expected = {
            "GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP",
            "NOR", "DNK", "FIN", "ISL", "CHE",
        }
        assert set(spei["iso3"].unique()) == expected

    def test_year_range(self, spei):
        assert spei["year"].min() == 1421
        assert spei["year"].max() == 2008

    def test_zero_mean_per_country(self, spei):
        """Each country's SPEI should have near-zero mean (standardised)."""
        for iso3, grp in spei.groupby("iso3"):
            mean_gs = grp["spei_growing"].mean()
            assert abs(mean_gs) < 0.01, f"{iso3} mean={mean_gs:.4f}"

    def test_unit_variance_per_country(self, spei):
        """Each country's SPEI should have unit standard deviation."""
        for iso3, grp in spei.groupby("iso3"):
            std_gs = grp["spei_growing"].std(ddof=1)
            assert abs(std_gs - 1.0) < 0.01, f"{iso3} std={std_gs:.4f}"

    def test_2003_drought_gbr(self, spei):
        """2003 European drought should be strongly negative for GBR."""
        val = spei.loc[(spei["iso3"] == "GBR") & (spei["year"] == 2003),
                       "spei_growing"].values
        assert len(val) == 1
        assert val[0] < -1.0, f"2003 GBR spei_growing={val[0]:.3f}, expected < -1.0"

    def test_2003_drought_fra(self, spei):
        """2003 European drought should be strongly negative for FRA."""
        val = spei.loc[(spei["iso3"] == "FRA") & (spei["year"] == 2003),
                       "spei_growing"].values
        assert len(val) == 1
        assert val[0] < -1.0, f"2003 FRA spei_growing={val[0]:.3f}, expected < -1.0"

    def test_no_nan_in_growing(self, spei):
        """spei_growing should not have NaN except possibly last row per country."""
        # Allow at most 1 NaN per country (last winter row may be NaN)
        for iso3, grp in spei.groupby("iso3"):
            nan_count = grp["spei_growing"].isna().sum()
            assert nan_count == 0, f"{iso3}: {nan_count} NaN in spei_growing"

    def test_reasonable_extremes(self, spei):
        """SPEI values should be within ±6 (extreme but plausible for long series)."""
        assert spei["spei_growing"].abs().max() < 6.0
        assert spei["spei_annual"].abs().max() < 6.0

    def test_single_country_subset(self):
        from analysis.paper4_shadow.long_shadow_fertility.data.spei_index import compute_spei
        df = compute_spei(countries=["FRA"])
        assert set(df["iso3"].unique()) == {"FRA"}
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Reconstruction sanity check
# ---------------------------------------------------------------------------

class TestAbsoluteReconstruction:
    def test_mean_absolute_t_plausible(self):
        """Reconstructed absolute T should be in a physically plausible range."""
        import pandas as pd
        clim = pd.read_parquet(
            "/Volumes/BIGDATA/HYDE35/analysis/data/cru_country_climatology_1901_1950.parquet"
        )
        mod = pd.read_parquet(
            "/Volumes/BIGDATA/HYDE35/analysis/data/modera_country_monthly_cropw.parquet"
        )
        merged = mod[mod["iso3"] == "GBR"].merge(
            clim[clim["iso3"] == "GBR"][["month", "tmp_c_clim"]],
            on="month",
        )
        T_abs = merged["tmp_c_clim"] + merged["t_anom_c"]
        # England annual mean should be between 5°C and 15°C on average
        assert 5.0 < T_abs.mean() < 15.0

    def test_p_abs_nonnegative(self):
        """Clipped precipitation should be non-negative."""
        import pandas as pd
        clim = pd.read_parquet(
            "/Volumes/BIGDATA/HYDE35/analysis/data/cru_country_climatology_1901_1950.parquet"
        )
        mod = pd.read_parquet(
            "/Volumes/BIGDATA/HYDE35/analysis/data/modera_country_monthly_cropw.parquet"
        )
        merged = mod[mod["iso3"].isin(["GBR", "FRA"])].merge(
            clim[clim["iso3"].isin(["GBR", "FRA"])][
                ["iso3", "month", "pre_mm_clim"]
            ],
            on=["iso3", "month"],
        )
        P_abs = (merged["pre_mm_clim"] + merged["p_anom_mm"]).clip(lower=0.0)
        assert P_abs.min() >= 0.0
