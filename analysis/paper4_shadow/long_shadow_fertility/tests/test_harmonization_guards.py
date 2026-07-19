"""Phase 7 Pillar D: edge-case guards on wage-harmonization OLS fit."""
import pandas as pd
import pytest


def _make_allen(iso, years, log_wage):
    return pd.DataFrame({"iso3": iso, "year": list(years), "log_real_wage": log_wage})


def _make_mad(iso, years, log_gdppc):
    return pd.DataFrame({"iso3": iso, "year": list(years), "log_gdppc": log_gdppc})


def test_negative_slope_triggers_median_shift(monkeypatch):
    """NLD-like: positive Allen variation, negative Maddison correlation."""
    import analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 as mod
    monkeypatch.setattr(mod, "COUNTRIES", ["NLD"])
    years = range(1820, 1851)
    allen = _make_allen("NLD", years, [1.0 + 0.02 * i for i in range(len(years))])
    mad = _make_mad("NLD", years, [8.0 - 0.03 * i for i in range(len(years))])
    coefs = mod._fit_per_country(allen, mad)
    a, b, method = coefs["NLD"]
    assert method == "median_shift_negative_slope"
    assert b == 1.0


def test_flat_slope_triggers_median_shift(monkeypatch):
    """ITA-like: OLS slope collapses to ~0 (nearly flat Maddison relative to Allen)."""
    import analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 as mod
    monkeypatch.setattr(mod, "COUNTRIES", ["ITA"])
    years = range(1820, 1851)
    allen = _make_allen("ITA", years, [1.5 + 0.02 * i for i in range(len(years))])
    mad = _make_mad("ITA", years, [7.5 + 0.0005 * i for i in range(len(years))])
    coefs = mod._fit_per_country(allen, mad)
    a, b, method = coefs["ITA"]
    assert method == "median_shift_flat_slope"
    assert b == 1.0


def test_widened_overlap_used_when_narrow_thin(monkeypatch):
    """ESP-like: 3 obs in (1820,1850); 20 obs in (1800,1913). Wider window kicks in."""
    import analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 as mod
    monkeypatch.setattr(mod, "COUNTRIES", ["ESP"])
    years_wide = list(range(1800, 1823))  # 23 obs in (1800, 1913), only 3 in (1820, 1850)
    log_wage = [1.0 + 0.05 * i for i in range(len(years_wide))]
    log_gdp = [8.0 + 0.05 * i for i in range(len(years_wide))]
    allen = _make_allen("ESP", years_wide, log_wage)
    mad = _make_mad("ESP", years_wide, log_gdp)
    coefs = mod._fit_per_country(allen, mad)
    a, b, method = coefs["ESP"]
    assert method == "ols"
    assert b > 0.05


def test_no_overlap_falls_back_to_median(monkeypatch):
    """Zero overlap obs anywhere -> median-shift fallback."""
    import analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 as mod
    monkeypatch.setattr(mod, "COUNTRIES", ["XYZ"])
    allen = _make_allen("XYZ", range(1820, 1851), [1.0] * 31)
    mad = _make_mad("XYZ", range(1900, 1950), [8.0] * 50)
    coefs = mod._fit_per_country(allen, mad)
    a, b, method = coefs["XYZ"]
    assert method == "median_shift_no_overlap"
    assert b == 1.0


def test_ok_case_returns_ols(monkeypatch):
    """Clean positive slope > 0.05 returns OLS fit."""
    import analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 as mod
    monkeypatch.setattr(mod, "COUNTRIES", ["GBR"])
    years = range(1820, 1851)
    allen = _make_allen("GBR", years, [1.0 + 0.02 * i for i in range(len(years))])
    mad = _make_mad("GBR", years, [8.0 + 0.03 * i for i in range(len(years))])
    coefs = mod._fit_per_country(allen, mad)
    a, b, method = coefs["GBR"]
    assert method == "ols"
    assert b > 0.05
