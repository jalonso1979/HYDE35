import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.build_sigl_volcanic_panel import (
    build_sigl_volcanic_panel,
)


def test_four_iso3():
    df = build_sigl_volcanic_panel()
    assert set(df["iso3"].unique()) == {"GBR", "FRA", "ITA", "SWE"}


def test_columns():
    df = build_sigl_volcanic_panel()
    for c in ("iso3", "year", "vssi_tg_s", "log_vssi"):
        assert c in df.columns


def test_tambora_krakatoa_have_nonzero_vssi():
    """Tambora 1815 and Krakatoa 1883 are well within eVolv2k coverage."""
    df = build_sigl_volcanic_panel().set_index(["iso3", "year"])
    for y in (1815, 1883):
        for iso in ("GBR", "FRA", "ITA", "SWE"):
            assert df.loc[(iso, y), "vssi_tg_s"] > 0


def test_quiet_year_zero_vssi():
    df = build_sigl_volcanic_panel().set_index(["iso3", "year"])
    assert df.loc[("GBR", 1750), "vssi_tg_s"] == 0


def test_post_1900_zero_vssi():
    """eVolv2k stops at 1900; we zero-fill so 1950 should be 0."""
    df = build_sigl_volcanic_panel().set_index(["iso3", "year"])
    assert df.loc[("GBR", 1950), "vssi_tg_s"] == 0
