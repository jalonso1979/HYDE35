"""Assemble unified multi-country panel for Phase 2 pooled estimation.

Concatenates 7 country-year fertility series (Phase 1 England + Phase 2
France/Italy/Sweden + Phase 5 Belgium/Netherlands/Spain) and merges with
climate, Maddison GDPpc, and controls.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility")
ENG = ROOT / "england_fertility_annual_1541_2020.parquet"
FRA = ROOT / "france_fertility_annual.parquet"
ITA = ROOT / "italy_fertility_annual.parquet"
SWE = ROOT / "sweden_fertility_annual.parquet"
BEL = ROOT / "bel_fertility_annual.parquet"
NLD = ROOT / "nld_fertility_annual.parquet"
ESP = ROOT / "esp_fertility_annual.parquet"
NOR = ROOT / "nor_fertility_annual.parquet"
DNK = ROOT / "dnk_fertility_annual.parquet"
FIN = ROOT / "fin_fertility_annual.parquet"
ISL = ROOT / "isl_fertility_annual.parquet"
CHE = ROOT / "che_fertility_annual.parquet"
CLIM = ROOT / "country_climate_annual.parquet"
GDP = ROOT / "maddison_multicountry_annual.parquet"
CTRL = ROOT / "controls_panel_country_year.parquet"
OUT = ROOT / "panel_multi_country_year.parquet"

ERUPTIONS = [1600, 1641, 1815, 1883, 1991]


def _normalize_england() -> pd.DataFrame:
    eng = pd.read_parquet(ENG)
    eng["iso3"] = "GBR"
    return eng[["iso3", "year", "log_cbr", "births", "population", "source"]]


def _normalize_other(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["iso3", "year", "log_cbr", "births", "population", "source"]]


def assemble_panel_multi(write: bool = False) -> pd.DataFrame:
    fert = pd.concat([
        _normalize_england(),
        _normalize_other(FRA),
        _normalize_other(ITA),
        _normalize_other(SWE),
        _normalize_other(BEL),
        _normalize_other(NLD),
        _normalize_other(ESP),
        _normalize_other(NOR),
        _normalize_other(DNK),
        _normalize_other(FIN),
        _normalize_other(ISL),
        _normalize_other(CHE),
    ], ignore_index=True)

    clim = pd.read_parquet(CLIM)
    gdp = pd.read_parquet(GDP)[["iso3", "year", "gdppc", "log_gdppc"]]
    ctrl = pd.read_parquet(CTRL)

    df = (fert
          .merge(clim, on=["iso3", "year"], how="outer", suffixes=("", "_clim"))
          .merge(gdp, on=["iso3", "year"], how="left")
          .merge(ctrl, on=["iso3", "year"], how="left", suffixes=("", "_ctrl")))

    df["is_eruption_year"] = df["year"].isin(ERUPTIONS).astype(int)
    df["years_to_nearest_eruption"] = df["year"].apply(
        lambda y: min(abs(y - e) for e in ERUPTIONS) if pd.notna(y) else None
    )

    # Drop suffixed duplicates from the controls merge (vol_t_10y/vol_p_10y
    # appear in both country_climate_annual and controls_panel; keep the
    # climate-panel originals).
    drop_cols = [c for c in df.columns if c.endswith("_clim") or c.endswith("_ctrl")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = assemble_panel_multi(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['iso3'].nunique()} iso3")
