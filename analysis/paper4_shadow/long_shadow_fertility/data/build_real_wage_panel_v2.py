"""Harmonized real-wage panel: Allen welfare-ratio rescaled to log(GDPpc) units.

For each country, fit OLS y = a + b*x on the overlap window 1820-1850 where
y = log Maddison GDPpc and x = Allen log_real_wage. Use (a, b) to transform
Allen pre-1914 values onto Maddison-comparable units.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

BIGDATA = Path("/Volumes/BIGDATA/HYDE35/analysis")
FERTILITY = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility"
)
ALLEN = BIGDATA / "data" / "allen_wage_climate_panel.parquet"
MADDISON = FERTILITY / "data" / "mpd2023_web.xlsx"
OUT = BIGDATA / "data" / "long_shadow_fertility" / "real_wage_panel_v2.parquet"

COUNTRIES = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"]
OVERLAP = (1820, 1850)


def _allen() -> pd.DataFrame:
    allen = pd.read_parquet(ALLEN)
    sub = allen.loc[allen["iso3"].isin(COUNTRIES), ["iso3", "year", "log_real_wage"]].copy()
    return sub


def _maddison() -> pd.DataFrame:
    raw = pd.read_excel(MADDISON, sheet_name="Full data")
    raw.columns = raw.columns.str.lower()
    sub = raw.loc[raw["countrycode"].isin(COUNTRIES), ["countrycode", "year", "gdppc"]].copy()
    sub = sub.rename(columns={"countrycode": "iso3"}).dropna(subset=["gdppc"])
    sub["log_gdppc"] = np.log(sub["gdppc"])
    return sub[["iso3", "year", "log_gdppc"]]


def _fit_per_country(
    allen: pd.DataFrame, mad: pd.DataFrame
) -> dict[str, tuple[float, float, str]]:
    """Fit (a, b, method) for each country with edge-case guards.

    Returns (intercept, slope, method_tag) where method_tag is one of:
      - "ols": standard OLS fit, slope >= 0.05
      - "median_shift_negative_slope": fitted slope < 0 -> b=1, a=median_diff
      - "median_shift_flat_slope": |b| < 0.05 -> b=1, a=median_diff
      - "median_shift_no_overlap": <5 obs in widened window -> b=1, a=median_diff
    """
    coefs: dict[str, tuple[float, float, str]] = {}
    for iso in COUNTRIES:
        ovp = (allen.loc[allen["iso3"] == iso]
                .merge(mad.loc[mad["iso3"] == iso], on=["iso3", "year"]))
        ovp_narrow = ovp.loc[ovp["year"].between(*OVERLAP)]
        ovp_wide = ovp.loc[ovp["year"].between(1800, 1913)]
        use_ovp = ovp_narrow if len(ovp_narrow) >= 10 else ovp_wide

        if len(use_ovp) < 5:
            allen_med = allen.loc[allen["iso3"] == iso, "log_real_wage"].median()
            mad_med = mad.loc[
                (mad["iso3"] == iso) & mad["year"].between(*OVERLAP), "log_gdppc"
            ].median()
            if pd.isna(mad_med):
                mad_med = mad.loc[mad["iso3"] == iso, "log_gdppc"].median()
            shift = float(mad_med - allen_med) if pd.notna(allen_med) else 0.0
            coefs[iso] = (shift, 1.0, "median_shift_no_overlap")
            continue

        x = use_ovp["log_real_wage"].to_numpy()
        y = use_ovp["log_gdppc"].to_numpy()
        b, a = np.polyfit(x, y, 1)
        b, a = float(b), float(a)

        if b < 0:
            allen_med = float(use_ovp["log_real_wage"].median())
            mad_med = float(use_ovp["log_gdppc"].median())
            coefs[iso] = (mad_med - allen_med, 1.0, "median_shift_negative_slope")
            continue
        if abs(b) < 0.05:
            allen_med = float(use_ovp["log_real_wage"].median())
            mad_med = float(use_ovp["log_gdppc"].median())
            coefs[iso] = (mad_med - allen_med, 1.0, "median_shift_flat_slope")
            continue

        coefs[iso] = (a, b, "ols")
    return coefs


def build_real_wage_panel_v2(write: bool = False) -> pd.DataFrame:
    allen = _allen()
    mad = _maddison()
    coefs = _fit_per_country(allen, mad)

    pre_parts = []
    for iso in COUNTRIES:
        sub = allen.loc[(allen["iso3"] == iso) & (allen["year"] <= 1913)].copy()
        if sub.empty:
            continue
        a, b, method = coefs[iso]
        sub["log_real_wage"] = a + b * sub["log_real_wage"]
        sub["source"] = "Allen_harmonized_to_Maddison"
        sub["harmonized"] = True
        sub["harmonization_method"] = method
        pre_parts.append(sub)
    pre = pd.concat(pre_parts, ignore_index=True) if pre_parts else pd.DataFrame()

    post = mad.loc[mad["year"] >= 1914].copy()
    post = post.rename(columns={"log_gdppc": "log_real_wage"})
    post["source"] = "Maddison_log_gdppc"
    post["harmonized"] = False
    post["harmonization_method"] = "maddison_native"

    df = pd.concat([pre, post], ignore_index=True)
    df = df.sort_values(["iso3", "year"]).drop_duplicates(["iso3", "year"], keep="first").reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_real_wage_panel_v2(write=True)
    print(f"wrote {OUT}: {len(df)} rows; sources: {df['source'].value_counts().to_dict()}")
