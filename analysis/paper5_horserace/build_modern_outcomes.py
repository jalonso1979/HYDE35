"""Build modern outcomes panel for the deep-determinants horserace (Paper 5).

Six country-level outcomes — four "modern" (1950-onward) and two
population-density outcomes drawn from HYDE 3.5:

1. log_pop_growth_1950_2025
   = log(P_2025 / P_1950)
   Sources:
     - P_1950: OWID population 1950-2023 (sourcing UN WPP 2022/2024)
     - P_2025: Gapminder population projections 1950-2100 (sourcing UN WPP 2024)
   Cache: analysis/data/deep_determinants/_raw/un_wpp/

2. urban_change_1950_2025
   = UrbanShare_2025 - UrbanShare_1950  (percentage points)
   Source: UN World Urbanization Prospects 2025, F02 file (Degree-of-Urbanization
   percentage by country), "Cities and Towns" sheet = total urban share.
   URL: https://population.un.org/wup/assets/Download/Countries%20and%20Aggregates/
        WUP2025-F02-Degree-of-Urbanization_percPop_by_category.xlsx
   Cache: analysis/data/deep_determinants/_raw/un_wpp/WUP2025-F02-PercUrban.xlsx

3. log_gdppc_2015
   = log(GDPpc in year 2015, in 2011 international USD)
   Source: Maddison Project Database 2023, "Full data" sheet.
   URL: https://dataverse.nl/api/access/datafile/421302
   Cache: analysis/data/deep_determinants/_raw/maddison/mpd2023_web.xlsx

4. dt_timing_year
   = first calendar year in which CBR (crude birth rate) fell below 25/1000.
   NaN if the threshold was never reached through 2023.
   Sources:
     - Pre-1950: Gapminder historical CBR 1800-2015 (sourcing Mitchell / Princeton
       European Fertility Project estimates)
     - 1950-2023: OWID crude birth rate 1950-2023 (sourcing UN WPP 2024)
   Cache: analysis/data/deep_determinants/_raw/un_wpp/

5. log_popd_1500
   = log(country population density in 1500 CE, persons/km²).
   Source: HYDE 3.5 country-level density file
   gbc2025_7apr_base/txt/popd_c.txt; column "1500".
   Mapped via hyde35_country_iso_mapping.csv (iso_num -> iso3).

6. log_popd_2025
   = log(country population density in 2025, persons/km²).
   Same source as (5), column "2025".

Output: analysis/data/deep_determinants/modern_outcomes.parquet
Columns: iso3, log_pop_growth_1950_2025, urban_change_1950_2025,
         log_gdppc_2015, dt_timing_year, log_popd_1500, log_popd_2025, source
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
RAW_WPP = ROOT / "analysis/data/deep_determinants/_raw/un_wpp"
RAW_MAD = ROOT / "analysis/data/deep_determinants/_raw/maddison"
OUT = ROOT / "analysis/data/deep_determinants/modern_outcomes.parquet"

# ---------------------------------------------------------------------------
# Remote URLs (used only if local cache is absent)
# ---------------------------------------------------------------------------
WUP_URL = (
    "https://population.un.org/wup/assets/Download/"
    "Countries%20and%20Aggregates/"
    "WUP2025-F02-Degree-of-Urbanization_percPop_by_category.xlsx"
)
OWID_POP_URL = (
    "https://ourworldindata.org/grapher/population.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)
OWID_CBR_URL = (
    "https://ourworldindata.org/grapher/crude-birth-rate.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)
GAPMINDER_POP_URL = (
    "https://raw.githubusercontent.com/open-numbers/"
    "ddf--gapminder--population/master/"
    "ddf--datapoints--population--by--country--year.csv"
)
GAPMINDER_CBR_URL = (
    "https://raw.githubusercontent.com/open-numbers/"
    "ddf--gapminder--systema_globalis/master/"
    "countries-etc-datapoints/"
    "ddf--datapoints--crude_birth_rate_births_per_1000_population--by--geo--time.csv"
)
GAPMINDER_CODES_URL = (
    "https://raw.githubusercontent.com/open-numbers/"
    "ddf--gapminder--population/master/"
    "ddf--entities--geo--country.csv"
)
MADDISON_URL = "https://dataverse.nl/api/access/datafile/421302"


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, raising on HTTP error."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest} ({dest.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_gapminder_codes() -> pd.DataFrame:
    """Return DataFrame with columns [country (lowercase), iso3]."""
    dest = RAW_WPP / "gapminder_country_codes.csv"
    if not dest.exists():
        _download(GAPMINDER_CODES_URL, dest)
    df = pd.read_csv(dest, usecols=["country", "iso3166_1_alpha3"])
    df = df.rename(columns={"iso3166_1_alpha3": "iso3"}).dropna(subset=["iso3"])
    return df


def load_population() -> pd.DataFrame:
    """Load and merge population for 1950 (OWID) and 2025 (Gapminder).

    Returns DataFrame with columns: iso3, pop1950, pop2025.
    """
    # --- P1950: OWID (WPP-based), ISO3 codes directly ---------------------
    owid_dest = RAW_WPP / "owid_pop_1950_2023.csv"
    if not owid_dest.exists():
        _download(OWID_POP_URL, owid_dest)
    owid = pd.read_csv(owid_dest)
    owid = owid[owid["Code"].str.match(r"^[A-Z]{3}$", na=False)]
    p1950 = (
        owid[owid["Year"] == 1950][["Code", "Population"]]
        .rename(columns={"Code": "iso3", "Population": "pop1950"})
        .reset_index(drop=True)
    )

    # --- P2025: Gapminder (WPP 2024 projections), lowercase codes ----------
    gm_pop_dest = RAW_WPP / "gapminder_pop_1950_2100.csv"
    if not gm_pop_dest.exists():
        _download(GAPMINDER_POP_URL, gm_pop_dest)
    gm_pop = pd.read_csv(gm_pop_dest)

    codes = _load_gapminder_codes()
    gm_pop = gm_pop.merge(codes, left_on="country", right_on="country", how="left")
    p2025 = (
        gm_pop[gm_pop["year"] == 2025][["iso3", "population"]]
        .dropna(subset=["iso3"])
        .rename(columns={"population": "pop2025"})
        .reset_index(drop=True)
    )

    # Merge
    pop = p2025.merge(p1950, on="iso3", how="inner")
    return pop


def load_urban_share() -> pd.DataFrame:
    """Load WUP 2025 urban share for 1950 and 2025 from F02 Excel file.

    The 'Cities and Towns' sheet contains the total urban percentage
    (100 - rural) using the degree-of-urbanization definition.

    Returns DataFrame with columns: iso3, urban_1950, urban_2025.
    """
    dest = RAW_WPP / "WUP2025-F02-PercUrban.xlsx"
    if not dest.exists():
        _download(WUP_URL, dest)

    import openpyxl  # lazy import — heavy; only needed here

    wb = openpyxl.load_workbook(str(dest), read_only=True, data_only=True)
    ws = wb["Cities and Towns"]

    # Parse header row to find year columns
    header = list(ws.iter_rows(max_row=1, values_only=True))[0]
    col_map = {str(v): i for i, v in enumerate(header) if v is not None}
    col_1950 = col_map["1950"]
    col_2025 = col_map["2025"]

    rows = []
    for row in ws.iter_rows(values_only=True):
        iso3 = row[4]  # ISO3_Code is column index 4
        if not iso3 or not isinstance(iso3, str) or len(iso3) != 3:
            continue
        u1950 = row[col_1950]
        u2025 = row[col_2025]
        if u1950 is None or u2025 is None:
            continue
        rows.append({"iso3": iso3, "urban_1950": float(u1950), "urban_2025": float(u2025)})

    wb.close()
    return pd.DataFrame(rows)


def load_maddison_gdppc_2015() -> pd.DataFrame:
    """Load Maddison Project Database 2023 GDPpc for year 2015.

    GDPpc is in 2011 international USD (PPP).

    Returns DataFrame with columns: iso3, gdppc_2015.
    """
    dest = RAW_MAD / "mpd2023_web.xlsx"
    if not dest.exists():
        _download(MADDISON_URL, dest)

    mad = pd.read_excel(str(dest), sheet_name="Full data")
    mad2015 = (
        mad[mad["year"] == 2015][["countrycode", "gdppc"]]
        .dropna()
        .rename(columns={"countrycode": "iso3", "gdppc": "gdppc_2015"})
        .reset_index(drop=True)
    )
    return mad2015


def load_cbr_series() -> pd.DataFrame:
    """Load combined CBR time series: Gapminder 1800-1949 + OWID 1950-2023.

    Returns long DataFrame with columns: iso3, year, cbr.
    """
    codes = _load_gapminder_codes()

    # --- Gapminder historical CBR 1800-2015 (pre-1950 segment used) --------
    gm_dest = RAW_WPP / "gapminder_cbr_1800_2015.csv"
    if not gm_dest.exists():
        _download(GAPMINDER_CBR_URL, gm_dest)
    gm = pd.read_csv(gm_dest)
    gm = gm.rename(
        columns={
            "time": "year",
            "crude_birth_rate_births_per_1000_population": "cbr",
        }
    )
    gm = gm.merge(codes, left_on="geo", right_on="country", how="left")
    gm_pre1950 = (
        gm[(gm["year"] < 1950) & gm["iso3"].notna()][["iso3", "year", "cbr"]]
        .copy()
    )

    # --- OWID CBR 1950-2023 (WPP 2024, more authoritative for modern era) --
    owid_dest = RAW_WPP / "owid_cbr_1950_2023.csv"
    if not owid_dest.exists():
        _download(OWID_CBR_URL, owid_dest)
    owid = pd.read_csv(owid_dest)
    owid = owid.rename(columns={"Code": "iso3", "Year": "year", "Birth rate": "cbr"})
    owid_post = (
        owid[owid["iso3"].str.match(r"^[A-Z]{3}$", na=False)][["iso3", "year", "cbr"]]
        .copy()
    )

    combined = (
        pd.concat([gm_pre1950, owid_post], ignore_index=True)
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )
    return combined


def load_hyde_popd(years: list[int]) -> pd.DataFrame:
    """Load HYDE 3.5 country-level population density (persons/km²) for the
    requested years from gbc2025_7apr_base/txt/popd_c.txt.

    Returns DataFrame with columns: iso3, popd_<year> for each year in `years`.
    """
    src = ROOT / "gbc2025_7apr_base/txt/popd_c.txt"
    df = pd.read_csv(src, sep=r"\s+", engine="python")
    # First column is 'region' (HYDE numeric iso_num)
    df = df[df["region"].astype(str).str.isdigit()].copy()
    df["iso_num"] = df["region"].astype(int)

    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    df = df.merge(iso_map[["iso_num", "iso3"]], on="iso_num", how="left")
    df = df.dropna(subset=["iso3"]).copy()

    keep = ["iso3"] + [str(y) for y in years]
    sub = df[keep].copy()
    rename = {str(y): f"popd_{y}" for y in years}
    sub = sub.rename(columns=rename)
    for y in years:
        sub[f"popd_{y}"] = pd.to_numeric(sub[f"popd_{y}"], errors="coerce")
    return sub


def compute_dt_timing(cbr_series: pd.DataFrame) -> pd.DataFrame:
    """Return first year CBR < 25/1000 for each country.

    Returns DataFrame with columns: iso3, dt_timing_year (int or NaN).
    """

    def _first_year_below(df: pd.DataFrame) -> float:
        crossed = df.loc[df["cbr"] < 25.0, "year"]
        if crossed.empty:
            return np.nan
        return float(crossed.iloc[0])

    result = (
        cbr_series.groupby("iso3")
        .apply(_first_year_below, include_groups=False)
        .reset_index()
    )
    result.columns = ["iso3", "dt_timing_year"]
    return result


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building modern outcomes panel...")

    # 1. Population growth
    print("\n[1] Population (OWID 1950 + Gapminder 2025)...")
    pop = load_population()
    pop["log_pop_growth_1950_2025"] = np.log(pop["pop2025"] / pop["pop1950"])
    print(f"    {len(pop)} countries")

    # 2. Urban share change
    print("\n[2] Urban share change (WUP 2025)...")
    urban = load_urban_share()
    urban["urban_change_1950_2025"] = urban["urban_2025"] - urban["urban_1950"]
    print(f"    {len(urban)} countries")

    # 3. GDPpc 2015
    print("\n[3] GDPpc 2015 (Maddison 2023)...")
    mad = load_maddison_gdppc_2015()
    mad["log_gdppc_2015"] = np.log(mad["gdppc_2015"])
    print(f"    {len(mad)} countries with valid 2015 GDPpc")

    # 4. Demographic transition timing
    print("\n[4] Demographic transition timing (Gapminder CBR + OWID CBR)...")
    cbr = load_cbr_series()
    dt = compute_dt_timing(cbr)
    n_crossed = dt["dt_timing_year"].notna().sum()
    print(f"    {n_crossed} countries with CBR < 25 crossing identified")

    # 5. HYDE pop density 1500 and 2025
    print("\n[5] HYDE 3.5 population density (1500, 2025)...")
    popd = load_hyde_popd([1500, 2025])
    popd["log_popd_1500"] = np.log(popd["popd_1500"].clip(lower=1e-3))
    popd["log_popd_2025"] = np.log(popd["popd_2025"].clip(lower=1e-3))
    print(f"    {len(popd)} countries from HYDE")
    print(f"    1500 non-null: {popd['popd_1500'].notna().sum()}, "
          f"2025 non-null: {popd['popd_2025'].notna().sum()}")

    # 6. Merge into panel
    print("\n[6] Merging outcomes...")
    # Start from the union of countries that appear in population data
    df = pop[["iso3", "log_pop_growth_1950_2025"]].copy()
    df = df.merge(
        urban[["iso3", "urban_change_1950_2025"]], on="iso3", how="left"
    )
    df = df.merge(
        mad[["iso3", "log_gdppc_2015"]], on="iso3", how="left"
    )
    df = df.merge(dt[["iso3", "dt_timing_year"]], on="iso3", how="left")
    df = df.merge(popd[["iso3", "log_popd_1500", "log_popd_2025"]],
                  on="iso3", how="left")

    df["source"] = (
        "Pop 1950: OWID/WPP; Pop 2025: Gapminder/WPP2024; "
        "Urban: UN WUP 2025 F02 (Cities+Towns); "
        "GDPpc: Maddison Project DB 2023 (2011$); "
        "CBR: Gapminder 1800-1949 + OWID/WPP 1950-2023; "
        "PopDensity: HYDE 3.5 gbc2025_7apr_base"
    )

    # Sort by iso3
    df = df.sort_values("iso3").reset_index(drop=True)

    print(f"\nFinal panel: {len(df)} countries")
    print(f"  log_pop_growth_1950_2025: {df['log_pop_growth_1950_2025'].notna().sum()} non-null")
    print(f"  urban_change_1950_2025:   {df['urban_change_1950_2025'].notna().sum()} non-null")
    print(f"  log_gdppc_2015:           {df['log_gdppc_2015'].notna().sum()} non-null")
    print(f"  dt_timing_year:           {dt['dt_timing_year'].notna().sum()} crossed (NaN = not yet)")
    print(f"  log_popd_1500:            {df['log_popd_1500'].notna().sum()} non-null")
    print(f"  log_popd_2025:            {df['log_popd_2025'].notna().sum()} non-null")

    # Validate ranges
    assert (df["log_pop_growth_1950_2025"].dropna() > -5).all(), "log_pop_growth suspiciously low"
    assert (df["log_pop_growth_1950_2025"].dropna() < 7).all(), "log_pop_growth suspiciously high"
    urban_valid = df["urban_change_1950_2025"].dropna()
    assert urban_valid.between(-100, 100).all(), "urban_change out of plausible range"
    log_gdp_valid = df["log_gdppc_2015"].dropna()
    assert log_gdp_valid.between(5, 13).all(), "log_gdppc_2015 out of plausible range"
    dt_valid = df["dt_timing_year"].dropna()
    assert (dt_valid >= 1800).all(), "dt_timing_year before 1800 is suspicious"
    assert (dt_valid <= 2023).all(), "dt_timing_year after 2023 is impossible"
    # log popd should be in a wide but bounded range (e.g. -8 .. 9 for 0.0003 to 8000 ppl/km²)
    for col in ("log_popd_1500", "log_popd_2025"):
        v = df[col].dropna()
        assert v.between(-10, 10).all(), f"{col} out of plausible range: {v.min()} .. {v.max()}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
