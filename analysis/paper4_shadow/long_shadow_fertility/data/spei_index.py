"""Standardized Precipitation Evapotranspiration Index (SPEI) for the Long Shadow panel.

Approach: Path A — True SPEI using reconstructed absolute T and P.

Construction steps
------------------
1. Baseline climatology (1901-1950 monthly means):
   ``cru_country_climatology_1901_1950.parquet`` — pre-computed country-month
   means of CRU TS absolute temperature (tmp_c_clim) and precipitation
   (pre_mm_clim).  This is a defensible reference period that avoids
   contamination with post-industrial warming while covering all ModE-RA
   countries.

   Note: the canonical WMO baseline is 1961-1990.  ERA5 country monthly data
   exists for 1950-2025 but has large monthly gaps in 1968-1990 for most
   European countries (artefact of the ERA5 country aggregation pipeline),
   making it unsuitable as a stand-alone 30-year baseline.  The CRU 1901-1950
   baseline is complete and is standard in historical climate reconstructions
   (Vicente-Serrano et al. 2010 use it for pre-instrumental SPEI).

2. Reconstruct absolute monthly T and P:
   T_abs[c, y, m] = CRU_clim[c, m].tmp_c + ModERA_anom[c, y, m].t_anom_c
   P_abs[c, y, m] = CRU_clim[c, m].pre_mm + ModERA_anom[c, y, m].p_anom_mm

3. Compute monthly PET via the Thornthwaite (1948) method:
   - Requires mean monthly temperature (°C) and centroid latitude.
   - Computes the heat index I = sum_{m=1..12}(max(T_m/5, 0)^1.514) from the
     long-run monthly mean T (averaged over the full ModE-RA period per
     country-month, since Thornthwaite intended climatological means).
   - Unadjusted PET_0 = 16 * (10 * T / I)^a  mm/month (when T > 0°C)
   - Daylight-hour correction: multiplied by N_m / (12 * 30), where N_m is the
     mean monthly daylight hours derived from latitude via the standard
     astronomical formula.
   - Clipped to zero when T ≤ 0°C.

4. Water-balance deficit:
   D[c, y, m] = P_abs[c, y, m] - PET[c, y, m]   (mm/month)

5. Growing-season accumulation (April–September):
   D_gs[c, y] = sum_{m=4..9} D[c, y, m]

6. Standardisation per country:
   spei_growing[c, y] = (D_gs[c, y] - mean_c(D_gs)) / std_c(D_gs)
   (zero-mean, unit-variance per country, preserving cross-year comparisons)

7. Analogously for winter (Oct–Mar, using winter_year convention) and annual.

References
----------
- Thornthwaite, C.W. (1948). "An approach toward a rational classification
  of climate." Geographical Review 38(1): 55-94.
- Vicente-Serrano, S.M., Begueria, S., Lopez-Moreno, J.I. (2010). "A
  Multiscalar Drought Index Sensitive to Global Warming: the Standardized
  Precipitation Evapotranspiration Index." Journal of Climate 23: 1696-1718.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Country centroid latitudes (degrees N) — from hyde_era5_extended_panel
# ---------------------------------------------------------------------------
COUNTRY_LATITUDES: dict[str, float] = {
    "GBR": 54.10,
    "FRA": 46.46,
    "ITA": 42.54,
    "SWE": 62.29,
    "BEL": 50.62,
    "NLD": 52.24,
    "ESP": 40.04,
}

# Default paths
import os as _os
_ROOT = "/Volumes/BIGDATA/HYDE35/analysis"
CRU_CLIM_PATH = _ROOT + "/data/cru_country_climatology_1901_1950.parquet"
MODERA_PATH = _ROOT + "/data/modera_country_monthly_cropw.parquet"


# ---------------------------------------------------------------------------
# Astronomical daylight hours
# ---------------------------------------------------------------------------

def _mean_monthly_daylight_hours(lat_deg: float) -> np.ndarray:
    """Return array[12] of mean daylight hours for months 1..12.

    Uses the standard astronomical formula:
        N = (24/pi) * arccos(-tan(phi) * tan(delta))
    where delta is the solar declination at the 15th of each month.

    Parameters
    ----------
    lat_deg : float
        Latitude in decimal degrees (positive = North).

    Returns
    -------
    np.ndarray shape (12,)
    """
    phi = np.radians(lat_deg)
    # Day-of-year at the 15th of each month
    mid_days = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349],
                        dtype=float)
    # Solar declination (radians)
    delta = 0.4093 * np.sin(2 * np.pi * (mid_days - 81) / 365)
    arg = -np.tan(phi) * np.tan(delta)
    # Clamp to avoid domain errors
    arg = np.clip(arg, -1.0, 1.0)
    omega = np.arccos(arg)  # half-day-length angle in radians
    N = (24.0 / np.pi) * omega  # daylight hours
    return N  # shape (12,)


# ---------------------------------------------------------------------------
# Thornthwaite PET
# ---------------------------------------------------------------------------

def _thornthwaite_monthly_pet(
    T_monthly_clim: np.ndarray,
    T_abs_monthly: np.ndarray,
    N_hours: np.ndarray,
) -> np.ndarray:
    """Compute Thornthwaite PET (mm/month) for one country-year sequence.

    Parameters
    ----------
    T_monthly_clim : np.ndarray, shape (12,)
        Long-run monthly mean temperature (°C) used to compute heat index I.
        Typically the CRU 1901-1950 monthly climatology for the country.
    T_abs_monthly : np.ndarray, shape (n_years * 12,) or (n_years, 12)
        Absolute monthly temperature (°C) for all years.
    N_hours : np.ndarray, shape (12,)
        Mean monthly daylight hours (from _mean_monthly_daylight_hours).

    Returns
    -------
    np.ndarray same shape as T_abs_monthly
        Monthly PET in mm.
    """
    # Heat index I — computed from climatological means
    T_pos = np.maximum(T_monthly_clim, 0.0)
    I = np.sum((T_pos / 5.0) ** 1.514)
    if I < 1e-6:
        # Polar region: no PET
        return np.zeros_like(T_abs_monthly)

    # Thornthwaite exponent
    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239

    # PET-0 (unadjusted): shape (n, 12) or flat
    T2d = np.atleast_2d(T_abs_monthly)  # (n_years, 12)
    T2d_pos = np.maximum(T2d, 0.0)
    pet0 = 16.0 * (10.0 * T2d_pos / I) ** a  # mm/month unscaled

    # Daylight correction: multiply by N_m / (12 * 30)
    # N_hours shape (12,) -> broadcast over years
    correction = N_hours[np.newaxis, :] / (12.0 * 30.0)
    pet = pet0 * correction

    # Zero out months where T ≤ 0
    pet = np.where(T2d <= 0.0, 0.0, pet)

    return pet.reshape(T_abs_monthly.shape)


# ---------------------------------------------------------------------------
# Main SPEI builder
# ---------------------------------------------------------------------------

def compute_spei(
    countries: list[str] | None = None,
    cru_clim_path: str = CRU_CLIM_PATH,
    modera_path: str = MODERA_PATH,
    country_latitudes: dict[str, float] | None = None,
    growing_season_months: tuple[int, ...] = (4, 5, 6, 7, 8, 9),
    winter_months: tuple[int, ...] = (10, 11, 12, 1, 2, 3),
) -> pd.DataFrame:
    """Build the SPEI growing-season, winter, and annual index.

    Returns a DataFrame with columns:
        iso3, year, spei_growing, spei_winter, spei_annual

    All three SPEI variants are standardised per country (zero mean, unit
    variance across available years).

    Parameters
    ----------
    countries : list[str] | None
        ISO-3 codes to process. Default: COUNTRY_LATITUDES keys.
    cru_clim_path : str
        Path to cru_country_climatology_1901_1950.parquet.
    modera_path : str
        Path to modera_country_monthly_cropw.parquet.
    country_latitudes : dict[str, float] | None
        Country centroid latitudes. Default: COUNTRY_LATITUDES.
    growing_season_months : tuple
        Calendar months defining growing season (default April-September).
    winter_months : tuple
        Calendar months defining winter (default Oct-Mar).
    """
    if country_latitudes is None:
        country_latitudes = COUNTRY_LATITUDES
    if countries is None:
        countries = list(country_latitudes.keys())

    # --- Load data ---
    clim = pd.read_parquet(cru_clim_path)
    # columns: iso3, month, tmp_c_clim, pre_mm_clim
    clim = clim[clim["iso3"].isin(countries)].copy()

    mod = pd.read_parquet(modera_path)
    # columns: iso3, year, month, t_anom_c, p_anom_mm
    mod = mod[mod["iso3"].isin(countries)].copy()

    # --- Reconstruct absolute T and P ---
    merged = mod.merge(
        clim[["iso3", "month", "tmp_c_clim", "pre_mm_clim"]],
        on=["iso3", "month"],
        how="left",
    )
    merged["T_abs"] = merged["tmp_c_clim"] + merged["t_anom_c"]
    merged["P_abs"] = merged["pre_mm_clim"] + merged["p_anom_mm"]
    # Precipitation cannot be negative
    merged["P_abs"] = merged["P_abs"].clip(lower=0.0)

    all_rows: list[pd.DataFrame] = []

    for iso3 in countries:
        lat = country_latitudes.get(iso3)
        if lat is None:
            continue

        sub = merged[merged["iso3"] == iso3].sort_values(["year", "month"]).copy()
        if sub.empty:
            continue

        # Climatological monthly means for this country (for heat index)
        clim_iso = clim[clim["iso3"] == iso3].set_index("month")["tmp_c_clim"]
        T_clim_12 = np.array([clim_iso.get(m, 0.0) for m in range(1, 13)])

        # Daylight hours
        N_hours = _mean_monthly_daylight_hours(lat)

        # Build year × month arrays
        years = sorted(sub["year"].unique())
        year_idx = {y: i for i, y in enumerate(years)}
        n_years = len(years)

        T_mat = np.full((n_years, 12), np.nan)
        P_mat = np.full((n_years, 12), np.nan)

        for _, row in sub.iterrows():
            yi = year_idx[row["year"]]
            mi = int(row["month"]) - 1
            T_mat[yi, mi] = row["T_abs"]
            P_mat[yi, mi] = row["P_abs"]

        # PET: shape (n_years, 12)
        pet_mat = _thornthwaite_monthly_pet(T_clim_12, T_mat, N_hours)

        # Water deficit D = P - PET
        D_mat = P_mat - pet_mat  # (n_years, 12), mm/month

        # ---- Seasonal accumulations ----
        # Growing season (April-September): month indices 3..8
        gs_idx = [m - 1 for m in growing_season_months]
        D_gs = np.nansum(D_mat[:, gs_idx], axis=1)  # (n_years,)

        # Winter: Oct-Mar using winter_year convention
        # winter_year y = Oct(y) + Nov(y) + Dec(y) + Jan(y+1) + Feb(y+1) + Mar(y+1)
        oct_dec_idx = [m - 1 for m in [10, 11, 12]]
        jan_mar_idx = [m - 1 for m in [1, 2, 3]]
        D_oct_dec = np.nansum(D_mat[:, oct_dec_idx], axis=1)  # for year y
        D_jan_mar = np.nansum(D_mat[:, jan_mar_idx], axis=1)  # for year y
        # winter at year y = D_oct_dec[y] + D_jan_mar[y+1]
        D_winter = np.full(n_years, np.nan)
        for i in range(n_years - 1):
            D_winter[i] = D_oct_dec[i] + D_jan_mar[i + 1]
        # Last year: no winter (no following January-March)

        # Annual
        D_annual = np.nansum(D_mat, axis=1)  # (n_years,)

        # ---- Standardise per country ----
        def _standardise(arr: np.ndarray) -> np.ndarray:
            m = np.nanmean(arr)
            s = np.nanstd(arr, ddof=1)
            if s < 1e-12:
                return np.zeros_like(arr, dtype=float)
            return (arr - m) / s

        spei_gs = _standardise(D_gs)
        spei_win = _standardise(D_winter)
        spei_ann = _standardise(D_annual)

        country_df = pd.DataFrame({
            "iso3": iso3,
            "year": years,
            "spei_growing": spei_gs,
            "spei_winter": spei_win,
            "spei_annual": spei_ann,
        })
        all_rows.append(country_df)

    if not all_rows:
        return pd.DataFrame(columns=["iso3", "year", "spei_growing",
                                     "spei_winter", "spei_annual"])

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["iso3", "year"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    df = compute_spei()
    print(f"SPEI index: {len(df)} rows, {df['iso3'].nunique()} countries")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(df.groupby("iso3")[["spei_growing", "spei_winter", "spei_annual"]]
          .describe().round(3).to_string())
