"""Transition trajectory feature extraction for Paper 1."""
import numpy as np
import pandas as pd


def detect_peak_extensification(years: np.ndarray, land_area: np.ndarray) -> float:
    if len(years) < 2:
        return np.nan
    growth = np.diff(land_area) / np.diff(years)
    idx = np.argmax(growth)
    return float(years[idx + 1])


def detect_intensification_onset(
    years: np.ndarray, intensification_index: np.ndarray, land_growth_rate: np.ndarray,
    intens_threshold: float = 0.05, land_growth_threshold: float = 0.01,
) -> float:
    for i in range(len(years)):
        if np.isnan(land_growth_rate[i]):
            continue
        if (intensification_index[i] >= intens_threshold
                and land_growth_rate[i] < land_growth_threshold):
            return float(years[i])
    return np.nan


def detect_urbanization_takeoff(
    years: np.ndarray, urban_share: np.ndarray, threshold: float = 0.05,
) -> float:
    for i in range(len(years)):
        if urban_share[i] >= threshold:
            return float(years[i])
    return np.nan


def _compute_land_growth_rate(years: np.ndarray, land_area: np.ndarray) -> np.ndarray:
    growth = np.full(len(years), np.nan)
    for i in range(1, len(years)):
        dt = years[i] - years[i - 1]
        if dt > 0 and land_area[i - 1] > 0:
            growth[i] = (land_area[i] - land_area[i - 1]) / (land_area[i - 1] * dt)
    return growth


def extract_trajectory_features(df: pd.DataFrame, entity_col: str = "country") -> pd.DataFrame:
    results = []
    for entity, grp in df.sort_values("year").groupby(entity_col):
        years = grp["year"].values
        land_area = grp["ag_output_proxy_mha"].values
        intens = grp["intensification_index"].values
        urban = grp["urban_share_mean"].values
        popdens = grp["popdens_p_km2_mean"].values
        llr = grp["land_labor_ratio"].values
        land_growth = _compute_land_growth_rate(years, land_area)
        results.append({
            entity_col: entity,
            "peak_extensification_year": detect_peak_extensification(years, land_area),
            "intensification_onset_year": detect_intensification_onset(years, intens, land_growth),
            "urbanization_takeoff_year": detect_urbanization_takeoff(years, urban),
            "max_pop_density": float(np.nanmax(popdens)),
            "min_land_labor_ratio": float(np.nanmin(llr)),
        })
    return pd.DataFrame(results)
