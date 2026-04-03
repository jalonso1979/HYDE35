"""Derived variable construction for Malthusian and UGT analysis."""
import numpy as np
import pandas as pd


def compute_land_labor_ratio(
    cropland: np.ndarray, grazing: np.ndarray, population: np.ndarray
) -> np.ndarray:
    """Arable land per capita: (cropland + grazing) / population."""
    total_land = np.asarray(cropland) + np.asarray(grazing)
    pop = np.asarray(population, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pop > 0, total_land / pop, np.nan)
    return ratio


def compute_ag_output_proxy(
    cropland: np.ndarray,
    grazing: np.ndarray,
    grazing_weight: float = 0.5,
) -> np.ndarray:
    """Agricultural output proxy: cropland + grazing_weight * grazing."""
    return np.asarray(cropland) + grazing_weight * np.asarray(grazing)


def compute_intensification_index(
    irrigation_share: np.ndarray,
    cropland: np.ndarray,
    grazing: np.ndarray,
) -> np.ndarray:
    """Intensification index: irrigation_share * (cropland / (cropland + grazing))."""
    crop = np.asarray(cropland, dtype="float64")
    graz = np.asarray(grazing, dtype="float64")
    total = crop + graz
    with np.errstate(divide="ignore", invalid="ignore"):
        crop_share = np.where(total > 0, crop / total, np.nan)
    return np.asarray(irrigation_share) * crop_share


def compute_pop_growth_rate(
    population: pd.Series, years: pd.Series
) -> pd.Series:
    """Annualized population growth rate between consecutive observations."""
    pop = population.values.astype("float64")
    yrs = years.values.astype("float64")
    growth = np.full(len(pop), np.nan)
    for i in range(1, len(pop)):
        if pop[i - 1] > 0 and not np.isnan(pop[i - 1]):
            dt = yrs[i] - yrs[i - 1]
            if dt > 0:
                growth[i] = (pop[i] / pop[i - 1]) ** (1.0 / dt) - 1.0
    return pd.Series(growth, index=population.index)


def pivot_country_panel_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format country panel to wide format."""
    mean_wide = df.pivot_table(
        index=["year", "country"], columns="var", values="mean"
    ).reset_index()
    mean_wide.columns = [
        f"{c}_mean" if c not in ("year", "country") else c
        for c in mean_wide.columns
    ]
    std_wide = df.pivot_table(
        index=["year", "country"], columns="var", values="std"
    ).reset_index()
    std_wide.columns = [
        f"{c}_std" if c not in ("year", "country") else c
        for c in std_wide.columns
    ]
    return mean_wide.merge(std_wide, on=["year", "country"])
