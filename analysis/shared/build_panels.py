"""Build analysis-ready panel datasets from existing HYDE35 CSVs.

Reads the pre-processed country and region panels, pivots to wide format,
computes derived Malthusian variables, and writes Parquet files.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.shared.config import (
    COUNTRY_YEAR_CSV,
    REGION_YEAR_CSV,
    ANTHRO_REGION_CSV,
    ANALYSIS_DATA,
    GRAZING_CALORIC_WEIGHT,
)
from analysis.shared.variables import (
    compute_land_labor_ratio,
    compute_ag_output_proxy,
    compute_intensification_index,
    compute_pop_growth_rate,
    pivot_country_panel_wide,
)
from analysis.shared.loaders import load_existing_country_panel, load_existing_scenario_panel
from analysis.shared.uncertainty import compute_scenario_stats


def build_country_analysis_panel() -> pd.DataFrame:
    """Build the country-level analysis panel with derived variables.

    Country CSV vars: nonrice_mha, pop_persons, irrigation_share, rice_mha,
    grazing_mha, etc.  After pivot_country_panel_wide these become
    <var>_mean and <var>_std columns.
    """
    df_long = load_existing_country_panel(COUNTRY_YEAR_CSV)
    df = pivot_country_panel_wide(df_long)

    # Country panel uses 'nonrice_mha' (not 'nonrice_cropland_mha')
    cropland = df.get("nonrice_mha_mean", pd.Series(0.0, index=df.index))
    rice = df.get("rice_mha_mean", pd.Series(0.0, index=df.index))
    total_cropland = cropland + rice
    grazing = df.get("grazing_mha_mean", pd.Series(0.0, index=df.index))
    # Country panel uses 'pop_persons' (not 'pop')
    pop = df.get("pop_persons_mean", pd.Series(0.0, index=df.index))
    # Country panel uses 'irrigation_share' (not 'irrigation_share_mean')
    irr_share = df.get("irrigation_share_mean", pd.Series(0.0, index=df.index))

    df["land_labor_ratio"] = compute_land_labor_ratio(
        total_cropland.values, grazing.values, pop.values
    )
    df["ag_output_proxy_mha"] = compute_ag_output_proxy(
        total_cropland.values, grazing.values, grazing_weight=GRAZING_CALORIC_WEIGHT
    )
    df["intensification_index"] = compute_intensification_index(
        irr_share.values, total_cropland.values, grazing.values
    )

    df = df.sort_values(["country", "year"])
    growth_parts = []
    for _, grp in df.groupby("country"):
        g = compute_pop_growth_rate(grp["pop_persons_mean"], grp["year"])
        growth_parts.append(g)
    df["pop_growth_rate"] = pd.concat(growth_parts)

    return df


def build_region_analysis_panel() -> pd.DataFrame:
    """Build the region-level analysis panel with scenario stats and derived vars.

    Region CSV vars: nonrice_cropland_mha, pop, rice_mha, grazing_mha, etc.
    No irrigation_share in the region panel.
    """
    df_scen = load_existing_scenario_panel(REGION_YEAR_CSV)
    df_stats = compute_scenario_stats(df_scen)

    mean_wide = df_stats.pivot_table(
        index=["year", "region"], columns="var", values="mean"
    ).reset_index()
    mean_wide.columns = [
        f"{c}_mean" if c not in ("year", "region") else c
        for c in mean_wide.columns
    ]

    std_wide = df_stats.pivot_table(
        index=["year", "region"], columns="var", values="std"
    ).reset_index()
    std_wide.columns = [
        f"{c}_std" if c not in ("year", "region") else c
        for c in std_wide.columns
    ]

    se_wide = df_stats.pivot_table(
        index=["year", "region"], columns="var", values="se"
    ).reset_index()
    se_wide.columns = [
        f"{c}_se" if c not in ("year", "region") else c
        for c in se_wide.columns
    ]

    df = mean_wide.merge(std_wide, on=["year", "region"]).merge(se_wide, on=["year", "region"])

    # Region panel uses 'nonrice_cropland_mha' (not 'nonrice_mha')
    cropland = df.get("nonrice_cropland_mha_mean", pd.Series(0.0, index=df.index))
    rice = df.get("rice_mha_mean", pd.Series(0.0, index=df.index))
    total_cropland = cropland + rice
    grazing = df.get("grazing_mha_mean", pd.Series(0.0, index=df.index))
    # Region panel uses 'pop' (not 'pop_persons')
    pop = df.get("pop_mean", pd.Series(0.0, index=df.index))

    df["land_labor_ratio"] = compute_land_labor_ratio(
        total_cropland.values, grazing.values, pop.values
    )
    df["ag_output_proxy_mha"] = compute_ag_output_proxy(
        total_cropland.values, grazing.values, grazing_weight=GRAZING_CALORIC_WEIGHT
    )

    df = df.sort_values(["region", "year"])
    growth_parts = []
    for _, grp in df.groupby("region"):
        g = compute_pop_growth_rate(grp["pop_mean"], grp["year"])
        growth_parts.append(g)
    df["pop_growth_rate"] = pd.concat(growth_parts)

    return df


def save_panels():
    """Build and save all analysis-ready panels to Parquet."""
    print("Building country analysis panel...")
    country_df = build_country_analysis_panel()
    country_path = ANALYSIS_DATA / "country_analysis_panel.parquet"
    country_df.to_parquet(country_path, index=False)
    print(f"  Saved {len(country_df)} rows to {country_path}")

    print("Building region analysis panel...")
    region_df = build_region_analysis_panel()
    region_path = ANALYSIS_DATA / "region_analysis_panel.parquet"
    region_df.to_parquet(region_path, index=False)
    print(f"  Saved {len(region_df)} rows to {region_path}")

    print("Done.")


if __name__ == "__main__":
    save_panels()
