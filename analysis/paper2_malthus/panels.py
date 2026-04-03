"""Malthusian panel construction for Paper 2."""
import numpy as np
import pandas as pd

def build_malthusian_panel(wide_panel: pd.DataFrame, entity_col: str = "country") -> pd.DataFrame:
    df = wide_panel.sort_values([entity_col, "year"]).copy()
    df["popdens_lag"] = df.groupby(entity_col)["popdens_p_km2_mean"].shift(1)
    df["land_labor_ratio_lag"] = df.groupby(entity_col)["land_labor_ratio"].shift(1)
    df["intens_lag"] = df.groupby(entity_col)["intensification_index"].shift(1)
    df = df.dropna(subset=["popdens_lag"]).copy()
    df["intens_x_density"] = df["intens_lag"] * df["popdens_lag"]
    df["log_popdens_lag"] = np.log(np.maximum(df["popdens_lag"], 1e-10))
    df["log_land_labor_lag"] = np.log(np.maximum(df["land_labor_ratio_lag"], 1e-10))
    return df.reset_index(drop=True)
