"""Spatial war spillover: inverse-distance-weighted war intensity of neighbours.

For each country-year (c, t) we compute the weather/conflict spillover from the
OTHER countries in the panel that same year:

    nearby_war_ct = sum_{j != c} w_cj * intensity_jt

with row-normalised inverse great-circle distance weights

    w_cj = (1 / d_cj) / sum_{k != c} (1 / d_ck).

Distances are great-circle (haversine) distances between capital cities. This
gives a single scalar per country-year capturing how war-torn a country's
neighbourhood is, used as a spatial-spillover shock in the extended system
LP-FEVD (Cholesky-ordered before own-war: a war next door is plausibly
predetermined relative to own fertility).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Capital-city coordinates (lat, lon) in degrees.
CAPITALS: dict[str, tuple[float, float]] = {
    "GBR": (51.51, -0.13),   # London
    "FRA": (48.86, 2.35),    # Paris
    "ITA": (41.90, 12.50),   # Rome
    "SWE": (59.33, 18.07),   # Stockholm
    "BEL": (50.85, 4.35),    # Brussels
    "NLD": (52.37, 4.90),    # Amsterdam
    "ESP": (40.42, -3.70),   # Madrid
    "NOR": (59.91, 10.75),   # Oslo
    "DNK": (55.68, 12.57),   # Copenhagen
    "FIN": (60.17, 24.94),   # Helsinki
    "ISL": (64.15, -21.94),  # Reykjavik
    "CHE": (46.95, 7.45),    # Bern
}

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points in degrees."""
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def inverse_distance_weights(
    units: list[str], capitals: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Row-normalised inverse great-circle distance weight matrix.

    Returns a DataFrame indexed and columned by `units`; row c gives the weights
    w_cj on every other unit j (the diagonal is 0, each row sums to 1).
    """
    capitals = CAPITALS if capitals is None else capitals
    n = len(units)
    inv = np.zeros((n, n), dtype=float)
    for i, c in enumerate(units):
        lat_c, lon_c = capitals[c]
        for k, j in enumerate(units):
            if i == k:
                continue
            lat_j, lon_j = capitals[j]
            d = haversine_km(lat_c, lon_c, lat_j, lon_j)
            inv[i, k] = 1.0 / d if d > 0 else 0.0
    row_sums = inv.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    w = inv / row_sums
    return pd.DataFrame(w, index=units, columns=units)


def add_nearby_war(
    df: pd.DataFrame,
    intensity_col: str = "log_war_fatalities",
    unit_col: str = "iso3",
    year_col: str = "year",
    out_col: str = "nearby_war",
    capitals: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Add an inverse-distance-weighted nearby-war spillover column.

    For each (c, t): nearby_war = sum_{j != c} w_cj * intensity_jt, where the
    weights are row-normalised inverse great-circle distances between capitals
    and the sum runs over the OTHER units present in the panel that year.

    Missing neighbour intensity values are treated as 0 (no recorded war
    intensity that year). Weights are RE-normalised within the set of neighbours
    that actually appear in a given year so that, if some country is absent that
    year, the remaining neighbours' weights still sum to 1.

    Returns a copy of `df` with `out_col` appended (same length / index as input).
    """
    capitals = CAPITALS if capitals is None else capitals
    out = df.copy()
    units = [u for u in out[unit_col].unique() if u in capitals]
    W = inverse_distance_weights(units, capitals=capitals)

    # Pivot intensity to a year x unit matrix (NaN where a country-year is absent).
    wide = out.pivot_table(index=year_col, columns=unit_col, values=intensity_col, aggfunc="mean")
    wide = wide.reindex(columns=units)

    present = wide.notna().astype(float)          # 1 where (year, unit) intensity exists
    intensity = wide.fillna(0.0)                  # 0 contribution where absent

    nearby = pd.DataFrame(index=wide.index, columns=units, dtype=float)
    for c in units:
        w_c = W.loc[c, units].to_numpy()          # weight on each unit j (0 for j == c)
        # Re-normalise weights to the neighbours present each year (exclude c).
        eff_w = present.to_numpy() * w_c          # (n_years, n_units)
        eff_w[:, units.index(c)] = 0.0            # never weight self
        denom = eff_w.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0.0, 1.0, denom)
        eff_w = eff_w / denom
        nearby[c] = (eff_w * intensity.to_numpy()).sum(axis=1)

    # Map (year, unit) -> nearby value back onto the original rows.
    long = nearby.reset_index().melt(id_vars=year_col, var_name=unit_col, value_name=out_col)
    merged = out.merge(long, on=[year_col, unit_col], how="left")
    merged.index = out.index
    return merged


if __name__ == "__main__":
    W = inverse_distance_weights(list(CAPITALS))
    print("Row-normalised inverse-distance weight matrix (capitals):")
    print(W.round(3))
    print("\nRow sums (should all be 1):")
    print(W.sum(axis=1).round(6).to_dict())
