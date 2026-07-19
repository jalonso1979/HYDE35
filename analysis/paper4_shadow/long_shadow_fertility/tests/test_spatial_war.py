"""Tests for the nearby-war spatial spillover variable."""
import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.spatial_war import (
    CAPITALS,
    add_nearby_war,
    haversine_km,
    inverse_distance_weights,
)


def test_row_weights_sum_to_one():
    units = list(CAPITALS)
    W = inverse_distance_weights(units)
    # Each row of weights (over the other countries) sums to 1.
    np.testing.assert_allclose(W.sum(axis=1).to_numpy(), np.ones(len(units)), atol=1e-9)
    # Diagonal (self-weight) is exactly zero.
    np.testing.assert_allclose(np.diag(W.to_numpy()), np.zeros(len(units)), atol=1e-12)


def test_haversine_known_distance():
    # London <-> Paris is ~340-345 km.
    d = haversine_km(*CAPITALS["GBR"], *CAPITALS["FRA"])
    assert 330.0 < d < 360.0


def test_war_torn_neighbour_raises_nearby_war():
    """A country whose near neighbour is war-torn gets a higher nearby_war than
    a country whose neighbours are peaceful."""
    # One year, four countries. FRA's nearest neighbour (BEL) is at war (intensity 10);
    # SWE's neighbours are all peaceful (intensity 0). FRA should score higher.
    df = pd.DataFrame(
        {
            "iso3": ["FRA", "BEL", "SWE", "ITA"],
            "year": [1900, 1900, 1900, 1900],
            "log_war_fatalities": [0.0, 10.0, 0.0, 0.0],
        }
    )
    out = add_nearby_war(df)
    nearby = out.set_index("iso3")["nearby_war"]
    # FRA is adjacent to the war-torn BEL; SWE is far away, so FRA feels it more.
    assert nearby["FRA"] > nearby["SWE"]
    # BEL itself has no war-torn neighbours here, so its nearby_war is 0.
    assert nearby["BEL"] == 0.0
    # Every country's nearby_war is a weighted slice of BEL's intensity (in (0, 10)).
    assert 0.0 < nearby["FRA"] < 10.0
    assert 0.0 < nearby["SWE"] < 10.0


def test_closer_neighbour_weighted_more():
    """Given two equally war-torn neighbours, the closer one drives a higher
    nearby_war for the focal country (NLD is closer to BEL than ESP is to BEL)."""
    df = pd.DataFrame(
        {
            "iso3": ["BEL", "NLD", "ESP"],
            "year": [1900, 1900, 1900],
            "log_war_fatalities": [5.0, 0.0, 0.0],
        }
    )
    out = add_nearby_war(df).set_index("iso3")["nearby_war"]
    # NLD (adjacent to BEL) should feel BEL's war more than distant ESP does.
    assert out["NLD"] > out["ESP"]


def test_output_length_matches_input():
    df = pd.DataFrame(
        {
            "iso3": ["FRA", "BEL", "SWE", "FRA", "BEL", "SWE"],
            "year": [1900, 1900, 1900, 1901, 1901, 1901],
            "log_war_fatalities": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    out = add_nearby_war(df)
    assert len(out) == len(df)
    assert "nearby_war" in out.columns
    assert out["nearby_war"].notna().all()


def test_weights_renormalise_when_country_absent():
    """If a country is absent in a given year, the remaining neighbours' effective
    weights still sum to 1 (verified via a fully-uniform-intensity check)."""
    # All present countries have intensity 1 in year 1901; ESP absent that year.
    df = pd.DataFrame(
        {
            "iso3": ["FRA", "BEL", "SWE"],
            "year": [1901, 1901, 1901],
            "log_war_fatalities": [1.0, 1.0, 1.0],
        }
    )
    out = add_nearby_war(df).set_index("iso3")["nearby_war"]
    # With every neighbour at intensity 1 and renormalised weights summing to 1,
    # each country's nearby_war must equal 1.
    np.testing.assert_allclose(out.to_numpy(), np.ones(len(out)), atol=1e-9)
