import pytest

import pycountry

# In order to avoid triggering data loads and graph plots that fail in testing context,
# we use the ast parsing method to extract the function since we patched it manually above.
# Wait, actually we ALREADY wrapped it in `if __name__ == "__main__":` in our manual patch above!
# So we can just import it directly now.

from analysis.figures.fig2_density_divergence import iso_num_to_alpha3

def test_iso_num_to_alpha3_valid():
    """Test with valid ISO numeric code"""
    assert iso_num_to_alpha3(840) == "USA"
    assert iso_num_to_alpha3("840") == "USA"
    assert iso_num_to_alpha3(8) == "ALB"
    assert iso_num_to_alpha3("008") == "ALB"

def test_iso_num_to_alpha3_invalid_numeric():
    """Test with invalid ISO numeric code that is still a number"""
    assert iso_num_to_alpha3(999) == "999"
    assert iso_num_to_alpha3("999") == "999"

def test_iso_num_to_alpha3_invalid_string():
    """Test with uncastable string"""
    assert iso_num_to_alpha3("invalid") == "invalid"

def test_iso_num_to_alpha3_type_error():
    """Test with empty/none values that raise Exception in casting"""
    assert iso_num_to_alpha3(None) == "None"
