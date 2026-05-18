# analysis/paper5_horserace/subsamples.py
"""Hand-coded sub-sample drop lists for the robustness battery."""
from __future__ import annotations

import pandas as pd

# Acemoglu-Johnson-Robinson (2001) colonial-origins-of-comparative-development
# countries with high colonial-extraction-history scores. List from AJR Table 1.
AJR_COLONIAL = [
    "ARG", "BFA", "BGD", "BLZ", "BOL", "BRA", "CAF", "CHL", "CMR",
    "COD", "COG", "COL", "CRI", "DOM", "DZA", "ECU", "EGY", "ETH",
    "GAB", "GHA", "GIN", "GTM", "GUF", "GUY", "HND", "HTI", "IDN",
    "IND", "JAM", "KEN", "MAR", "MDG", "MEX", "MLI", "MOZ", "MWI",
    "NER", "NGA", "NIC", "PAK", "PAN", "PER", "PHL", "PRY", "RWA",
    "SDN", "SEN", "SLE", "SLV", "SUR", "TGO", "TUN", "TZA", "UGA",
    "URY", "VEN", "VNM", "ZAF", "ZMB", "ZWE",
]

# Small-island states with <500k population in 1950 (UN classification).
SMALL_ISLAND = [
    "ATG", "BHS", "BHR", "BRB", "BLZ", "COM", "DMA", "FJI", "GRD",
    "ISL", "KIR", "MDV", "MLT", "MHL", "MUS", "FSM", "NRU", "PLW",
    "STP", "SYC", "SLB", "TON", "TTO", "TUV", "VUT", "WSM", "LCA",
    "VCT", "KNA",
]

# Post-Columbian Americas (24 countries settled post-1492 by European migrants).
AMERICAS_POST_1492 = [
    "ARG", "BLZ", "BOL", "BRA", "CAN", "CHL", "COL", "CRI", "CUB",
    "DOM", "ECU", "GTM", "GUY", "HND", "HTI", "JAM", "MEX", "NIC",
    "PAN", "PER", "PRY", "SLV", "SUR", "URY", "USA", "VEN",
]

# Ashraf-Galor heavily-imputed countries (ancestry-adjusted heterozygosity
# is heavily reweighted because post-1500 population is largely descendants
# of migrant ancestors).
AG_HEAVILY_IMPUTED = [
    "AUS", "CAN", "NZL", "USA", "ARG", "URY", "BRA", "CHL",
]


# ---------------------------------------------------------------------------
# UN M.49 continent classification, condensed to 6 groups.
# Keys are ISO3 alpha-3 codes; values are short continent codes:
#   AF=Africa, AS=Asia, EU=Europe, AM_N=North America, AM_S=South America,
#   OC=Oceania
# ---------------------------------------------------------------------------
CONTINENT_MAP: dict[str, str] = {
    # Africa
    "AGO": "AF", "BEN": "AF", "BFA": "AF", "BDI": "AF", "CMR": "AF",
    "CPV": "AF", "CAF": "AF", "TCD": "AF", "COM": "AF", "COG": "AF",
    "COD": "AF", "CIV": "AF", "DJI": "AF", "EGY": "AF", "GNQ": "AF",
    "ERI": "AF", "ESH": "AF", "ETH": "AF", "GAB": "AF", "GMB": "AF",
    "GHA": "AF", "GIN": "AF", "GNB": "AF", "KEN": "AF", "LSO": "AF",
    "LBR": "AF", "LBY": "AF", "MDG": "AF", "MWI": "AF", "MLI": "AF",
    "MRT": "AF", "MUS": "AF", "MYT": "AF", "MAR": "AF", "MOZ": "AF",
    "NAM": "AF", "NER": "AF", "NGA": "AF", "REU": "AF", "RWA": "AF",
    "STP": "AF", "SEN": "AF", "SYC": "AF", "SLE": "AF", "SOM": "AF",
    "ZAF": "AF", "SSD": "AF", "SDN": "AF", "SWZ": "AF", "TZA": "AF",
    "TGO": "AF", "TUN": "AF", "UGA": "AF", "ZMB": "AF", "ZWE": "AF",
    "DZA": "AF", "BWA": "AF",
    # Asia
    "AFG": "AS", "ARM": "AS", "AZE": "AS", "BHR": "AS", "BGD": "AS",
    "BTN": "AS", "BRN": "AS", "KHM": "AS", "CHN": "AS", "CYP": "AS",
    "GEO": "AS", "HKG": "AS", "IND": "AS", "IDN": "AS", "IRN": "AS",
    "IRQ": "AS", "ISR": "AS", "JPN": "AS", "JOR": "AS", "KAZ": "AS",
    "PRK": "AS", "KOR": "AS", "KWT": "AS", "KGZ": "AS", "LAO": "AS",
    "LBN": "AS", "MYS": "AS", "MDV": "AS", "MNG": "AS", "MMR": "AS",
    "NPL": "AS", "OMN": "AS", "PAK": "AS", "PHL": "AS", "QAT": "AS",
    "SAU": "AS", "SGP": "AS", "LKA": "AS", "SYR": "AS", "TWN": "AS",
    "TJK": "AS", "THA": "AS", "TLS": "AS", "TUR": "AS", "TKM": "AS",
    "ARE": "AS", "UZB": "AS", "VNM": "AS", "YEM": "AS", "PSE": "AS",
    # Europe
    "ALB": "EU", "AND": "EU", "AUT": "EU", "BLR": "EU", "BEL": "EU",
    "BIH": "EU", "BGR": "EU", "HRV": "EU", "CZE": "EU", "DNK": "EU",
    "EST": "EU", "FIN": "EU", "FRA": "EU", "FRO": "EU", "DEU": "EU",
    "GRC": "EU", "GRL": "EU", "HUN": "EU", "ISL": "EU", "IRL": "EU",
    "ITA": "EU", "LVA": "EU", "LIE": "EU", "LTU": "EU", "LUX": "EU",
    "MLT": "EU", "MDA": "EU", "MCO": "EU", "MNE": "EU", "NLD": "EU",
    "MKD": "EU", "NOR": "EU", "POL": "EU", "PRT": "EU", "ROU": "EU",
    "RUS": "EU", "SMR": "EU", "SRB": "EU", "SJM": "EU", "SVK": "EU",
    "SVN": "EU", "ESP": "EU", "SWE": "EU", "CHE": "EU", "UKR": "EU",
    "GBR": "EU", "VAT": "EU", "UNK": "EU",  # UNK = Kosovo
    # North America (inc. Central America + Caribbean)
    "ATG": "AM_N", "BHS": "AM_N", "BRB": "AM_N", "BLZ": "AM_N",
    "CAN": "AM_N", "CRI": "AM_N", "CUB": "AM_N", "DMA": "AM_N",
    "DOM": "AM_N", "SLV": "AM_N", "GRD": "AM_N", "GLP": "AM_N",
    "GTM": "AM_N", "HTI": "AM_N", "HND": "AM_N", "JAM": "AM_N",
    "KNA": "AM_N", "LCA": "AM_N", "MEX": "AM_N", "MTQ": "AM_N",
    "NIC": "AM_N", "PAN": "AM_N", "PRI": "AM_N", "SPM": "AM_N",
    "TCA": "AM_N", "TTO": "AM_N", "USA": "AM_N", "VCT": "AM_N",
    "VIR": "AM_N",
    # South America (inc. Guianas)
    "ARG": "AM_S", "BOL": "AM_S", "BRA": "AM_S", "CHL": "AM_S",
    "COL": "AM_S", "ECU": "AM_S", "FLK": "AM_S", "GUF": "AM_S",
    "GUY": "AM_S", "PRY": "AM_S", "PER": "AM_S", "SUR": "AM_S",
    "URY": "AM_S", "VEN": "AM_S",
    # Oceania
    "AUS": "OC", "FJI": "OC", "KIR": "OC", "MHL": "OC", "FSM": "OC",
    "NCL": "OC", "NRU": "OC", "NZL": "OC", "PLW": "OC", "PNG": "OC",
    "WSM": "OC", "SLB": "OC", "TON": "OC", "TUV": "OC", "VUT": "OC",
}


def add_continent_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot continent dummies (6 groups) to df.

    Returns a copy of df with columns continent_AF, continent_AS, continent_EU,
    continent_AM_N, continent_AM_S, continent_OC added.  Countries not in
    CONTINENT_MAP get NaN for all dummies (they are dropped by downstream
    dropna anyway).
    """
    df = df.copy()
    df["_continent"] = df["iso3"].map(CONTINENT_MAP)
    dummies = pd.get_dummies(df["_continent"], prefix="continent").astype(float)
    # Re-index to ensure all 6 columns are always present even if a group is
    # absent in a sub-sample.
    all_cols = ["continent_AF", "continent_AM_N", "continent_AM_S",
                "continent_AS", "continent_EU", "continent_OC"]
    for col in all_cols:
        if col not in dummies.columns:
            dummies[col] = 0.0
    dummies = dummies[all_cols]
    return pd.concat([df.drop(columns=["_continent"]), dummies], axis=1)


# Five sequential sub-sample stages (each stage CUMULATES the previous drops).
SUBSAMPLE_STAGES = [
    ("baseline", []),
    ("drop_ajr_colonial", AJR_COLONIAL),
    ("drop_small_island", AJR_COLONIAL + SMALL_ISLAND),
    ("drop_americas_post_1492",
     AJR_COLONIAL + SMALL_ISLAND + AMERICAS_POST_1492),
    ("drop_ag_imputed",
     AJR_COLONIAL + SMALL_ISLAND + AMERICAS_POST_1492 + AG_HEAVILY_IMPUTED),
]
