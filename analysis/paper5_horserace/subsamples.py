# analysis/paper5_horserace/subsamples.py
"""Hand-coded sub-sample drop lists for the robustness battery."""

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
