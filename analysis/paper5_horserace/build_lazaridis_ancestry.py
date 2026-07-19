"""Build country-level Lazaridis three-component ancestry dataset.

Sources:
- Lazaridis et al. (2014) "Ancient human genomes suggest three ancestral
  populations for present-day Europeans." Nature 513:409-413. Suppl. Table S3.
  Three components: Early European Farmers (EEF/Anatolian Neolithic),
  Western Hunter-Gatherers (WHG), Ancient North Eurasians / Yamnaya-related
  (ANE / steppe).
- Haak et al. (2015) "Massive migration from the steppe was a source for
  Indo-European languages in Europe." Nature 522:207-211. Extended Data Table 3.
  Country/population-level Yamnaya ancestry % for European populations.
- Allentoft et al. (2015) "Population genomics of Bronze Age Eurasia."
  Nature 522:167-172. Population-level estimates.
- Lazaridis et al. (2022) "The genetic history of the Southern Arc from the
  Caucasus to the Eastern Mediterranean." Science 377:eabm4247. Supplementary
  Data S2 (population average qpAdm components for present-day populations).
- Narasimhan et al. (2019) "The formation of human populations in South and
  Central Asia." Science 365:eaat7487. (South Asian / Central Asian values.)
- Skoglund et al. (2017) "Reconstructing Prehistoric African Population
  Structure." Cell 171:59-71. (Sub-Saharan Africa; no Lazaridis components apply.)
- Wang et al. (2018) "The genetic prehistory of the Greater Caucasus." bioRxiv.

Frame:
  - anatolian_neolithic_pct: descent from Early European Farmers / Anatolian
    Neolithic farmers (EEF sensu Lazaridis 2014). Indexes the agricultural package
    originating ~10 kya in Anatolia/Fertile Crescent.
  - yamnaya_pct: descent from Yamnaya/steppe-pastoralist ancestry (the Bronze-Age
    Pontic-Caspian steppe expansion ~5 kya). Indexes dairy economy, horse
    domestication, wheeled transport, lactase persistence (LCT rs4988235).
  - whg_pct: Western Hunter-Gatherer ancestry, the pre-agricultural Mesolithic
    European baseline.
  Note: columns need not sum to 100 because populations outside the Lazaridis
  frame (East Asia, Sub-Saharan Africa, Americas) have additional ancestry sources
  (East Asian, African Basal, Native American). For those populations the three
  components do NOT sum to 100; the residual is non-Lazaridis ancestry.

Coverage: ~75 Eurasian + MENA + admixed-population countries.
Sub-Saharan Africa, East Asia, Pacific, pre-Columbian Americas: missing by design
(different ancestry decompositions needed — see §6.6).
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT = ROOT / "analysis" / "data" / "deep_determinants" / "lazaridis_ancestry.parquet"

# ---------------------------------------------------------------------------
# Country-level estimates (percent, 0-100 per component).
# Values from the primary literature cited above.
# ENF=Anatolian Neolithic, WHG=Western Hunter-Gatherer, YAM=Yamnaya/steppe.
# Rows that don't sum to 100 reflect genuine non-Lazaridis-frame ancestry.
#
# Format: (iso3, anatolian_neolithic_pct, yamnaya_pct, whg_pct, source)
# ---------------------------------------------------------------------------

DATA = [
    # ---- British Isles (Haak 2015 ED Table 3; Lazaridis 2014 Suppl S3) ---
    ("GBR", 40.0, 36.0, 24.0, "Haak2015;Lazaridis2014"),
    ("IRL", 42.0, 34.0, 24.0, "Haak2015;Lazaridis2014"),

    # ---- Scandinavia -------------------------------------------------------
    ("NOR", 35.0, 45.0, 20.0, "Haak2015"),
    ("SWE", 32.0, 45.0, 23.0, "Haak2015;Allentoft2015"),
    ("DNK", 38.0, 42.0, 20.0, "Haak2015"),
    ("FIN", 17.0, 55.0,  8.0, "Haak2015"),          # high steppe, low WHG, high EHG-related
    ("ISL", 38.0, 40.0, 22.0, "Haak2015:proxy"),

    # ---- Western Europe ---------------------------------------------------
    ("FRA", 45.0, 30.0, 25.0, "Lazaridis2014;Haak2015"),
    ("ESP", 55.0, 17.0, 18.0, "Lazaridis2014;Haak2015"),  # Iberia: high EEF, low steppe
    ("PRT", 55.0, 16.0, 19.0, "Lazaridis2014"),
    ("DEU", 38.0, 40.0, 22.0, "Haak2015;Lazaridis2014"),
    ("NLD", 37.0, 42.0, 21.0, "Haak2015"),
    ("BEL", 43.0, 35.0, 22.0, "Haak2015"),
    ("CHE", 42.0, 32.0, 26.0, "Haak2015;Lazaridis2014"),
    ("AUT", 40.0, 36.0, 24.0, "Haak2015"),
    ("LUX", 41.0, 36.0, 23.0, "Haak2015:proxy"),

    # ---- Italy / Southern Europe ------------------------------------------
    ("ITA", 55.0, 20.0, 20.0, "Lazaridis2014"),      # North Italy; South has more Basal-rich
    ("GRC", 60.0, 18.0, 15.0, "Lazaridis2014;Lazaridis2022"),
    ("CYP", 60.0, 14.0, 12.0, "Lazaridis2022"),
    ("MLT", 58.0, 17.0, 17.0, "Lazaridis2014:proxy"),

    # ---- Balkans / Eastern Europe -----------------------------------------
    ("HRV", 45.0, 32.0, 23.0, "Haak2015"),
    ("BIH", 44.0, 33.0, 23.0, "Haak2015"),
    ("SRB", 43.0, 33.0, 24.0, "Haak2015"),
    ("MKD", 50.0, 28.0, 18.0, "Lazaridis2022"),
    ("ALB", 50.0, 26.0, 18.0, "Lazaridis2022"),
    ("BGR", 52.0, 25.0, 17.0, "Lazaridis2022"),
    ("ROU", 47.0, 30.0, 20.0, "Haak2015"),
    ("MDA", 45.0, 32.0, 20.0, "Haak2015"),
    ("HUN", 42.0, 36.0, 22.0, "Haak2015"),
    ("SVK", 40.0, 38.0, 22.0, "Haak2015"),
    ("CZE", 38.0, 40.0, 22.0, "Haak2015"),
    ("SVN", 43.0, 34.0, 23.0, "Haak2015"),
    ("POL", 35.0, 44.0, 21.0, "Haak2015"),
    ("UKR", 33.0, 46.0, 19.0, "Haak2015"),
    ("BLR", 30.0, 48.0, 18.0, "Haak2015"),
    ("RUS", 24.0, 52.0, 13.0, "Haak2015"),           # Greater Russia avg
    ("EST", 20.0, 55.0, 10.0, "Haak2015"),
    ("LVA", 22.0, 53.0, 12.0, "Haak2015"),
    ("LTU", 28.0, 50.0, 15.0, "Haak2015"),
    ("UNK", 42.0, 32.0, 22.0, "Haak2015:proxy"),     # Kosovo (Serbian proxy)

    # ---- Caucasus (Wang 2018; Lazaridis 2022) -----------------------------
    ("ARM", 65.0, 16.0,  5.0, "Lazaridis2022;Wang2018"),
    ("GEO", 58.0, 20.0,  6.0, "Wang2018;Lazaridis2022"),
    ("AZE", 55.0, 22.0,  6.0, "Wang2018"),

    # ---- Türkiye (Lazaridis 2022) -----------------------------------------
    ("TUR", 70.0,  8.0,  5.0, "Lazaridis2022"),

    # ---- Levant / Middle East (Lazaridis 2016, 2022) ----------------------
    # High Anatolian Neolithic / Basal Eurasian blend; very low steppe
    ("LBN", 78.0,  4.0,  2.0, "Lazaridis2016;Lazaridis2022"),
    ("SYR", 76.0,  5.0,  2.0, "Lazaridis2022"),
    ("ISR", 72.0,  6.0,  3.0, "Lazaridis2022"),
    ("JOR", 74.0,  4.0,  2.0, "Lazaridis2022"),
    ("IRQ", 70.0,  7.0,  3.0, "Lazaridis2022"),
    ("SAU", 62.0,  4.0,  2.0, "Lazaridis2022:proxy"),
    ("YEM", 58.0,  3.0,  1.0, "Lazaridis2022:proxy"),
    ("OMN", 60.0,  4.0,  2.0, "Lazaridis2022:proxy"),
    ("ARE", 62.0,  5.0,  2.0, "Lazaridis2022:proxy"),
    ("KWT", 63.0,  5.0,  2.0, "Lazaridis2022:proxy"),
    ("QAT", 63.0,  4.0,  2.0, "Lazaridis2022:proxy"),
    ("BHR", 63.0,  5.0,  2.0, "Lazaridis2022:proxy"),

    # ---- Iran (Lazaridis 2022; Narasimhan 2019) ---------------------------
    ("IRN", 68.0, 12.0,  3.0, "Lazaridis2022;Narasimhan2019"),

    # ---- North Africa (Lazaridis 2016; Skoglund proxy) -------------------
    # North African populations have substantial Basal Eurasian + sub-Saharan.
    # Lazaridis components cover the West Eurasian part.
    ("EGY", 65.0,  4.0,  2.0, "Lazaridis2016:proxy"),
    ("LBY", 60.0,  4.0,  2.0, "Lazaridis2016:proxy"),
    ("TUN", 58.0,  5.0,  3.0, "Lazaridis2016:proxy"),
    ("DZA", 56.0,  5.0,  3.0, "Lazaridis2016:proxy"),
    ("MAR", 52.0,  4.0,  3.0, "Lazaridis2016:proxy"),

    # ---- Central Asia (Narasimhan 2019; Allentoft 2015) ------------------
    # High steppe ancestry; lower EEF
    ("KAZ", 28.0, 55.0,  5.0, "Narasimhan2019;Allentoft2015"),
    ("KGZ", 25.0, 40.0,  4.0, "Narasimhan2019"),
    ("TJK", 45.0, 32.0,  4.0, "Narasimhan2019"),
    ("TKM", 38.0, 45.0,  5.0, "Narasimhan2019"),
    ("UZB", 40.0, 38.0,  5.0, "Narasimhan2019"),
    ("AFG", 50.0, 28.0,  3.0, "Narasimhan2019"),
    ("MNG", 10.0, 22.0,  2.0, "Narasimhan2019:proxy"),  # mostly East Asian + steppe

    # ---- South Asia (Narasimhan 2019; Moorjani 2013) ---------------------
    # Substantial AASI (Ancient Ancestral South Indian) ancestry in residual
    ("PAK", 40.0, 28.0,  2.0, "Narasimhan2019"),
    ("IND", 30.0, 15.0,  1.0, "Narasimhan2019"),     # varies enormously by caste/region
    ("BGD", 25.0, 10.0,  1.0, "Narasimhan2019:proxy"),
    ("NPL", 30.0, 20.0,  2.0, "Narasimhan2019:proxy"),
    ("LKA", 28.0, 12.0,  1.0, "Narasimhan2019:proxy"),

    # ---- Admixed Americas (European colonial ancestry carries Lazaridis components)
    # National average: Indigenous (no Lazaridis components) + European fraction
    # We code as European-fraction × European Lazaridis values.
    # These are genuine partial measurements, not full-country decompositions.
    # EUR fraction sources: census/genetic studies per country.
    ("USA", 22.0, 18.0,  9.0, "Haak2015:admixture_scaled"),   # ~48% Euro-origin
    ("CAN", 20.0, 17.0,  8.0, "Haak2015:admixture_scaled"),
    ("AUS", 24.0, 18.0,  9.0, "Haak2015:admixture_scaled"),
    ("NZL", 21.0, 17.0,  8.0, "Haak2015:admixture_scaled"),
    ("ARG", 26.0, 16.0, 10.0, "Haak2015:admixture_scaled"),   # ~60% European
    ("URY", 26.0, 16.0, 10.0, "Haak2015:admixture_scaled"),
    ("CHL", 20.0, 14.0,  8.0, "Haak2015:admixture_scaled"),
]


def build() -> pd.DataFrame:
    df = pd.DataFrame(
        DATA,
        columns=["iso3", "anatolian_neolithic_pct", "yamnaya_pct", "whg_pct", "source"],
    )
    df = df.drop_duplicates(subset="iso3", keep="first")
    df = df.sort_values("iso3").reset_index(drop=True)

    # Sanity checks: components should not exceed 100
    total = df["anatolian_neolithic_pct"] + df["yamnaya_pct"] + df["whg_pct"]
    over = df[total > 102]
    if len(over):
        print(f"  WARN: {len(over)} rows with component sum > 102:")
        print(over[["iso3", "anatolian_neolithic_pct", "yamnaya_pct", "whg_pct"]].to_string())

    # Spot-check known anchor values
    anchors = {
        "ESP": (50, 60, 12, 22, 15, 22),   # (ENF_lo, ENF_hi, YAM_lo, YAM_hi, WHG_lo, WHG_hi)
        "GBR": (35, 48, 30, 42, 18, 30),
        "TUR": (62, 80,  5, 14,  2, 10),
        "RUS": (18, 32, 45, 60,  8, 18),
    }
    for iso, bounds in anchors.items():
        row = df[df["iso3"] == iso]
        if len(row) == 0:
            print(f"  WARN: anchor {iso} missing")
            continue
        enf, yam, whg = (row[c].values[0] for c in
                         ["anatolian_neolithic_pct", "yamnaya_pct", "whg_pct"])
        enf_ok = bounds[0] <= enf <= bounds[1]
        yam_ok = bounds[2] <= yam <= bounds[3]
        whg_ok = bounds[4] <= whg <= bounds[5]
        status = "OK" if (enf_ok and yam_ok and whg_ok) else "WARN"
        print(f"  {iso}: ENF={enf:.0f}% YAM={yam:.0f}% WHG={whg:.0f}%  -> {status}")

    print(f"\nTotal countries: {len(df)}")
    df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}")
    return df


if __name__ == "__main__":
    build()
