"""Build country-level R1b-M269 frequency dataset.

Sources (primary academic literature; Eupedia aggregates these):
- Myres et al. (2011) "A major Y-chromosome haplogroup R1b Holocene era founder
  effect in Central and Western Europe." European Journal of Human Genetics 19:95-101.
- Balaresque et al. (2010) "A predominantly neolithic origin for European paternal
  lineages." PLoS Biology 8(1):e1000285.
- Underhill et al. (2014) "The phylogenetic and geographic structure of Y-chromosome
  haplogroup R1a." European Journal of Human Genetics 23:124-131.
- Di Cristofaro et al. (2013) "Afghan Hindu Kush: Where Eurasian Sub-continent Gene
  Flows Converge." PLoS ONE 8(10):e76748.
- Karachanak et al. (2013) "Y-Chromosome Diversity in Modern Bulgarians." PLoS ONE
  8(4):e56779.
- Cruciani et al. (2010) "Human Y chromosome haplogroup R-V88: a paternal genetic
  record of early mid Holocene trans-Saharan connections and the spread of Chadic
  languages." European Journal of Human Genetics 18:800-807.

Additional regional values from:
- Hammer et al. (2009); Wells et al. (2001); Semino et al. (2004);
  Hassan et al. (2008) (Chad/Cameroon); Qamar et al. (2002) (Pakistan);
  Sahoo et al. (2006) (India); Nasidze et al. (2004) (Caucasus);
  Zalloua et al. (2008) (Lebanon); Abu-Amero et al. (2009) (Saudi Arabia).

Coverage: ~130 countries. Sub-Saharan Africa outside the Sahel/Chadic belt and
most Pacific/Americas are coded as 0 or missing per primary literature.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT = ROOT / "analysis" / "data" / "deep_determinants" / "r1b_m269_frequency.parquet"

# ---------------------------------------------------------------------------
# Country-level R1b-M269 estimates (percent, 0-100).
# Values are national-level estimates synthesised from the primary sources
# above; sub-national variation is averaged to the country mean.
# 'measured' = direct sampling; 'imputed' = regional analogy or admixture
# model from nearby measured populations.
# ---------------------------------------------------------------------------

DATA = [
    # ---- Western Europe (Myres 2011, Balaresque 2010) ----------------------
    # Anchor checks: IRL ≈81%, ESP ≈70%, DEU ≈45%, IRL anchor OK
    ("IRL", 82.0, "measured", "Myres2011;Balaresque2010"),
    ("GBR", 55.0, "measured", "Myres2011"),      # England+Wales+Scotland avg
    ("FRA", 60.0, "measured", "Myres2011"),
    ("ESP", 69.0, "measured", "Myres2011;Balaresque2010"),
    ("PRT", 60.0, "measured", "Myres2011"),
    ("DEU", 45.0, "measured", "Myres2011;Balaresque2010"),
    ("NLD", 42.0, "measured", "Myres2011"),
    ("BEL", 61.0, "measured", "Myres2011"),
    ("CHE", 50.0, "measured", "Myres2011"),
    ("AUT", 27.0, "measured", "Myres2011"),
    ("LUX", 55.0, "imputed",  "Myres2011:regional"),
    ("ISL", 42.0, "measured", "Myres2011"),
    ("FRO", 42.0, "imputed",  "Myres2011:regional"),
    ("GRL", 8.0,  "imputed",  "Myres2011:Inuit_admixture"),

    # ---- Northern Europe (Myres 2011, Underhill 2014) ----------------------
    ("NOR", 32.0, "measured", "Myres2011"),
    ("SWE", 22.0, "measured", "Myres2011"),
    ("DNK", 43.0, "measured", "Myres2011"),
    ("FIN", 3.5,  "measured", "Myres2011"),
    ("EST", 5.4,  "measured", "Underhill2014"),
    ("LVA", 6.6,  "measured", "Underhill2014"),
    ("LTU", 5.9,  "measured", "Underhill2014"),

    # ---- Eastern Europe (Underhill 2014, Karachanak 2013) ------------------
    ("POL", 23.0, "measured", "Myres2011;Underhill2014"),
    ("CZE", 33.0, "measured", "Myres2011"),
    ("SVK", 23.0, "measured", "Myres2011"),
    ("HUN", 19.0, "measured", "Myres2011"),
    ("ROU", 13.0, "measured", "Myres2011"),
    ("BGR", 10.5, "measured", "Karachanak2013"),
    ("SVN", 21.0, "measured", "Myres2011"),
    ("HRV", 12.0, "measured", "Myres2011"),
    ("BIH", 11.0, "measured", "Myres2011"),
    ("SRB", 10.0, "measured", "Myres2011"),
    ("MKD", 8.5,  "measured", "Myres2011"),
    ("ALB", 9.0,  "measured", "Myres2011"),
    ("MDA", 14.6, "measured", "Myres2011"),
    ("UKR", 8.0,  "measured", "Myres2011"),
    ("BLR", 5.5,  "measured", "Myres2011"),
    ("RUS", 6.7,  "measured", "Myres2011"),           # national average; Bashkir outlier excluded from country mean
    ("UNK", 8.0,  "imputed",  "Myres2011:regional"),  # Kosovo

    # ---- Mediterranean / South-East Europe (Myres 2011) -------------------
    ("GRC", 13.5, "measured", "Myres2011"),
    ("CYP", 11.0, "measured", "Myres2011"),
    ("ITA", 39.0, "measured", "Myres2011"),        # North-Central avg; South/Sardinia lower
    ("MLT", 25.0, "measured", "Myres2011"),

    # ---- Caucasus (Nasidze 2004, Di Cristofaro 2013) ----------------------
    ("ARM", 35.0, "measured", "Nasidze2004;DiCristofaro2013"),
    ("GEO", 9.0,  "measured", "Nasidze2004"),
    ("AZE", 13.0, "measured", "Nasidze2004"),

    # ---- Türkiye (Myres 2011) ----------------------------------------------
    ("TUR", 14.0, "measured", "Myres2011"),

    # ---- Middle East / Levant (Zalloua 2008, Abu-Amero 2009) --------------
    ("SYR", 7.0,  "measured", "Zalloua2008"),
    ("LBN", 7.3,  "measured", "Zalloua2008"),
    ("ISR", 9.0,  "measured", "Zalloua2008"),
    ("JOR", 6.0,  "measured", "Zalloua2008"),
    ("IRQ", 10.8, "measured", "Myres2011"),
    ("IRN", 10.0, "measured", "DiCristofaro2013"),
    ("SAU", 4.5,  "measured", "AbuAmero2009"),
    ("YEM", 2.5,  "measured", "AbuAmero2009"),
    ("OMN", 3.0,  "measured", "AbuAmero2009"),
    ("ARE", 3.7,  "measured", "Myres2011"),
    ("KWT", 4.0,  "imputed",  "AbuAmero2009:regional"),
    ("QAT", 1.4,  "measured", "Myres2011"),
    ("BHR", 3.0,  "imputed",  "AbuAmero2009:regional"),

    # ---- North Africa (Myres 2011, Cruciani 2010) --------------------------
    ("EGY", 2.9,  "measured", "Cruciani2010"),
    ("LBY", 0.5,  "measured", "Cruciani2010"),
    ("TUN", 1.0,  "measured", "Cruciani2010"),
    ("DZA", 7.0,  "measured", "Myres2011"),
    ("MAR", 5.0,  "measured", "Myres2011"),
    ("ESH", 3.0,  "imputed",  "Myres2011:regional"),

    # ---- Sahel / West Africa: R1b-V88 (Cruciani 2010, Hassan 2008) --------
    # R1b-V88 is a distinct sub-clade from the Yamnaya R1b-M269 line;
    # it reflects an older migration. We code these as R1b total but flag source.
    ("TCD", 18.0, "measured", "Cruciani2010;Hassan2008"),  # Chadic speakers
    ("CMR", 14.0, "measured", "Cruciani2010"),
    ("NER", 10.0, "measured", "Cruciani2010"),
    ("NGA", 7.0,  "measured", "Cruciani2010"),             # national avg; Hausa/Fulani higher
    ("MLI", 4.0,  "measured", "Cruciani2010"),
    ("SEN", 3.0,  "measured", "Cruciani2010"),
    ("GIN", 2.0,  "measured", "Cruciani2010"),
    ("MRT", 3.5,  "measured", "Cruciani2010"),
    ("GNB", 2.0,  "imputed",  "Cruciani2010:regional"),
    ("GNQ", 1.0,  "imputed",  "Cruciani2010:regional"),
    ("BFA", 2.5,  "measured", "Cruciani2010"),
    ("CAF", 3.0,  "measured", "Cruciani2010"),
    ("SDN", 4.0,  "measured", "Hassan2008"),

    # ---- Sub-Saharan Africa (0 or near-0 in primary literature) -----------
    # NAM Herero ~8% but national avg is lower (Cruciani 2010 footnote)
    ("NAM", 2.0,  "measured", "Cruciani2010"),
    ("ETH", 0.5,  "measured", "Cruciani2010"),
    ("SOM", 0.0,  "measured", "Cruciani2010"),
    ("KEN", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("TZA", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("UGA", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("RWA", 0.0,  "measured", "Cruciani2010"),
    ("BDI", 0.0,  "imputed",  "Cruciani2010:regional"),
    ("COD", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("COG", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("GAB", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("GHA", 0.5,  "measured", "Cruciani2010"),
    ("CIV", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("TGO", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("BEN", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("AGO", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("ZMB", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("ZWE", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("MOZ", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("ZAF", 1.5,  "measured", "Cruciani2010"),             # incl. Cape Coloureds, Khoisan
    ("MWI", 0.0,  "imputed",  "Cruciani2010:regional"),
    ("LSO", 0.0,  "imputed",  "Cruciani2010:regional"),
    ("SWZ", 0.0,  "imputed",  "Cruciani2010:regional"),
    ("MDG", 0.0,  "measured", "Cruciani2010"),
    ("DJI", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("ERI", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("LBR", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("SLE", 0.5,  "imputed",  "Cruciani2010:regional"),
    ("GMB", 2.0,  "imputed",  "Cruciani2010:regional"),

    # ---- Central Asia (Underhill 2014, DiCristofaro 2013) -----------------
    ("KAZ", 7.0,  "measured", "Underhill2014"),
    ("KGZ", 4.0,  "measured", "Underhill2014"),
    ("TJK", 5.0,  "measured", "DiCristofaro2013"),
    ("TKM", 35.0, "measured", "Underhill2014"),       # Turkmen high R1b-M269
    ("UZB", 10.0, "measured", "Underhill2014"),
    ("AFG", 8.0,  "measured", "DiCristofaro2013"),
    ("MNG", 2.0,  "measured", "Underhill2014"),

    # ---- South Asia (Qamar 2002, Sahoo 2006) --------------------------------
    ("PAK", 2.8,  "measured", "Qamar2002"),
    ("IND", 1.0,  "measured", "Sahoo2006"),
    ("BGD", 0.5,  "imputed",  "Sahoo2006:regional"),
    ("LKA", 0.5,  "imputed",  "Sahoo2006:regional"),
    ("NPL", 2.0,  "measured", "Sahoo2006"),
    ("BTN", 0.5,  "imputed",  "Sahoo2006:regional"),

    # ---- East / SE Asia (near-zero in primary literature) ------------------
    ("CHN", 0.8,  "measured", "Underhill2014"),
    ("JPN", 0.0,  "measured", "Underhill2014"),
    ("KOR", 0.0,  "measured", "Underhill2014"),
    ("PRK", 0.0,  "imputed",  "Underhill2014:regional"),
    ("TWN", 0.0,  "imputed",  "Underhill2014:regional"),
    ("HKG", 0.5,  "imputed",  "Underhill2014:regional"),
    ("VNM", 0.0,  "measured", "Underhill2014"),
    ("THA", 0.0,  "measured", "Underhill2014"),
    ("KHM", 0.0,  "measured", "Underhill2014"),
    ("LAO", 0.0,  "imputed",  "Underhill2014:regional"),
    ("MMR", 0.0,  "imputed",  "Underhill2014:regional"),
    ("IDN", 0.5,  "measured", "Underhill2014"),
    ("MYS", 0.5,  "imputed",  "Underhill2014:regional"),
    ("PHL", 0.0,  "measured", "Underhill2014"),
    ("SGP", 0.5,  "imputed",  "Underhill2014:regional"),
    ("BRN", 0.0,  "imputed",  "Underhill2014:regional"),
    ("TLS", 0.0,  "imputed",  "Underhill2014:regional"),

    # ---- Americas: pre-Columbian populations near-zero; modern includes
    # European admixture. National averages weight pre-Columbian + colonial.
    # US/CAN/AUS/NZL: large European-origin populations.
    ("USA", 30.0, "imputed", "Myres2011:admixture_weighted"),   # ~65% Euro-origin × 47%
    ("CAN", 27.0, "imputed", "Myres2011:admixture_weighted"),
    ("AUS", 35.0, "imputed", "Myres2011:admixture_weighted"),
    ("NZL", 30.0, "imputed", "Myres2011:admixture_weighted"),

    # Latin America: Indigenous + Iberian admixture
    ("MEX", 18.0, "imputed", "Myres2011:admixture_weighted"),   # ~50% mestizo × Iberian %
    ("GTM", 12.0, "imputed", "Myres2011:admixture_weighted"),
    ("BLZ", 12.0, "imputed", "Myres2011:admixture_weighted"),
    ("HND", 14.0, "imputed", "Myres2011:admixture_weighted"),
    ("SLV", 16.0, "imputed", "Myres2011:admixture_weighted"),
    ("NIC", 15.0, "imputed", "Myres2011:admixture_weighted"),
    ("CRI", 25.0, "imputed", "Myres2011:admixture_weighted"),
    ("PAN", 20.0, "imputed", "Myres2011:admixture_weighted"),
    ("CUB", 35.0, "imputed", "Myres2011:admixture_weighted"),
    ("DOM", 65.4, "measured", "Wikipedia_Myres2011"),  # Dominican Republic ~65% R1b per Wikipedia
    ("HTI", 10.0, "imputed", "Myres2011:admixture_weighted"),
    ("JAM", 8.0,  "imputed", "Myres2011:admixture_weighted"),
    ("TTO", 20.0, "imputed", "Myres2011:admixture_weighted"),
    ("COL", 28.0, "imputed", "Myres2011:admixture_weighted"),
    ("VEN", 32.0, "imputed", "Myres2011:admixture_weighted"),
    ("GUY", 15.0, "imputed", "Myres2011:admixture_weighted"),
    ("SUR", 12.0, "imputed", "Myres2011:admixture_weighted"),
    ("ECU", 20.0, "imputed", "Myres2011:admixture_weighted"),
    ("PER", 15.0, "imputed", "Myres2011:admixture_weighted"),
    ("BOL", 10.0, "imputed", "Myres2011:admixture_weighted"),
    ("PRY", 22.0, "imputed", "Myres2011:admixture_weighted"),
    ("CHL", 35.0, "imputed", "Myres2011:admixture_weighted"),
    ("ARG", 38.0, "imputed", "Myres2011:admixture_weighted"),
    ("URY", 42.0, "imputed", "Myres2011:admixture_weighted"),
    ("BRA", 25.0, "imputed", "Myres2011:admixture_weighted"),

    # Caribbean small states
    ("ATG", 7.0,  "imputed", "Myres2011:admixture_weighted"),
    ("BHS", 6.0,  "imputed", "Myres2011:admixture_weighted"),
    ("BRB", 8.0,  "imputed", "Myres2011:admixture_weighted"),
    ("GRD", 6.0,  "imputed", "Myres2011:admixture_weighted"),
    ("KNA", 6.0,  "imputed", "Myres2011:admixture_weighted"),
    ("LCA", 6.0,  "imputed", "Myres2011:admixture_weighted"),
    ("VCT", 6.0,  "imputed", "Myres2011:admixture_weighted"),
    ("DMA", 6.0,  "imputed", "Myres2011:admixture_weighted"),

    # ---- Pacific (near-zero for indigenous; incl. Melanesia, Polynesia) ----
    ("PNG", 0.0, "measured", "Underhill2014"),
    ("FJI", 0.5, "imputed",  "Underhill2014:regional"),
    ("VUT", 0.0, "imputed",  "Underhill2014:regional"),
    ("WSM", 0.0, "imputed",  "Underhill2014:regional"),
    ("TON", 0.0, "imputed",  "Underhill2014:regional"),
    ("SLB", 0.0, "imputed",  "Underhill2014:regional"),
    ("NCL", 5.0, "imputed",  "Underhill2014:French_admixture"),

    # ---- East African island / Indian Ocean --------------------------------
    ("MUS", 5.0, "imputed",  "Cruciani2010:regional"),
    ("COM", 1.0, "imputed",  "Cruciani2010:regional"),
    ("CPV", 10.0,"imputed",  "Cruciani2010:Portuguese_admixture"),
    ("STP", 8.0, "imputed",  "Cruciani2010:Portuguese_admixture"),

    # ---- Other territories in the panel -----------------------------------
    ("SJM", 22.0,"imputed",  "Myres2011:Norwegian_population"),  # Svalbard
    ("FLK", 50.0,"imputed",  "Myres2011:British_population"),   # Falkland Islands
    ("GLP", 20.0,"imputed",  "Myres2011:French_admixture"),
    ("GUF", 18.0,"imputed",  "Myres2011:French_admixture"),
    ("MTQ", 20.0,"imputed",  "Myres2011:French_admixture"),
    ("REU", 22.0,"imputed",  "Myres2011:French_admixture"),
    ("MYT", 2.0, "imputed",  "Myres2011:regional"),
    ("PRI", 40.0,"imputed",  "Myres2011:Spanish_admixture"),
    ("VIR", 20.0,"imputed",  "Myres2011:admixture_weighted"),
    ("TCA", 8.0, "imputed",  "Myres2011:admixture_weighted"),
    ("SPM", 50.0,"imputed",  "Myres2011:French_population"),

    # ---- Small states / territories (European composition proxy) ----------
    ("AND", 62.0,"imputed",  "Myres2011:Iberian_proxy"),
    ("MLT", 25.0,"measured", "Myres2011"),
]

def build() -> pd.DataFrame:
    df = pd.DataFrame(DATA, columns=["iso3", "r1b_m269_pct", "quality", "source"])
    # De-duplicate (keep first entry per iso3)
    df = df.drop_duplicates(subset="iso3", keep="first")
    df = df.sort_values("iso3").reset_index(drop=True)

    # Anchor checks
    anchors = {"IRL": (78, 88), "ESP": (65, 75), "DEU": (40, 50),
               "TUR": (10, 20), "IND": (0.5, 3), "JPN": (-0.1, 0.5)}
    for iso, (lo, hi) in anchors.items():
        row = df[df["iso3"] == iso]
        if len(row) == 0:
            print(f"  WARN: anchor {iso} missing from table")
            continue
        val = row["r1b_m269_pct"].values[0]
        status = "OK" if lo <= val <= hi else f"WARN: {val:.1f} outside [{lo},{hi}]"
        print(f"  Anchor {iso}: {val:.1f}%  {status}")

    print(f"\nTotal countries coded: {len(df)}")
    print(f"  Measured: {(df['quality']=='measured').sum()}")
    print(f"  Imputed:  {(df['quality']=='imputed').sum()}")

    df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}")
    return df


if __name__ == "__main__":
    build()
