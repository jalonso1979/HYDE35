"""Build country-level functional-allele frequencies for the horserace.

Eight loci with established functional / economic-history channels, mapped to
country level via the Putterman-Weil 1500 ancestry matrix (same construction
logic as Ashraf-Galor's pdiv_aa).

Loci and primary sources:
  - LCT/MCM6 rs4988235 (lactase persistence) — Itan et al. 2010 PLoS Comp Biol;
    Liebert et al. 2017 Hum Genet
  - ADH1B rs1229984 (His47, fast alcohol metabolism) — Li et al. 2008 AJHG;
    Peng et al. 2010 BMC Evol Biol
  - AMY1 copy number (tag freq) — Perry et al. 2007 Nat Genet
  - EDAR rs3827760 (370A) — Sabeti et al. 2007 Nature; Kamberov et al. 2013 Cell
  - DARC/FY rs2814778 (Duffy-null) — Howes et al. 2011 PLoS Med
  - SLC24A5 rs1426654 (Ala111Thr) — Norton et al. 2007 Mol Biol Evol;
    Beleza et al. 2013 Mol Biol Evol
  - HBB rs334 (HbS sickle) — Piel et al. 2010 Lancet; Piel et al. 2013 Nature Comm
  - FADS1/2 rs174570 — Mathieson et al. 2015 Nature; Ameur et al. 2012 AJHG

Approach: 19 anchor populations with literature-derived per-locus derived-allele
frequencies; each PW-1500-source country is assigned to one anchor; each modern
country's frequency is the PW-weighted sum over its 1500 source compositions.

Output: analysis/data/deep_determinants/functional_alleles_pwadj.parquet
Columns: iso3, fa_lct, fa_adh1b, fa_amy1, fa_edar, fa_darc, fa_slc24a5, fa_hbb,
         fa_fads, source, n_pw_sources
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PW = ROOT / "analysis/data/deep_determinants/_raw/putterman_weil_2010/pw_migration_matrix_v1p1.xlsx"
OUT = ROOT / "analysis/data/deep_determinants/functional_alleles_pwadj.parquet"

# ---------------------------------------------------------------------------
# Anchor population × locus derived-allele frequency table.
# Values are literature-anchored point estimates with primary citations in the
# module docstring. Rows = 19 anchor populations covering 1500 source countries.
# Columns = 8 loci. Entries are derived-allele frequencies in [0, 1].
# ---------------------------------------------------------------------------

ANCHOR_FREQ = pd.DataFrame({
    "fa_lct":    {"NW_EUR": 0.75, "S_EUR": 0.30, "E_EUR": 0.50,
                  "ME_LEV": 0.20, "ARABIAN": 0.30, "N_AFR": 0.20,
                  "W_AFR": 0.05,  "C_AFR": 0.05,  "E_AFR_BANTU": 0.10,
                  "E_AFR_AAA": 0.30, "S_AFR": 0.05,
                  "C_ASIA": 0.40, "S_ASIA": 0.20,
                  "SE_ASIA": 0.05, "E_ASIA_HAN": 0.01, "E_ASIA_NE": 0.01,
                  "NATIVE_AMER": 0.05, "OCEANIAN": 0.05, "AUSTRALIAN": 0.01},
    "fa_adh1b":  {"NW_EUR": 0.05, "S_EUR": 0.05, "E_EUR": 0.10,
                  "ME_LEV": 0.10, "ARABIAN": 0.10, "N_AFR": 0.05,
                  "W_AFR": 0.02,  "C_AFR": 0.02,  "E_AFR_BANTU": 0.02,
                  "E_AFR_AAA": 0.02, "S_AFR": 0.02,
                  "C_ASIA": 0.40, "S_ASIA": 0.10,
                  "SE_ASIA": 0.60, "E_ASIA_HAN": 0.70, "E_ASIA_NE": 0.65,
                  "NATIVE_AMER": 0.10, "OCEANIAN": 0.05, "AUSTRALIAN": 0.05},
    "fa_amy1":   {"NW_EUR": 0.70, "S_EUR": 0.70, "E_EUR": 0.65,
                  "ME_LEV": 0.70, "ARABIAN": 0.55, "N_AFR": 0.60,
                  "W_AFR": 0.35,  "C_AFR": 0.35,  "E_AFR_BANTU": 0.40,
                  "E_AFR_AAA": 0.50, "S_AFR": 0.20,
                  "C_ASIA": 0.55, "S_ASIA": 0.70,
                  "SE_ASIA": 0.70, "E_ASIA_HAN": 0.80, "E_ASIA_NE": 0.75,
                  "NATIVE_AMER": 0.40, "OCEANIAN": 0.35, "AUSTRALIAN": 0.20},
    "fa_edar":   {"NW_EUR": 0.01, "S_EUR": 0.01, "E_EUR": 0.01,
                  "ME_LEV": 0.01, "ARABIAN": 0.01, "N_AFR": 0.01,
                  "W_AFR": 0.02,  "C_AFR": 0.02,  "E_AFR_BANTU": 0.02,
                  "E_AFR_AAA": 0.02, "S_AFR": 0.02,
                  "C_ASIA": 0.40, "S_ASIA": 0.05,
                  "SE_ASIA": 0.70, "E_ASIA_HAN": 0.93, "E_ASIA_NE": 0.85,
                  "NATIVE_AMER": 0.85, "OCEANIAN": 0.30, "AUSTRALIAN": 0.30},
    "fa_darc":   {"NW_EUR": 0.01, "S_EUR": 0.01, "E_EUR": 0.01,
                  "ME_LEV": 0.05, "ARABIAN": 0.10, "N_AFR": 0.40,
                  "W_AFR": 0.95,  "C_AFR": 0.95,  "E_AFR_BANTU": 0.95,
                  "E_AFR_AAA": 0.85, "S_AFR": 0.90,
                  "C_ASIA": 0.05, "S_ASIA": 0.05,
                  "SE_ASIA": 0.05, "E_ASIA_HAN": 0.01, "E_ASIA_NE": 0.01,
                  "NATIVE_AMER": 0.02, "OCEANIAN": 0.02, "AUSTRALIAN": 0.02},
    "fa_slc24a5":{"NW_EUR": 1.00, "S_EUR": 1.00, "E_EUR": 1.00,
                  "ME_LEV": 0.95, "ARABIAN": 0.85, "N_AFR": 0.70,
                  "W_AFR": 0.01,  "C_AFR": 0.01,  "E_AFR_BANTU": 0.05,
                  "E_AFR_AAA": 0.50, "S_AFR": 0.10,
                  "C_ASIA": 0.95, "S_ASIA": 0.85,
                  "SE_ASIA": 0.05, "E_ASIA_HAN": 0.05, "E_ASIA_NE": 0.05,
                  "NATIVE_AMER": 0.05, "OCEANIAN": 0.05, "AUSTRALIAN": 0.05},
    "fa_hbb":    {"NW_EUR": 0.00, "S_EUR": 0.02, "E_EUR": 0.00,
                  "ME_LEV": 0.05, "ARABIAN": 0.10, "N_AFR": 0.05,
                  "W_AFR": 0.15,  "C_AFR": 0.15,  "E_AFR_BANTU": 0.10,
                  "E_AFR_AAA": 0.05, "S_AFR": 0.05,
                  "C_ASIA": 0.01, "S_ASIA": 0.05,
                  "SE_ASIA": 0.01, "E_ASIA_HAN": 0.00, "E_ASIA_NE": 0.00,
                  "NATIVE_AMER": 0.00, "OCEANIAN": 0.00, "AUSTRALIAN": 0.00},
    "fa_fads":   {"NW_EUR": 0.65, "S_EUR": 0.50, "E_EUR": 0.55,
                  "ME_LEV": 0.40, "ARABIAN": 0.30, "N_AFR": 0.30,
                  "W_AFR": 0.05,  "C_AFR": 0.05,  "E_AFR_BANTU": 0.10,
                  "E_AFR_AAA": 0.20, "S_AFR": 0.05,
                  "C_ASIA": 0.50, "S_ASIA": 0.45,
                  "SE_ASIA": 0.50, "E_ASIA_HAN": 0.65, "E_ASIA_NE": 0.65,
                  "NATIVE_AMER": 0.40, "OCEANIAN": 0.30, "AUSTRALIAN": 0.30},
})

# ---------------------------------------------------------------------------
# Mapping from PW source country (1500 populations) to anchor population.
# Source codes are lowercase 3-letter ISO3 as in the PW spreadsheet.
# In 1500, the Americas, Australia, and Oceania were populated by Indigenous
# populations; settler-majority modern populations enter via the post-1500
# Putterman-Weil migration matrix, not via the 1500 source assignment.
# ---------------------------------------------------------------------------

PW_TO_ANCHOR = {
    # NW Europe (1500 populations broadly consistent with modern composition)
    "gbr": "NW_EUR", "fra": "NW_EUR", "deu": "NW_EUR", "bel": "NW_EUR",
    "nld": "NW_EUR", "dnk": "NW_EUR", "irl": "NW_EUR", "isl": "NW_EUR",
    "lux": "NW_EUR", "che": "NW_EUR", "aut": "NW_EUR",
    "fin": "NW_EUR", "nor": "NW_EUR", "swe": "NW_EUR",
    # S Europe
    "esp": "S_EUR", "prt": "S_EUR", "ita": "S_EUR", "grc": "S_EUR",
    "mlt": "S_EUR", "cyp": "S_EUR",
    # E Europe / Balkans / Baltics
    "pol": "E_EUR", "cze": "E_EUR", "svk": "E_EUR", "hun": "E_EUR",
    "rom": "E_EUR", "bgr": "E_EUR", "ukr": "E_EUR", "rus": "E_EUR",
    "blr": "E_EUR", "ltu": "E_EUR", "lva": "E_EUR", "est": "E_EUR",
    "mda": "E_EUR", "hrv": "E_EUR", "svn": "E_EUR", "bih": "E_EUR",
    "mkd": "E_EUR", "yug": "E_EUR", "alb": "E_EUR",
    # Middle East / Levant
    "lbn": "ME_LEV", "syr": "ME_LEV", "jor": "ME_LEV", "isr": "ME_LEV",
    "irq": "ME_LEV", "tur": "ME_LEV", "irn": "ME_LEV", "arm": "ME_LEV",
    "geo": "ME_LEV", "aze": "ME_LEV",
    # Arabian peninsula
    "sau": "ARABIAN", "yem": "ARABIAN", "omn": "ARABIAN", "are": "ARABIAN",
    "kwt": "ARABIAN", "bhr": "ARABIAN", "qat": "ARABIAN",
    # North Africa
    "dza": "N_AFR", "egy": "N_AFR", "lby": "N_AFR", "mar": "N_AFR",
    "tun": "N_AFR", "mrt": "N_AFR",
    # West Africa
    "nga": "W_AFR", "gha": "W_AFR", "civ": "W_AFR", "sen": "W_AFR",
    "mli": "W_AFR", "bfa": "W_AFR", "gin": "W_AFR", "gnb": "W_AFR",
    "gmb": "W_AFR", "lbr": "W_AFR", "sle": "W_AFR", "tgo": "W_AFR",
    "ben": "W_AFR", "cpv": "W_AFR", "ner": "W_AFR",
    # Central Africa
    "zar": "C_AFR", "cmr": "C_AFR", "gab": "C_AFR", "cog": "C_AFR",
    "caf": "C_AFR", "tcd": "C_AFR", "gnq": "C_AFR", "stp": "C_AFR",
    # East Africa - Bantu / Nilotic
    "tza": "E_AFR_BANTU", "ken": "E_AFR_BANTU", "uga": "E_AFR_BANTU",
    "rwa": "E_AFR_BANTU", "bdi": "E_AFR_BANTU",
    # East Africa - Afro-Asiatic
    "eth": "E_AFR_AAA", "eri": "E_AFR_AAA", "som": "E_AFR_AAA",
    "sdn": "E_AFR_AAA",
    # Southern Africa
    "zaf": "S_AFR", "nam": "S_AFR", "bwa": "S_AFR", "zwe": "S_AFR",
    "moz": "S_AFR", "mwi": "S_AFR", "zmb": "S_AFR", "ago": "S_AFR",
    "lso": "S_AFR", "swz": "S_AFR", "mdg": "S_AFR", "mus": "S_AFR",
    "com": "S_AFR",
    # Central Asia (Turkic / Iranic steppe)
    "kaz": "C_ASIA", "uzb": "C_ASIA", "kgz": "C_ASIA", "tkm": "C_ASIA",
    "tjk": "C_ASIA", "afg": "C_ASIA",
    # South Asia
    "ind": "S_ASIA", "pak": "S_ASIA", "bgd": "S_ASIA", "lka": "S_ASIA",
    "npl": "S_ASIA", "btn": "S_ASIA",
    # Southeast Asia
    "vnm": "SE_ASIA", "tha": "SE_ASIA", "khm": "SE_ASIA", "lao": "SE_ASIA",
    "mmr": "SE_ASIA", "mys": "SE_ASIA", "idn": "SE_ASIA", "phl": "SE_ASIA",
    "sgp": "SE_ASIA", "hkg": "SE_ASIA",
    # East Asia
    "chn": "E_ASIA_HAN",
    "jpn": "E_ASIA_NE", "kor": "E_ASIA_NE", "mng": "E_ASIA_NE",
    "prk": "E_ASIA_NE",
    # Americas (1500 = Indigenous populations)
    "mex": "NATIVE_AMER", "gtm": "NATIVE_AMER", "slv": "NATIVE_AMER",
    "hnd": "NATIVE_AMER", "nic": "NATIVE_AMER", "cri": "NATIVE_AMER",
    "pan": "NATIVE_AMER", "cub": "NATIVE_AMER", "dom": "NATIVE_AMER",
    "hti": "NATIVE_AMER", "jam": "NATIVE_AMER", "blz": "NATIVE_AMER",
    "col": "NATIVE_AMER", "ven": "NATIVE_AMER", "ecu": "NATIVE_AMER",
    "per": "NATIVE_AMER", "bol": "NATIVE_AMER", "pry": "NATIVE_AMER",
    "ury": "NATIVE_AMER", "arg": "NATIVE_AMER", "chl": "NATIVE_AMER",
    "bra": "NATIVE_AMER", "usa": "NATIVE_AMER", "can": "NATIVE_AMER",
    "pri": "NATIVE_AMER", "tto": "NATIVE_AMER", "vct": "NATIVE_AMER",
    "guy": "NATIVE_AMER", "cil": "NATIVE_AMER",
    # Oceania (1500 = Austronesian / Papuan / Aboriginal)
    "png": "OCEANIAN", "tmp": "OCEANIAN",
    "fji": "OCEANIAN", "ton": "OCEANIAN", "wsm": "OCEANIAN",
    "niu": "OCEANIAN", "oan": "OCEANIAN",
    "aus": "AUSTRALIAN", "nzl": "AUSTRALIAN",  # NZ 1500 = Maori Polynesian — use Oceanian
}
# NZ correction: should be Oceanian Polynesian, not Aboriginal Australian
PW_TO_ANCHOR["nzl"] = "OCEANIAN"


def _build_country_frequencies(pw_matrix: pd.DataFrame) -> pd.DataFrame:
    """For each modern country (row of pw_matrix), compute PW-weighted
    allele frequency across the 8 loci.
    """
    pw = pw_matrix.copy()
    modern = pw[["wbcode", "wbname"]].copy()
    modern["wbcode"] = modern["wbcode"].astype(str).str.upper().str.strip()
    src_cols = [c for c in pw.columns if c not in ("wbcode", "wbname", "update")]

    # Skip any source columns we don't have an anchor for; track them
    unmapped = [c for c in src_cols if c not in PW_TO_ANCHOR]
    if unmapped:
        print(f"  WARNING: unmapped PW source columns (will be treated as 0 weight): {unmapped}")
        src_cols = [c for c in src_cols if c in PW_TO_ANCHOR]

    # For each modern country, count which PW sources had positive weight
    weights = pw[src_cols].fillna(0).values
    n_pw_sources = (weights > 1e-4).sum(axis=1)

    # Compute PW-weighted allele frequency for each locus
    out = {"iso3": modern["wbcode"].values, "n_pw_sources": n_pw_sources}
    for locus in ANCHOR_FREQ.columns:
        anchor_freq_per_src = np.array(
            [ANCHOR_FREQ.loc[PW_TO_ANCHOR[c], locus] for c in src_cols]
        )
        # Weighted sum across sources for each modern country
        country_freq = weights @ anchor_freq_per_src
        # Normalise by total weight (in case PW rows don't sum exactly to 1)
        wsum = weights.sum(axis=1)
        wsum[wsum == 0] = np.nan
        out[locus] = country_freq / wsum

    df = pd.DataFrame(out)
    return df


# ISO3 normalisation: PW uses some non-canonical codes
PW_TO_ISO3 = {
    "ZAR": "COD",  # DRC
    "ROM": "ROU",  # Romania
    "TMP": "TLS",  # Timor-Leste
    "WBG": "PSE",
    "ADO": "AND",
    "YUG": "SRB",  # PW Yugoslavia → Serbia (largest successor)
}


def main() -> None:
    if not PW.exists():
        raise FileNotFoundError(f"PW migration matrix missing at {PW}")

    pw = pd.read_excel(PW)
    print(f"PW matrix: {pw.shape[0]} modern countries × {pw.shape[1]-2} source codes")

    df = _build_country_frequencies(pw)
    df["iso3"] = df["iso3"].replace(PW_TO_ISO3)

    # Filter to valid ISO3
    df = df[df["iso3"].str.match(r"^[A-Z]{3}$")].copy()
    df = df.drop_duplicates(subset=["iso3"]).reset_index(drop=True)

    df["source"] = (
        "PW2010 ancestry-weighted from 19-anchor literature frequencies; "
        "loci: LCT (Itan10, Liebert17), ADH1B (Li08, Peng10), AMY1 (Perry07), "
        "EDAR (Sabeti07, Kamberov13), DARC (Howes11), SLC24A5 (Norton07, Beleza13), "
        "HBB (Piel10), FADS (Mathieson15, Ameur12)."
    )

    df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(f"  N countries: {len(df)}")
    print(f"  Mean weight in PW (n_pw_sources): {df['n_pw_sources'].mean():.1f}")
    print(f"\nAllele-frequency summary (PW-weighted):")
    for locus in ANCHOR_FREQ.columns:
        s = df[locus]
        print(f"  {locus}: mean={s.mean():.3f}, range=[{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    main()
