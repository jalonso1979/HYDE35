"""Build Ashraf-Galor predicted heterozygosity, ancestry-adjusted.

Source: Ashraf & Galor (2013, AER) "The 'Out of Africa' Hypothesis,
Human Genetic Diversity, and Comparative Economic Development",
replication archive `country.dta` placed at
`analysis/data/deep_determinants/_raw/ashraf_galor_2013/country.dta`.

The AG country file uses 1000-km units for migratory distance; we convert
to km. The ancestry-adjusted predicted diversity `pdiv_aa` reweights the
predicted diversity by the year-1500 ancestral composition of each
country's year-2000 population (Putterman-Weil 2010 World Migration
Matrix), and is the column the AG 2013 paper uses for development
regressions on post-1500 outcomes.

Output: analysis/data/deep_determinants/predicted_het_pw_adjusted.parquet
Columns: iso3, H_pred, H_pred_pwadj, mdist_addis, mdist_addis_pwadj, source
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
RAW = ROOT / "analysis/data/deep_determinants/_raw/ashraf_galor_2013/country.dta"
OUT = ROOT / "analysis/data/deep_determinants/predicted_het_pw_adjusted.parquet"

# Non-standard codes in AG country.dta → ISO3 canonical
AG_TO_ISO3 = {
    "ROM": "ROU",  # Romania
    "TMP": "TLS",  # Timor-Leste
    "ZAR": "COD",  # DRC
    "ADO": "AND",  # Andorra (AG uses World Bank pre-2013 code ADO)
    "WBG": "PSE",  # West Bank and Gaza → Palestine
}


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"AG 2013 country.dta missing at {RAW}. "
            "Download from the user's Drive 'TheOutofAfrica/data/' folder."
        )

    df = pd.read_stata(str(RAW))
    df = df[["code", "pdiv", "pdiv_aa", "mdist_addis", "mdist_addis_aa"]].copy()

    df = df.rename(columns={
        "code": "iso3",
        "pdiv": "H_pred",
        "pdiv_aa": "H_pred_pwadj",
        "mdist_addis_aa": "mdist_addis_pwadj",
    })

    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df["iso3"] = df["iso3"].replace(AG_TO_ISO3)

    # AG stores migratory distance in 1000-km units; convert to km
    df["mdist_addis"] = df["mdist_addis"] * 1000.0
    df["mdist_addis_pwadj"] = df["mdist_addis_pwadj"] * 1000.0

    # Drop rows missing the headline pdiv
    df = df.dropna(subset=["H_pred"])

    # Where ancestry-adjusted is missing (Old-World countries with no
    # post-1500 migration), AG sets pdiv_aa equal to pdiv. We do the same.
    df["H_pred_pwadj"] = df["H_pred_pwadj"].fillna(df["H_pred"])
    df["mdist_addis_pwadj"] = df["mdist_addis_pwadj"].fillna(df["mdist_addis"])

    # Filter to valid ISO3 only
    df = df[df["iso3"].str.match(r"^[A-Z]{3}$")].copy()
    df = df.drop_duplicates(subset=["iso3"])

    df["source"] = "Ashraf-Galor 2013 AER replication archive (country.dta)"

    df = df[["iso3", "H_pred", "H_pred_pwadj",
             "mdist_addis", "mdist_addis_pwadj", "source"]]
    df.to_parquet(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"  Rows: {len(df)}")
    print(f"  H_pred range:       [{df['H_pred'].min():.4f}, {df['H_pred'].max():.4f}]")
    print(f"  H_pred_pwadj range: [{df['H_pred_pwadj'].min():.4f}, {df['H_pred_pwadj'].max():.4f}]")
    moved = (df["H_pred_pwadj"] - df["H_pred"]).abs() > 0.005
    print(f"  Countries with non-trivial ancestry adjustment: {moved.sum()}")


if __name__ == "__main__":
    main()
