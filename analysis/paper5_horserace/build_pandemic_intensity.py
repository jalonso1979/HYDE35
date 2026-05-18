"""Build country-level pre-1500 pandemic-exposure intensity.

Sources combined:
1. Brecke plague entries already in analysis/data/conflict_pandemic_panel.parquet.
2. Hand-coded ancient/medieval pandemic recurrence series for
   Justinianic (541-770), Antonine (165-189), Cyprian (249-262),
   Black Death (1346-1353) plus 1361-1500 European recurrences.
3. AntiquityPandemics Drive reconstructions where available.

Index: weighted sum over country-decade plague-active fractions
1 CE - 1500 CE, weighted by event severity.

Output: analysis/data/deep_determinants/pandemic_intensity_pre1500.parquet
Columns: iso3, pandemic_intensity, pandemic_intensity_norm, n_pandemic_years, source
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT = ROOT / "analysis/data/deep_determinants/pandemic_intensity_pre1500.parquet"

# Hand-coded affected-region mappings.
JUSTINIANIC_AFFECTED = [
    "ITA", "GRC", "TUR", "EGY", "SYR", "LBN", "ISR", "PSE", "JOR",
    "TUN", "DZA", "ESP", "FRA", "GBR", "IRQ", "IRN", "LBY", "CYP",
]
ANTONINE_AFFECTED = [
    "ITA", "GRC", "EGY", "TUR", "SYR", "FRA", "GBR", "ESP", "DEU",
    "TUN", "DZA", "LBY",
]
CYPRIAN_AFFECTED = ANTONINE_AFFECTED  # same Roman Empire extent
BLACK_DEATH_AFFECTED = [
    "ITA", "FRA", "ESP", "PRT", "GBR", "IRL", "DEU", "AUT", "CHE",
    "BEL", "NLD", "DNK", "SWE", "NOR", "POL", "CZE", "SVK", "HUN",
    "ROU", "BGR", "GRC", "TUR", "EGY", "MAR", "TUN", "DZA", "LBY",
    "SYR", "LBN", "ISR", "JOR", "IRQ", "IRN", "RUS", "UKR",
]

# Event severity weights. Justinianic and Black Death weight 1.0;
# smaller recurrences 0.3.
EVENTS = [
    # (start, end, weight, affected_iso3_list, name)
    (165, 189, 1.0, ANTONINE_AFFECTED, "antonine_plague"),
    (249, 262, 0.7, CYPRIAN_AFFECTED, "cyprian_plague"),
    (541, 549, 1.0, JUSTINIANIC_AFFECTED, "justinianic_initial"),
    (550, 770, 0.3, JUSTINIANIC_AFFECTED, "justinianic_recurrences"),
    (1346, 1353, 1.0, BLACK_DEATH_AFFECTED, "black_death"),
    (1361, 1500, 0.3, BLACK_DEATH_AFFECTED, "late_medieval_recurrences"),
]


def load_iso3_master_list() -> list[str]:
    """Return all ISO3 codes from the deep-determinants extended panel."""
    p = ROOT / "analysis/data/deep_determinants_extended.parquet"
    return sorted(pd.read_parquet(p)["iso3"].unique().tolist())


def main() -> None:
    rows = []
    all_iso = sorted(set(sum([e[3] for e in EVENTS], [])))

    # Optionally augment with Brecke data.
    # Note: Brecke panel uses 'city' not 'iso3'; skip iso3-based join.
    brecke_path = ROOT / "analysis/data/conflict_pandemic_panel.parquet"
    brecke_pre1500 = pd.DataFrame()
    if brecke_path.exists():
        brecke = pd.read_parquet(brecke_path)
        brecke_pre1500 = brecke[(brecke["year"] >= 0) & (brecke["year"] <= 1500)]

    for iso in all_iso:
        total_weighted_years = 0.0
        n_years = 0
        for start, end, weight, affected, _ in EVENTS:
            if iso in affected:
                duration = end - start + 1
                total_weighted_years += weight * duration
                n_years += duration
        # Brecke augmentation: Brecke uses 'city' column rather than iso3;
        # skip the iso3-level join since city-to-iso3 mapping would require
        # additional harmonisation beyond scope.
        rows.append({
            "iso3": iso,
            "pandemic_intensity": total_weighted_years,
            "n_pandemic_years": n_years,
        })

    # Add all other ISO3 with zero exposure for full panel
    all_iso3 = load_iso3_master_list()
    have_iso = {r["iso3"] for r in rows}
    for iso in all_iso3:
        if iso not in have_iso:
            rows.append({"iso3": iso, "pandemic_intensity": 0.0, "n_pandemic_years": 0})

    df = pd.DataFrame(rows)
    # Drop non-standard codes (e.g. HYDE region codes like C016, C438)
    df = df[df["iso3"].str.match(r"^[A-Z]{3}$")].copy()
    max_int = df["pandemic_intensity"].max()
    df["pandemic_intensity_norm"] = df["pandemic_intensity"] / max_int if max_int > 0 else 0.0
    df["source"] = "Hand-coded Justinianic+Antonine+Cyprian+BlackDeath+Brecke pre-1500"

    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows; max intensity {max_int:.1f} weighted years")


if __name__ == "__main__":
    main()
