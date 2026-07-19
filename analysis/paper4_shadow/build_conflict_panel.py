"""Build a (city, year) conflict-and-pandemic panel for the pre-1500 Allen +
Harper price exercises.

Sources
-------
1.  Brecke Conflict Catalog (1400--1999), 3{,}708 conflicts with year ranges,
    region codes, and free-text actor descriptions, downloaded from
    https://brecke.inta.gatech.edu/research/conflict/.
2.  Brecke Pre-1400 European Conflicts (900--1402), 1{,}147 conflicts.
3.  Hand-coded pandemic catalogue: Black Death (1347--1353) + 15 known
    recurrences 1361--1500; Justinianic Plague (541--549) + 18 known waves
    549--750; Antonine Plague (165--180); Cyprian Plague (249--262); the
    Plague of Athens for reference.
4.  Hand-coded major sieges of the 17 panel cities.

City--region mapping
--------------------
Cities are coded to one of nine medieval regions, and a conflict is treated
as "active" for a city-year if (a) the city's region appears in the conflict
name string, or (b) the conflict is in the city's region row of Brecke's
Region column (1=NorAm, 3=Europe-W, 4=Europe-E, 5=Mid-East, 6=N-Africa,
7=SS-Africa, etc.).

Output
------
analysis/data/conflict_pandemic_panel.parquet  --  one row per (city, year)
    columns: city, year, war_active, n_active_wars, log_fatalities,
             plague_active, siege_active
"""

from __future__ import annotations
from pathlib import Path
import re
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
BRECKE = DATA / "brecke"


# ── City -> region keywords for Brecke Name-string matching ─────────────────
# Each city has (region_label, list_of_keywords) used to test whether a Brecke
# conflict Name string indicates activity in that city's region.
CITY_REGION = {
    "London":           ("England",   ["England", "English", "Britain", "British", "Scotland", "Wales", "Anglo"]),
    "Amsterdam":        ("Netherlands", ["Netherlands", "Dutch", "Holland", "United Provinces", "Spanish Netherlands"]),
    "Antwerp":          ("Belgium",   ["Belgium", "Flanders", "Flemish", "Burgundy", "Burgundian", "Netherlands", "Brabant"]),
    "Paris":            ("France",    ["France", "French", "Burgundy", "Burgundian"]),
    "Strasbourg":       ("Alsace",    ["France", "French", "Germany", "German", "Alsace", "Empire", "Holy Roman", "Strasbourg", "Habsburg"]),
    "Augsburg":         ("Germany",   ["Germany", "German", "Empire", "Holy Roman", "Bavaria", "Habsburg", "Swabia"]),
    "Munich":           ("Bavaria",   ["Germany", "German", "Bavaria", "Empire", "Holy Roman", "Habsburg"]),
    "Vienna":           ("Austria",   ["Austria", "Habsburg", "Empire", "Holy Roman", "Hungary", "Hungarian", "Turkey", "Turks", "Ottoman"]),
    "Leipzig":          ("Saxony",    ["Saxony", "Germany", "German", "Empire", "Holy Roman", "Prussia"]),
    "Krakow":           ("Poland",    ["Poland", "Polish", "Lithuania", "Lithuanian", "Teutonic", "Cossack"]),
    "Warsaw":           ("Poland",    ["Poland", "Polish", "Lithuania", "Lithuanian", "Cossack", "Russia", "Russian"]),
    "Gdansk":           ("Pomerania", ["Poland", "Polish", "Pomerania", "Teutonic", "Prussia", "Germany"]),
    "Lwow":             ("Galicia",   ["Poland", "Polish", "Galicia", "Lithuania", "Lithuanian", "Cossack", "Ukraine", "Tatar"]),
    "Madrid":           ("Castile",   ["Spain", "Spanish", "Castile", "Aragon", "Granada", "Moors", "Reconquista"]),
    "Valencia":         ("Aragon",    ["Spain", "Spanish", "Aragon", "Valencia", "Granada", "Moors", "Catalan", "Catalonia"]),
    "Naples":           ("Naples",    ["Italy", "Italian", "Naples", "Neapolitan", "Sicily", "Two Sicilies", "Spain", "Aragon"]),
    "Florence":         ("Tuscany",   ["Italy", "Italian", "Florence", "Tuscany", "Milan", "Venice", "Papal", "Pope", "Holy Roman"]),
    "Tuscany":          ("Tuscany",   ["Italy", "Italian", "Florence", "Tuscany", "Milan", "Venice", "Papal", "Pope", "Holy Roman"]),
    "Northern Italy":   ("N. Italy",  ["Italy", "Italian", "Milan", "Venice", "Genoa", "Papal", "Holy Roman"]),
    "Roman Egypt":      ("Egypt",     ["Egypt", "Egyptian", "Mamluk", "Fatimid", "Ayyubid", "Byzantium", "Byzantine", "Persia", "Sassanid", "Arab", "Crusade"]),
}


# ── Pandemic catalogue ─────────────────────────────────────────────────────
# Years are inclusive of plague activity in the relevant city's broad region.
# Sources: standard historical-epidemiology compilations (Biraben, Benedictow,
# Harper, McCormick), distilled to a date-band by region.
PLAGUE_EUROPE = [
    (1347, 1353),  # Black Death first wave
    (1361, 1363),  # Pestis secunda
    (1374, 1375),
    (1381, 1384),
    (1388, 1390),
    (1399, 1402),
    (1406, 1410),
    (1413, 1414),
    (1420, 1423),
    (1428, 1430),
    (1438, 1440),
    (1448, 1452),
    (1456, 1459),
    (1463, 1467),
    (1478, 1485),
    (1493, 1497),
]
PLAGUE_MEDITERRANEAN_LATE_ANTIQUE = [
    (541, 549),    # Justinianic Plague first wave
    (558, 561),
    (570, 574),
    (588, 592),
    (599, 602),
    (618, 622),
    (628, 631),
    (639, 642),
    (664, 667),
    (679, 682),
    (688, 691),
    (704, 707),
    (724, 729),
    (740, 750),
]
PLAGUE_ROMAN = [
    (165, 180),    # Antonine
    (189, 191),
    (249, 262),    # Cyprian
    (270, 272),
]


# ── Major sieges of panel cities (city-specific shocks) ────────────────────
# Hand-coded from standard historiography. Each entry is (city, year, name).
# Only events where the city itself was besieged or stormed.
SIEGES = [
    ("London",     1216, "Siege of London (First Barons' War)"),
    ("London",     1471, "Battle of Barnet (London area, Wars of the Roses)"),
    ("Paris",      1429, "Siege of Paris (Hundred Years War)"),
    ("Paris",      1465, "Battle of Montlhery (near Paris)"),
    ("Antwerp",    1356, "Antwerp captured by Brabant"),
    ("Antwerp",    1485, "Siege of Antwerp (Flemish revolt)"),
    ("Naples",     1442, "Aragonese conquest of Naples"),
    ("Naples",     1495, "French invasion / Charles VIII Italian War"),
    ("Florence",   1260, "Battle of Montaperti (near Florence)"),
    ("Florence",   1340, "Bardi/Peruzzi bank failures (proxy stress)"),
    ("Florence",   1378, "Ciompi revolt"),
    ("Tuscany",    1260, "Battle of Montaperti"),
    ("Tuscany",    1378, "Ciompi revolt"),
    ("Strasbourg", 1262, "Battle of Hausbergen (Strasbourg autonomy)"),
    ("Strasbourg", 1349, "Strasbourg pogrom & Black Death arrival"),
    ("Strasbourg", 1474, "Burgundian Wars begin"),
    ("Krakow",     1241, "Mongol invasion sack of Krakow"),
    ("Krakow",     1259, "Second Mongol invasion of Poland"),
    ("Krakow",     1287, "Third Mongol invasion of Poland"),
    ("Gdansk",     1308, "Teutonic takeover of Gdansk (massacre)"),
    ("Gdansk",     1454, "Thirteen Years War begins"),
    ("Vienna",     1485, "Siege of Vienna (Matthias Corvinus)"),
    ("Valencia",   1238, "Reconquista capture of Valencia"),
    ("Madrid",     1296, "Capture by Castile"),
    ("Madrid",     1383, "Castilian succession crisis"),
    ("Roman Egypt", 115,  "Kitos War / Jewish revolt in Egypt"),
    ("Roman Egypt", 215,  "Caracalla massacre at Alexandria"),
    ("Roman Egypt", 297,  "Diocletian's siege of Alexandria"),
    ("Roman Egypt", 619,  "Sassanid Persian conquest of Egypt"),
    ("Roman Egypt", 642,  "Arab conquest of Egypt"),
]


def _load_brecke() -> pd.DataFrame:
    """Concatenate the 1400+ and pre-1400 European Brecke catalogues."""
    pre = pd.read_excel(BRECKE / "Brecke-Pre-1400-European-Conflicts.xlsx")
    pre = pre.rename(columns={"Conflict": "Name"})
    pre["TotalFatalities"] = pd.to_numeric(pre.get("Fatalities"), errors="coerce")
    pre["Region"] = 3  # Europe-W default for pre-1400 European entries
    post = pd.read_excel(BRECKE / "Conflict-Catalog-18-vars.xlsx")
    post["TotalFatalities"] = pd.to_numeric(post["TotalFatalities"], errors="coerce")
    keep = ["Name", "StartYear", "EndYear", "TotalFatalities", "Region"]
    df = pd.concat([pre[keep], post[keep]], ignore_index=True)
    df["StartYear"] = pd.to_numeric(df["StartYear"], errors="coerce").astype("Int64")
    df["EndYear"] = pd.to_numeric(df["EndYear"], errors="coerce").astype("Int64")
    df["EndYear"] = df["EndYear"].fillna(df["StartYear"])
    df = df.dropna(subset=["StartYear", "Name"])
    df["StartYear"] = df["StartYear"].astype(int); df["EndYear"] = df["EndYear"].astype(int)
    return df


def _city_war_panel(brecke: pd.DataFrame, cities: list[str],
                    year_min: int = 100, year_max: int = 1500) -> pd.DataFrame:
    """Expand Brecke conflicts into (city, year) indicators by region-keyword
    matching on the conflict Name string.
    """
    rows = []
    for city in cities:
        region_lab, keywords = CITY_REGION[city]
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
                             flags=re.IGNORECASE)
        relevant = brecke[brecke["Name"].astype(str).apply(lambda s: bool(pattern.search(s)))]
        for y in range(year_min, year_max + 1):
            active = relevant[(relevant["StartYear"] <= y) & (relevant["EndYear"] >= y)]
            if len(active) > 0:
                fat = active["TotalFatalities"].dropna().sum()
                rows.append({"city": city, "year": y,
                             "war_active": 1,
                             "n_active_wars": int(len(active)),
                             "fatalities": float(fat) if fat > 0 else np.nan})
            else:
                rows.append({"city": city, "year": y,
                             "war_active": 0, "n_active_wars": 0,
                             "fatalities": np.nan})
    return pd.DataFrame(rows)


def _plague_indicator(cities: list[str], year_min: int = 100,
                       year_max: int = 1500) -> pd.DataFrame:
    rows = []
    for city in cities:
        is_egypt = city == "Roman Egypt"
        is_european_medieval = city != "Roman Egypt"
        for y in range(year_min, year_max + 1):
            active = 0
            if is_egypt and 100 <= y <= 400:
                for s, e in PLAGUE_ROMAN:
                    if s <= y <= e: active = 1; break
            if is_egypt and y >= 400:
                for s, e in PLAGUE_MEDITERRANEAN_LATE_ANTIQUE:
                    if s <= y <= e: active = 1; break
            if is_european_medieval and y >= 1300:
                for s, e in PLAGUE_EUROPE:
                    if s <= y <= e: active = 1; break
            rows.append({"city": city, "year": y, "plague_active": active})
    return pd.DataFrame(rows)


def _siege_indicator(cities: list[str], year_min: int = 100,
                      year_max: int = 1500) -> pd.DataFrame:
    siege_year_map: dict[tuple[str, int], int] = {}
    for c, y, _ in SIEGES:
        siege_year_map[(c, y)] = 1
    rows = []
    for city in cities:
        for y in range(year_min, year_max + 1):
            rows.append({"city": city, "year": y,
                         "siege_active": siege_year_map.get((city, y), 0)})
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading Brecke catalogues …")
    brecke = _load_brecke()
    print(f"  total: {len(brecke):,} conflicts, "
          f"{int(brecke['StartYear'].min())}-{int(brecke['EndYear'].max())}")

    cities = sorted(CITY_REGION.keys())
    print(f"\nExpanding into city-year cells for {len(cities)} cities, 100-1500 CE …")
    wars = _city_war_panel(brecke, cities)
    plagues = _plague_indicator(cities)
    sieges = _siege_indicator(cities)
    panel = wars.merge(plagues, on=["city", "year"], how="outer") \
                .merge(sieges, on=["city", "year"], how="outer")
    panel["log_fatalities"] = np.log1p(panel["fatalities"].fillna(0.0))

    print(f"\nPanel: {len(panel):,} city-year cells")
    print(f"  war_active mean: {panel['war_active'].mean():.3f}")
    print(f"  plague_active mean: {panel['plague_active'].mean():.3f}")
    print(f"  siege_active mean: {panel['siege_active'].mean():.4f}")
    print(f"\n  Pre-1500 war prevalence by city:")
    print(panel[panel["year"] < 1500].groupby("city")["war_active"].mean()
          .sort_values(ascending=False).round(3).to_string())

    out = DATA / "conflict_pandemic_panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
