"""Add war and pandemic controls to the §3.2 pre-industrial Malthusian regression.

Defensive robustness exercise. The headline Malthusian regression of population
growth on lagged density and climate forcing (Equation 5 in the paper) treats
wars and pandemics as residual noise inside the country fixed effect. We
verify the climate coefficients are robust to explicit war/pandemic controls.

Country-year war indicator: for each Brecke conflict 900-1950 we extract the
country names mentioned in the Name actor string and produce a binary
war_active_{i,t} per (iso3, year). Country-year plague indicator: hand-coded
broad-regional plague waves (Black Death + recurrences for Europe, Justinianic
+ recurrences for Mediterranean, Antonine/Cyprian for the Roman period, with
country-region membership inferred from continent and approximate latitude).

The Malthus regression intervals are HYDE-decadal in 1500-1950 and HYDE-century
1421-1500. For each interval we compute the fraction of years with active
war and the fraction with active plague, then add these to the right-hand
side of Equation 5.

Outputs:
    analysis/data/malthus_conflict_controls.parquet  -- the merged panel
    analysis/data/malthus_conflict_results.parquet   -- before/after comparison
"""

from __future__ import annotations
from pathlib import Path
import re
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
BRECKE = DATA / "brecke"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def _country_keywords() -> dict[str, list[str]]:
    """Map iso3 -> list of search keywords (English country name + common variants
    + the obvious historical names) for matching Brecke actor strings."""
    iso = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso = iso.dropna(subset=["iso3", "name"]).copy()
    extras = {  # historical and adjective forms not in pycountry
        "GBR": ["England", "English", "Britain", "British", "Scotland", "Wales"],
        "FRA": ["France", "French", "Burgundy", "Burgundian", "Frankish"],
        "DEU": ["Germany", "German", "Prussia", "Bavaria", "Saxony", "Holy Roman"],
        "ITA": ["Italy", "Italian", "Naples", "Florence", "Venice", "Milan", "Tuscany", "Papal"],
        "ESP": ["Spain", "Spanish", "Castile", "Aragon", "Granada"],
        "PRT": ["Portugal", "Portuguese"],
        "NLD": ["Netherlands", "Dutch", "Holland", "United Provinces"],
        "BEL": ["Belgium", "Flanders", "Flemish", "Brabant"],
        "POL": ["Poland", "Polish", "Lithuania", "Lithuanian"],
        "RUS": ["Russia", "Russian", "Muscovy", "Soviet"],
        "TUR": ["Turkey", "Turkish", "Ottoman", "Ottomans"],
        "AUT": ["Austria", "Austrian", "Habsburg"],
        "HUN": ["Hungary", "Hungarian", "Magyar"],
        "CZE": ["Bohemia", "Czech", "Czechoslovakia"],
        "SWE": ["Sweden", "Swedish"],
        "NOR": ["Norway", "Norwegian"],
        "DNK": ["Denmark", "Danish"],
        "FIN": ["Finland", "Finnish"],
        "GRC": ["Greece", "Greek", "Byzantium", "Byzantine"],
        "EGY": ["Egypt", "Egyptian", "Mamluk", "Fatimid"],
        "IRN": ["Persia", "Persian", "Iran", "Iranian", "Sassanid", "Safavid"],
        "IRQ": ["Iraq", "Mesopotamia", "Babylon"],
        "SYR": ["Syria", "Syrian", "Levant"],
        "ISR": ["Israel", "Palestine", "Judea"],
        "SAU": ["Saudi", "Arabia", "Arab", "Hijaz"],
        "MAR": ["Morocco", "Moroccan", "Moors"],
        "TUN": ["Tunisia", "Tunisian", "Carthage"],
        "DZA": ["Algeria", "Algerian", "Algiers"],
        "LBY": ["Libya", "Libyan"],
        "ETH": ["Ethiopia", "Ethiopian", "Abyssinia"],
        "NGA": ["Nigeria", "Nigerian", "Sokoto"],
        "GHA": ["Ghana", "Ashanti"],
        "ZAF": ["South Africa", "Zulu", "Boer"],
        "CHN": ["China", "Chinese", "Qing", "Ming", "Yuan", "Mongol"],
        "JPN": ["Japan", "Japanese"],
        "KOR": ["Korea", "Korean", "Joseon"],
        "IND": ["India", "Indian", "Mughal", "Maratha", "Sikh"],
        "PAK": ["Pakistan", "Punjab", "Sindh"],
        "BGD": ["Bangladesh", "Bengal"],
        "VNM": ["Vietnam", "Vietnamese", "Annam"],
        "THA": ["Thailand", "Thai", "Siam"],
        "MMR": ["Burma", "Burmese", "Myanmar"],
        "IDN": ["Indonesia", "Java", "Sumatra", "Sumatran", "Dutch East"],
        "PHL": ["Philippines", "Filipino"],
        "USA": ["United States", "American", "US", "U.S."],
        "CAN": ["Canada", "Canadian"],
        "MEX": ["Mexico", "Mexican", "Aztec"],
        "BRA": ["Brazil", "Brazilian", "Portuguese Brazil"],
        "ARG": ["Argentina", "Argentine"],
        "CHL": ["Chile", "Chilean"],
        "PER": ["Peru", "Peruvian", "Inca"],
        "COL": ["Colombia", "New Granada"],
        "VEN": ["Venezuela"],
        "AUS": ["Australia", "Australian"],
        "NZL": ["New Zealand"],
    }
    out: dict[str, list[str]] = {}
    for _, r in iso.iterrows():
        kws = [r["name"]]
        if r["iso3"] in extras:
            kws = extras[r["iso3"]] + kws
        # Dedup, prefer multi-word first
        out[r["iso3"]] = list(dict.fromkeys(kws))
    return out


def _load_brecke() -> pd.DataFrame:
    pre = pd.read_excel(BRECKE / "Brecke-Pre-1400-European-Conflicts.xlsx")
    pre = pre.rename(columns={"Conflict": "Name"})
    pre["TotalFatalities"] = pd.to_numeric(pre.get("Fatalities"), errors="coerce")
    post = pd.read_excel(BRECKE / "Conflict-Catalog-18-vars.xlsx")
    post["TotalFatalities"] = pd.to_numeric(post["TotalFatalities"], errors="coerce")
    keep = ["Name", "StartYear", "EndYear", "TotalFatalities"]
    df = pd.concat([pre[keep], post[keep]], ignore_index=True)
    df["StartYear"] = pd.to_numeric(df["StartYear"], errors="coerce").astype("Int64")
    df["EndYear"] = pd.to_numeric(df["EndYear"], errors="coerce").astype("Int64")
    df["EndYear"] = df["EndYear"].fillna(df["StartYear"])
    df = df.dropna(subset=["StartYear", "Name"])
    df["StartYear"] = df["StartYear"].astype(int); df["EndYear"] = df["EndYear"].astype(int)
    return df


# Region membership for the broader plague catalogue
EUROPEAN_ISO = {"GBR","FRA","DEU","ITA","ESP","PRT","NLD","BEL","POL","RUS",
                 "AUT","HUN","CZE","SWE","NOR","DNK","FIN","GRC","CHE","IRL",
                 "ROU","BGR","SVK","SVN","HRV","SRB","UKR","BLR","LTU","LVA",
                 "EST","ALB","MKD","BIH","MNE","MDA","ISL","LUX","MLT","CYP",
                 "AND","MCO","SMR","LIE","VAT"}
MED_ISO = {"EGY","SYR","TUR","ISR","JOR","LBN","TUN","MAR","DZA","LBY","IRQ","IRN"}

PLAGUE_EUROPE = [(1347,1353),(1361,1363),(1374,1375),(1381,1384),(1388,1390),
                  (1399,1402),(1406,1410),(1413,1414),(1420,1423),(1428,1430),
                  (1438,1440),(1448,1452),(1456,1459),(1463,1467),(1478,1485),
                  (1493,1497),(1518,1521),(1545,1549),(1563,1566),(1575,1577),
                  (1592,1593),(1599,1605),(1623,1627),(1629,1631),(1647,1657),
                  (1665,1666),(1675,1684),(1700,1713),(1720,1722),(1770,1772)]
PLAGUE_MED = [(165,180),(189,191),(249,262),(270,272),
               (541,549),(558,561),(570,574),(588,592),(599,602),(618,622),
               (628,631),(639,642),(664,667),(679,682),(688,691),(704,707),
               (724,729),(740,750),(1347,1353),(1361,1363),(1374,1375),
               (1399,1402),(1438,1440),(1771,1775),(1812,1819),(1834,1838)]


def _build_country_year_indicators(year_min: int = 1421, year_max: int = 1950) -> pd.DataFrame:
    """Country-year war_active and plague_active for every iso3 x year."""
    brecke = _load_brecke()
    kws = _country_keywords()

    # Precompile per-country regexes
    rx = {iso: re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
                            flags=re.IGNORECASE)
          for iso, keys in kws.items() if keys}

    # For each Brecke conflict, find implicated countries
    impl = {iso: [] for iso in rx}
    for _, c in brecke.iterrows():
        name = str(c["Name"])
        ys = int(c["StartYear"]); ye = int(c["EndYear"])
        if ye < year_min or ys > year_max: continue
        for iso, r in rx.items():
            if r.search(name):
                impl[iso].append((max(ys, year_min), min(ye, year_max)))

    rows = []
    years = list(range(year_min, year_max + 1))
    for iso, ranges in impl.items():
        active = np.zeros(len(years), dtype=int)
        for s, e in ranges:
            i0 = s - year_min; i1 = e - year_min + 1
            active[i0:i1] = 1
        is_eu = iso in EUROPEAN_ISO
        is_med = iso in MED_ISO
        plague = np.zeros(len(years), dtype=int)
        if is_eu:
            for s, e in PLAGUE_EUROPE:
                i0 = max(s - year_min, 0); i1 = min(e - year_min + 1, len(years))
                plague[i0:i1] = 1
        if is_med:
            for s, e in PLAGUE_MED:
                i0 = max(s - year_min, 0); i1 = min(e - year_min + 1, len(years))
                plague[i0:i1] = 1
        for j, y in enumerate(years):
            rows.append({"iso3": iso, "year": y,
                          "war_active": int(active[j]),
                          "plague_active": int(plague[j])})
    return pd.DataFrame(rows)


def _attach_to_panel(panel: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Compute war_frac and plague_frac per (iso3, interval) from the country-year
    indicator panel."""
    idx = idx.set_index(["iso3", "year"]).sort_index()
    war_frac = []; plague_frac = []
    for _, r in panel.iterrows():
        i = r["iso3"]; ys = int(r["year"]); ye = int(r["year_next"])
        try:
            sub = idx.loc[i].loc[ys:ye - 1]
            war_frac.append(float(sub["war_active"].mean()))
            plague_frac.append(float(sub["plague_active"].mean()))
        except (KeyError, TypeError):
            war_frac.append(0.0); plague_frac.append(0.0)
    out = panel.copy()
    out["war_frac"] = war_frac
    out["plague_frac"] = plague_frac
    return out


def _run_fe(df: pd.DataFrame, regressors: list[str]) -> dict:
    d = df.dropna(subset=["pop_growth_ann"] + regressors).copy()
    g = d.groupby("iso3")
    for c in regressors:
        d[c] = d[c] - g[c].transform("mean")
    d["pop_growth_ann"] = d["pop_growth_ann"] - g["pop_growth_ann"].transform("mean")
    X = sm.add_constant(d[regressors])
    res = sm.OLS(d["pop_growth_ann"], X).fit(cov_type="cluster",
                                                cov_kwds={"groups": d["iso3"]})
    return {"n": int(res.nobs), "r2": float(res.rsquared),
            "params": res.params, "bse": res.bse, "p": res.pvalues}


def main() -> None:
    print("Building country-year war/pandemic indicators 1421-1950 ...")
    cyi = _build_country_year_indicators()
    print(f"  rows = {len(cyi):,}; countries = {cyi['iso3'].nunique()}")
    print(f"  war_active mean = {cyi['war_active'].mean():.3f}")
    print(f"  plague_active mean = {cyi['plague_active'].mean():.3f}")

    print("\nLoading extended Malthus panel ...")
    panel = pd.read_parquet(DATA / "preindustrial_malthus_panel_extended.parquet")
    panel = _attach_to_panel(panel, cyi)
    panel.to_parquet(DATA / "malthus_conflict_controls.parquet", index=False)
    print(f"  panel = {len(panel):,} cells")
    print(f"  war_frac mean = {panel['war_frac'].mean():.3f}")
    print(f"  plague_frac mean = {panel['plague_frac'].mean():.3f}")

    base_X = ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"]
    ctrl_X = base_X + ["war_frac", "plague_frac"]

    print("\n=== Pooled 1421-1950, country FE ===")
    rB = _run_fe(panel, base_X)
    rC = _run_fe(panel, ctrl_X)
    print(f"  {'regressor':>22} | baseline (SE)              | with controls (SE)")
    for c in ctrl_X:
        b_b = rB["params"].get(c, np.nan); b_b_se = rB["bse"].get(c, np.nan); b_p = rB["p"].get(c, np.nan)
        c_b = rC["params"].get(c, np.nan); c_b_se = rC["bse"].get(c, np.nan); c_p = rC["p"].get(c, np.nan)
        def _s(p): return ("***" if p < 0.01 else "**" if p < 0.05
                            else "*" if p < 0.10 else "") if not np.isnan(p) else ""
        b_str = f"{b_b:+.5f} ({b_b_se:.5f}){_s(b_p)}" if not np.isnan(b_b) else "--"
        c_str = f"{c_b:+.5f} ({c_b_se:.5f}){_s(c_p)}"
        print(f"  {c:>22} | {b_str:<27} | {c_str}")
    print(f"  N: baseline {rB['n']}, with ctrl {rC['n']}    R²: {rB['r2']:.4f} -> {rC['r2']:.4f}")

    print("\n=== Pathway-stratified with conflict controls ===")
    pwrows = []
    for cl in sorted(panel["cluster"].unique()):
        sub = panel[panel["cluster"] == cl]
        if sub["iso3"].nunique() < 2: continue
        rb = _run_fe(sub, base_X)
        rc = _run_fe(sub, ctrl_X)
        pwrows.append({
            "pathway": PATHWAY_NAMES[cl], "n": rc["n"],
            "beta_density_base": rb["params"].get("log_density", np.nan),
            "p_density_base":    rb["p"].get("log_density", np.nan),
            "beta_density_ctrl": rc["params"].get("log_density", np.nan),
            "p_density_ctrl":    rc["p"].get("log_density", np.nan),
            "beta_Tstd_base":    rb["params"].get("t_std_int", np.nan),
            "p_Tstd_base":       rb["p"].get("t_std_int", np.nan),
            "beta_Tstd_ctrl":    rc["params"].get("t_std_int", np.nan),
            "p_Tstd_ctrl":       rc["p"].get("t_std_int", np.nan),
            "beta_war":          rc["params"].get("war_frac", np.nan),
            "p_war":             rc["p"].get("war_frac", np.nan),
            "beta_plague":       rc["params"].get("plague_frac", np.nan),
            "p_plague":          rc["p"].get("plague_frac", np.nan),
        })
    res = pd.DataFrame(pwrows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    res.to_parquet(DATA / "malthus_conflict_results.parquet", index=False)
    print(f"\nSaved {DATA/'malthus_conflict_results.parquet'}")


if __name__ == "__main__":
    main()
