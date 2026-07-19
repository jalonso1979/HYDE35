"""Exercise 1 (CHRE-robustness): re-run the headline Shapley decomposition
with `pandemic_intensity_norm` replaced by a CHRE-coin-hoards-derived index.

Logic: the hand-coded `pandemic_intensity_norm` top-codes 11 Old-World
countries at 1.0 (Antonine + Cyprian + Justinianic + BD exposure).  The
CHRE coin-hoards index from `/shared/pandemics_v3/data/coin_hoards_region_25yr.csv`
gives within-Roman-empire heterogeneity that the hand-coded index lacks
(Gaul 2.88, Italy 2.39, Africa-Maghreb 0.91 mean log1p hoards in the
150-775 CE pandemic window).

If the Shapley shares are stable under this swap, that's evidence the
horserace headline result is insensitive to the pandemic-intensity
measurement convention.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper5_horserace.shapley import shapley_r2_decomposition

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
SHARED = ROOT / "analysis/shared/pandemics_v3/data"

# CHRE region -> ISO3 mapping (rough; matches Roman provincial geography
# to modern state borders)
REGION_TO_ISO = {
    "Italy":             ["ITA"],
    "Gaul":              ["FRA"],
    "Britain":           ["GBR"],
    "Iberia":            ["ESP", "PRT"],
    "Germania":          ["DEU"],
    "Pannonia":          ["HUN", "AUT"],
    "Greece-Macedonia":  ["GRC"],
    "Anatolia":          ["TUR"],
    "Syria-Levant":      ["SYR", "LBN", "JOR", "ISR"],
    "Egypt-Cyrenaica":   ["EGY", "LBY"],
    "Africa-Maghreb":    ["DZA", "TUN", "MAR"],
    "Crimea":            ["UKR"],
}


def build_chre_alt_index() -> pd.DataFrame:
    """Return DataFrame with iso3, pandemic_intensity_chre (normalised [0,1])."""
    h = pd.read_csv(SHARED / "coin_hoards_region_25yr.csv")
    # 150-775 CE = full Antonine→post-Justinianic window
    win = h[(h.year_bin_start >= 150) & (h.year_bin_start <= 775)]
    agg = win.groupby("region")["log1p_n_hoards"].mean().reset_index(name="mean_log_hoards")
    rows = []
    for region, isos in REGION_TO_ISO.items():
        m = agg.loc[agg.region == region, "mean_log_hoards"]
        v = float(m.iloc[0]) if not m.empty else np.nan
        for iso in isos:
            rows.append({"iso3": iso, "chre_log_hoards": v})
    chre = pd.DataFrame(rows).groupby("iso3", as_index=False)["chre_log_hoards"].mean()
    # Normalise to [0,1] across the 20 CHRE-covered countries.
    chre["pandemic_intensity_chre"] = ((chre["chre_log_hoards"] - chre["chre_log_hoards"].min())
                                        / (chre["chre_log_hoards"].max() - chre["chre_log_hoards"].min()))
    return chre[["iso3", "pandemic_intensity_chre"]]


SUBSTRATES_BASE = ["sigma_v_T_pre1750", "H_pred_pwadj", "ancestral_yield_log"]
PAND_BASE = "pandemic_intensity_norm"
PAND_CHRE = "pandemic_intensity_chre"
OUTCOMES = ["log_pop_growth_1950_2025", "urban_change_1950_2025",
             "log_gdppc_2015", "dt_timing_year"]
CONTROLS = ["abs_lat", "log_area", "landlocked", "ruggedness_proxy",
             "log_dist_neolithic"]


def main() -> None:
    df = pd.read_parquet(PANEL)
    chre = build_chre_alt_index()
    df = df.merge(chre, on="iso3", how="left")
    print(f"Panel N: {len(df)}, CHRE-pandemic coverage: "
          f"{df.pandemic_intensity_chre.notna().sum()}")
    # For countries OUTSIDE the CHRE Roman-empire footprint, set the alt
    # pandemic index to 0 (consistent with the hand-coded approach: non-
    # exposure → no pandemic intensity).
    df["pandemic_intensity_chre"] = df["pandemic_intensity_chre"].fillna(0.0)
    print(f"After fill-zero for non-Roman countries: "
          f"{df.pandemic_intensity_chre.notna().sum()} (should equal panel N)")

    # Two parallel Shapley runs: base index vs CHRE alt
    for label, pand_var in [("baseline (hand-coded)", PAND_BASE),
                             ("CHRE-z robustness",     PAND_CHRE)]:
        print(f"\n=== Shapley shares — {label} ({pand_var}) ===")
        rows = []
        for outcome in OUTCOMES:
            substrates = SUBSTRATES_BASE + [pand_var]
            keep = [outcome] + substrates + CONTROLS
            sub = df.dropna(subset=keep).copy()
            result = shapley_r2_decomposition(
                df=sub,
                y_col=outcome,
                substrates=substrates,
                controls=CONTROLS,
            )
            shares = result.get("shapley", {})
            r = {s: float(shares.get(s, np.nan)) for s in substrates}
            r["outcome"] = outcome
            r["N"] = int(result.get("n_obs", len(sub)))
            rows.append(r)
        out = pd.DataFrame(rows).set_index("outcome")
        print(out.round(4).to_string())

    print("\nDone.  Compare the 'pandemic_intensity_*' columns across the two specs.")


if __name__ == "__main__":
    main()
