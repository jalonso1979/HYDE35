"""Placebo test for the volcanic event study.

Re-run the panel event study on 5 randomly-drawn non-eruption years
between 1600 and 1991. The pathway x shock interaction should be
INSIGNIFICANT on placebo years if the true result reflects real volcanic
forcing rather than spurious noise.

We repeat for 1000 random draws and report the distribution of:
  - the high-density intensive x T_shock interaction coefficient
  - its p-value

The true result is interaction beta = -0.0035, p = 0.032.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}

# Real eruptions
ERUPTION_YEARS = {1600, 1641, 1815, 1883, 1991}
# Plus exclude ±5 years around each real eruption from placebo pool
EXCLUDE = set()
for y in ERUPTION_YEARS:
    EXCLUDE.update(range(y - 5, y + 5))

PRE_WINDOW = (-5, -1)
POST_WINDOW = (0, 2)
N_PLACEBO_DRAWS = 1000


def _hyde_population_wide() -> pd.DataFrame:
    sub = pd.read_csv(ROOT / "gbc2025_7apr_base" / "subpop_4apr2025.csv")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    sub = sub.dropna(subset=["isolink"]).copy()
    sub["iso_num"] = (sub["isolink"].astype(int) // 1000).astype(int)
    sub["iso3"] = sub["iso_num"].map(num_to_iso3)
    sub = sub.dropna(subset=["iso3"]).copy()
    ycols = [c for c in sub.columns if c.startswith("y")]
    return sub.groupby("iso3", as_index=False)[ycols].sum(min_count=1)


def _build_panel_for_year(placebo_year: int, climate: pd.DataFrame,
                            pop_wide: pd.DataFrame) -> pd.DataFrame:
    """Construct one (event, country) panel for a placebo year y."""
    pre_yrs = list(range(placebo_year + PRE_WINDOW[0], placebo_year + PRE_WINDOW[1] + 1))
    post_yrs = list(range(placebo_year + POST_WINDOW[0], placebo_year + POST_WINDOW[1] + 1))
    c = climate[climate["year"].isin(pre_yrs + post_yrs)]
    c_pre = c[c["year"].isin(pre_yrs)].groupby("iso3")["t_c"].mean().rename("t_pre")
    c_post = c[c["year"].isin(post_yrs)].groupby("iso3")["t_c"].mean().rename("t_post")
    c_pre_p = c[c["year"].isin(pre_yrs)].groupby("iso3")["p_mm"].mean().rename("p_pre")
    c_post_p = c[c["year"].isin(post_yrs)].groupby("iso3")["p_mm"].mean().rename("p_post")
    s = pd.concat([c_pre, c_post, c_pre_p, c_post_p], axis=1).dropna()
    s["t_shock"] = s["t_post"] - s["t_pre"]
    s["p_shock"] = s["p_post"] - s["p_pre"]
    s = s.reset_index()

    # Population change in the surrounding decade
    ya = (placebo_year // 10) * 10
    yb = ya + 10
    if f"y{ya}" not in pop_wide.columns or f"y{yb}" not in pop_wide.columns:
        return pd.DataFrame()
    pp = pop_wide[["iso3", f"y{ya}", f"y{yb}"]].copy()
    pp.columns = ["iso3", "pop_pre", "pop_post"]
    pp = pp[(pp["pop_pre"] > 0) & (pp["pop_post"] > 0)]
    pp["pop_growth_ann"] = (np.log(pp["pop_post"]) - np.log(pp["pop_pre"])) / 10
    s = s.merge(pp[["iso3", "pop_growth_ann"]], on="iso3", how="inner")
    s["placebo_year"] = placebo_year
    return s


def run_placebo_regression(panel: pd.DataFrame) -> dict:
    df = panel.dropna(subset=["t_shock", "p_shock", "pop_growth_ann", "cluster"]).copy()
    if len(df) < 50: return {}
    df_dum = pd.get_dummies(df["placebo_year"], prefix="erup", drop_first=True).astype(float)
    pw_dum = pd.get_dummies(df["pathway"], prefix="pw", drop_first=True).astype(float)
    pw_int = pw_dum.multiply(df["t_shock"].values, axis=0)
    pw_int.columns = [c + "_x_Tshock" for c in pw_int.columns]
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   df[["t_shock", "p_shock"]], pw_dum, pw_int, df_dum], axis=1).astype(float)
    y = df["pop_growth_ann"].astype(float)
    try:
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["iso3"]})
    except Exception:
        return {}
    out = {}
    # Look for the high-density intensive x T_shock interaction
    for col in r.params.index:
        if "High-density intensive" in col and "Tshock" in col:
            out["beta"] = r.params[col]
            out["p"] = r.pvalues[col]
            out["se"] = r.bse[col]
            return out
    return out


def main() -> None:
    print("Loading data...")
    climate = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    climate = climate[["iso3", "year", "t_c", "p_mm"]]
    pop_wide = _hyde_population_wide()
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)

    rng = np.random.default_rng(42)
    candidate_years = [y for y in range(1605, 1990)
                       if y not in EXCLUDE]
    print(f"  Candidate placebo years: {len(candidate_years)} (excluding ±5 of real eruptions)")

    print(f"\nRunning {N_PLACEBO_DRAWS} placebo draws of 5 years each...")
    results = []
    for i in range(N_PLACEBO_DRAWS):
        placebo_years = rng.choice(candidate_years, size=5, replace=False)
        panels = []
        for y in placebo_years:
            p = _build_panel_for_year(int(y), climate, pop_wide)
            if len(p) > 0:
                panels.append(p)
        if len(panels) < 3:
            continue
        full = pd.concat(panels, ignore_index=True)
        full = full.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
        full["pathway"] = full["cluster"].map(PATHWAY_NAMES)
        r = run_placebo_regression(full)
        if r:
            results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  draw {i+1}/{N_PLACEBO_DRAWS}, valid so far: {len(results)}", flush=True)
    res = pd.DataFrame(results)
    res.to_parquet(DATA / "volcanic_placebo_results.parquet", index=False)

    print(f"\n=== Placebo distribution (high-density intensive x T_shock) ===")
    print(f"N placebo draws with valid regression: {len(res)}")
    print(f"Mean placebo beta:   {res['beta'].mean():+.5f}")
    print(f"SD   placebo beta:   {res['beta'].std():.5f}")
    print(f"Median placebo p:    {res['p'].median():.3f}")
    print(f"% with placebo p < 0.05:  {(res['p'] < 0.05).mean()*100:.1f}%")
    print(f"% with placebo p < 0.10:  {(res['p'] < 0.10).mean()*100:.1f}%")
    print()
    print(f"Real result:  beta = -0.0035, p = 0.032")
    print(f"Quantile of |real beta| in |placebo beta|:")
    q = (res['beta'].abs() >= abs(-0.0035)).mean()
    print(f"  Pr(|placebo| >= |real|) = {q:.4f}  (smaller = more impressive)")


if __name__ == "__main__":
    main()
