"""Volcanic eruptions as quasi-natural experiments for the Malthusian model.

For each major eruption in 1421-2008, we compute country-level cold-shock
exposure (T anomaly post-eruption minus a pre-eruption baseline) and link
to pre/post HYDE population growth.

Identification: eruptions are exogenous to economic activity; the spatial
pattern of cooling generates variation across countries with the same
pathway. Differences in demographic response by pathway speak to
Prediction 3 (vulnerability of intensive vs pastoral systems).

Eruptions selected by Sigl et al. (2015) volcanic forcing record (top
historical injections within ModE-RA coverage):
    Huaynaputina 1600, Parker 1641, Tambora 1815, Krakatoa 1883, Pinatubo 1991.

Outputs:
    analysis/data/volcanic_event_panel.parquet
    analysis/data/volcanic_event_results.parquet
    analysis/figures/paper4/fig11_volcanic_event.png
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4"
FIG.mkdir(parents=True, exist_ok=True)

PATHWAY_NAMES = {
    0: "Crop-dominant late",
    1: "Pastoral/mixed late",
    2: "Irrigation pioneer",
    3: "High-density intensive",
    4: "Early extensifiers",
}

ERUPTIONS = [
    ("Huaynaputina", 1600),
    ("Parker",       1641),
    ("Tambora",      1815),
    ("Krakatoa",     1883),
    ("Pinatubo",     1991),
]

PRE_WINDOW = (-5, -1)   # baseline window
POST_WINDOW = (0, 2)    # shock window (year of eruption + 2 years)


def _load_hyde_population_wide() -> pd.DataFrame:
    """Return iso3 x year wide population table by summing sub-national rows.

    subpop_4apr2025.csv has rows like isolink=108001 (Burundi sub-region 001).
    We extract the leading 3-digit ISO numeric and sum across sub-regions.
    """
    subpop = pd.read_csv(ROOT / "gbc2025_7apr_base" / "subpop_4apr2025.csv")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))

    subpop = subpop.dropna(subset=["isolink"]).copy()
    # leading digits = country numeric
    subpop["iso_num"] = (subpop["isolink"].astype(int) // 1000).astype(int)
    subpop["iso3"] = subpop["iso_num"].map(num_to_iso3)
    subpop = subpop.dropna(subset=["iso3"]).copy()
    year_cols = [c for c in subpop.columns if c.startswith("y")]
    agg = subpop.groupby("iso3", as_index=False)[year_cols].sum(min_count=1)
    return agg


def _pop_at(wide_df: pd.DataFrame, year: int) -> pd.Series:
    col = f"y{year}"
    if col not in wide_df.columns:
        return None
    return wide_df.set_index("iso3")[col]


def _build_event_panel() -> pd.DataFrame:
    climate = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    climate = climate[["iso3", "year", "t_c", "p_mm"]]

    wide = _load_hyde_population_wide()

    rows = []
    for name, y in ERUPTIONS:
        pre_yrs = list(range(y + PRE_WINDOW[0], y + PRE_WINDOW[1] + 1))
        post_yrs = list(range(y + POST_WINDOW[0], y + POST_WINDOW[1] + 1))

        # Country-level T anomaly: post-mean minus pre-mean, within country
        c = climate[climate["year"].isin(pre_yrs + post_yrs)]
        c_pre = c[c["year"].isin(pre_yrs)].groupby("iso3")["t_c"].mean().rename("t_pre")
        c_post = c[c["year"].isin(post_yrs)].groupby("iso3")["t_c"].mean().rename("t_post")
        c_pre_p = c[c["year"].isin(pre_yrs)].groupby("iso3")["p_mm"].mean().rename("p_pre")
        c_post_p = c[c["year"].isin(post_yrs)].groupby("iso3")["p_mm"].mean().rename("p_post")
        shock = pd.concat([c_pre, c_post, c_pre_p, c_post_p], axis=1).dropna()
        shock["t_shock"] = shock["t_post"] - shock["t_pre"]
        shock["p_shock"] = shock["p_post"] - shock["p_pre"]
        shock["eruption"] = name
        shock["eruption_year"] = y

        # Population response: 10-year change centered on eruption year if HYDE has it.
        # For 1600/1641 we use 1600->1700; for 1815 we use 1810->1820; 1883: 1880->1890; 1991: 1990->2000.
        # Round to the nearest available HYDE year.
        year_pre_map = {
            1600: ("y1500", "y1600"),
            1641: ("y1600", "y1700"),
            1815: ("y1810", "y1820"),
            1883: ("y1880", "y1890"),
            1991: ("y1990", "y2000"),
        }
        a, b = year_pre_map[y]
        if a not in wide.columns or b not in wide.columns:
            continue
        sub = wide[["iso3", a, b]].copy()
        sub = sub.rename(columns={a: "pop_pre", b: "pop_post"})
        sub = sub[(sub["pop_pre"] > 0) & (sub["pop_post"] > 0)]
        sub["pop_growth_around"] = (
            np.log(sub["pop_post"]) - np.log(sub["pop_pre"])
        )
        pre_y_int = int(a[1:])
        post_y_int = int(b[1:])
        sub["window_years"] = post_y_int - pre_y_int
        sub["pop_growth_ann"] = sub["pop_growth_around"] / sub["window_years"]

        shock = shock.merge(sub, left_index=True, right_on="iso3", how="inner")
        rows.append(shock)

    panel = pd.concat(rows, ignore_index=True)
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)
    clust["cluster"] = clust["cluster"].astype(int)
    panel = panel.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    panel["pathway"] = panel["cluster"].map(PATHWAY_NAMES)
    return panel


def main() -> None:
    print("Building volcanic event panel...", flush=True)
    panel = _build_event_panel()
    print(f"Panel: {len(panel):,} (country, eruption) cells, "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['eruption'].nunique()} eruptions", flush=True)
    panel.to_parquet(DATA / "volcanic_event_panel.parquet", index=False)

    # Summary by eruption
    print("\n=== Median country T-shock by eruption ===")
    print(panel.groupby("eruption").agg(
        n=("iso3", "count"),
        med_t_shock=("t_shock", "median"),
        med_p_shock=("p_shock", "median"),
        med_pop_growth_ann=("pop_growth_ann", "median"),
    ).round(4))

    # Main regression: pop_growth_ann ~ t_shock interacted with pathway
    # Two-way FE: country + eruption
    print("\n=== Regression: annual pop growth ~ T-shock × pathway, eruption FE ===")
    df = panel.dropna(subset=["t_shock", "p_shock", "pop_growth_ann", "cluster"]).copy()
    # eruption fixed effects
    df_dum = pd.get_dummies(df["eruption"], prefix="erup", drop_first=True).astype(float)
    # pathway dummies
    pw_dum = pd.get_dummies(df["pathway"], prefix="pw", drop_first=True).astype(float)
    # interaction: pathway × t_shock
    pw_int = pw_dum.multiply(df["t_shock"].values, axis=0)
    pw_int.columns = [c + "_x_Tshock" for c in pw_int.columns]

    X = pd.concat([
        pd.Series(1.0, index=df.index, name="const"),
        df[["t_shock", "p_shock"]],
        pw_dum,
        pw_int,
        df_dum,
    ], axis=1).astype(float)
    y = df["pop_growth_ann"].astype(float)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["iso3"]})
    print(res.summary())
    coef = pd.concat([res.params.rename("beta"), res.bse.rename("se"),
                      res.pvalues.rename("p")], axis=1)
    coef.to_csv(DATA / "volcanic_event_full_regression.csv")

    # Simpler: by-pathway slope of pop_growth on t_shock with eruption FE
    print("\n=== By-pathway slope of pop_growth_ann on T-shock (eruption FE) ===")
    rows = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        if len(sub) < 15:
            continue
        e_dum = pd.get_dummies(sub["eruption"], drop_first=True).astype(float)
        X2 = pd.concat([pd.Series(1.0, index=sub.index, name="const"),
                        sub[["t_shock", "p_shock"]], e_dum], axis=1).astype(float)
        y2 = sub["pop_growth_ann"].astype(float)
        r = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups": sub["iso3"]})
        rows.append({
            "pathway": PATHWAY_NAMES[cl],
            "n": int(r.nobs),
            "beta_t_shock": r.params["t_shock"],
            "se_t_shock": r.bse["t_shock"],
            "p_t_shock": r.pvalues["t_shock"],
            "beta_p_shock": r.params["p_shock"],
            "p_p_shock": r.pvalues["p_shock"],
            "rsq": r.rsquared,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_parquet(DATA / "volcanic_event_results.parquet", index=False)

    # Plot
    if len(out):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        order = out.sort_values("beta_t_shock").reset_index(drop=True)
        y_pos = np.arange(len(order))
        ax.errorbar(order["beta_t_shock"], y_pos,
                    xerr=1.96 * order["se_t_shock"],
                    fmt="o", color="#0072B2", capsize=4)
        for i, row in order.iterrows():
            mark = ("***" if row["p_t_shock"] < 0.01
                    else "**" if row["p_t_shock"] < 0.05
                    else "*" if row["p_t_shock"] < 0.1
                    else "")
            ax.text(row["beta_t_shock"], i,
                    f"  n={row['n']}  {mark}", va="center", fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(order["pathway"])
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("β: annual pop growth response per °C of volcanic cooling")
        ax.set_title("Volcanic event study (5 eruptions, 1600–1991)\n"
                     "Demographic response to cold shock, by pathway")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = FIG / "fig11_volcanic_event.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
