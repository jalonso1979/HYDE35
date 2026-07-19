"""Individual-eruption placebo: Tambora 1815 alone.

The 5-event pooled placebo at the 25th percentile is honest but not sharp.
We sharpen by running the pathway x shock interaction on Tambora alone (the
cleanest historical signal, Year Without a Summer in 1816), then compare to
1000 placebo years matched on (i) being non-eruption years and (ii) being
in the same general historical era (1750-1900).

The Tambora-1815 effect should be much sharper than the pooled five-event
effect if the cooling signal at 1816 is really volcanic.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}

ERUPTION_YEARS = {1600, 1641, 1815, 1883, 1991}
EXCLUDE = set()
for y in ERUPTION_YEARS:
    EXCLUDE.update(range(y - 5, y + 5))

PRE = (-5, -1); POST = (0, 2)
TARGET_YEAR = 1815


def _hyde_pop_wide() -> pd.DataFrame:
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


def _single_event_panel(year: int, climate: pd.DataFrame,
                         pop_wide: pd.DataFrame, clust: pd.DataFrame) -> pd.DataFrame:
    pre_yrs = list(range(year + PRE[0], year + PRE[1] + 1))
    post_yrs = list(range(year + POST[0], year + POST[1] + 1))
    c = climate[climate["year"].isin(pre_yrs + post_yrs)]
    c_pre = c[c["year"].isin(pre_yrs)].groupby("iso3")["t_c"].mean().rename("t_pre")
    c_post = c[c["year"].isin(post_yrs)].groupby("iso3")["t_c"].mean().rename("t_post")
    s = pd.concat([c_pre, c_post], axis=1).dropna()
    s["t_shock"] = s["t_post"] - s["t_pre"]
    s = s.reset_index()
    ya = (year // 10) * 10
    yb = ya + 10
    if f"y{ya}" not in pop_wide.columns or f"y{yb}" not in pop_wide.columns:
        return pd.DataFrame()
    pp = pop_wide[["iso3", f"y{ya}", f"y{yb}"]].rename(
        columns={f"y{ya}": "pop_pre", f"y{yb}": "pop_post"})
    pp = pp[(pp["pop_pre"] > 0) & (pp["pop_post"] > 0)]
    pp["pop_growth_ann"] = (np.log(pp["pop_post"]) - np.log(pp["pop_pre"])) / 10
    df = s.merge(pp[["iso3", "pop_growth_ann"]], on="iso3", how="inner")
    df = df.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)
    return df


def _run_interaction_reg(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["t_shock", "pop_growth_ann", "cluster"])
    if len(df) < 30:
        return {}
    pw_dum = pd.get_dummies(df["pathway"], prefix="pw", drop_first=True).astype(float)
    pw_int = pw_dum.multiply(df["t_shock"].values, axis=0)
    pw_int.columns = [c + "_x_Tshock" for c in pw_int.columns]
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   df[["t_shock"]], pw_dum, pw_int], axis=1).astype(float)
    y = df["pop_growth_ann"].astype(float)
    try:
        r = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception:
        return {}
    out = {}
    for col in r.params.index:
        if "High-density intensive" in col and "Tshock" in col:
            out["beta"] = r.params[col]
            out["se"] = r.bse[col]
            out["p"] = r.pvalues[col]
            return out
    return out


def main() -> None:
    print("Loading data...")
    climate = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")[["iso3","year","t_c"]]
    pop_wide = _hyde_pop_wide()
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3","cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)

    # Real Tambora 1815
    tambora = _single_event_panel(TARGET_YEAR, climate, pop_wide, clust)
    print(f"\nTambora panel: {len(tambora)} countries")
    print(f"Median country T-shock (post - pre):  {tambora['t_shock'].median():+.3f} °C")
    print(f"Median pop_growth_ann around 1810-1820: {tambora['pop_growth_ann'].median():+.4f}")
    r_real = _run_interaction_reg(tambora)
    print(f"\nTambora alone — high-density intensive x T_shock interaction:")
    print(f"  beta = {r_real['beta']:+.5f},  SE = {r_real['se']:.5f},  p = {r_real['p']:.4f}")

    # Placebo: random years 1750-1900 (excluding ±5 of real eruptions, exclude
    # Tambora itself); use same decadal window setup
    rng = np.random.default_rng(123)
    candidate_years = [y for y in range(1755, 1895) if y not in EXCLUDE]
    print(f"\nRunning 1000 placebo single-year events drawn from {len(candidate_years)} candidates...")

    results = []
    for i in range(1000):
        y = int(rng.choice(candidate_years))
        df = _single_event_panel(y, climate, pop_wide, clust)
        if len(df) == 0: continue
        r = _run_interaction_reg(df)
        if r:
            results.append({"year": y, **r})
        if (i + 1) % 200 == 0:
            print(f"  draw {i+1}/1000", flush=True)
    res = pd.DataFrame(results)
    res.to_parquet(DATA / "tambora_individual_placebo.parquet", index=False)

    print(f"\n=== Placebo distribution (Tambora individual) ===")
    print(f"Valid placebo regressions: {len(res)}")
    print(f"Median placebo beta:  {res['beta'].median():+.5f}")
    print(f"SD     placebo beta:  {res['beta'].std():.5f}")
    print(f"Median placebo p:     {res['p'].median():.3f}")
    print(f"% placebo p < 0.05:   {(res['p'] < 0.05).mean()*100:.1f}%")
    print(f"% placebo p < 0.10:   {(res['p'] < 0.10).mean()*100:.1f}%")
    quant_below = (np.abs(res['beta']) >= np.abs(r_real['beta'])).mean()
    print(f"\nReal Tambora beta = {r_real['beta']:+.5f} (p = {r_real['p']:.3f})")
    print(f"Pr(|placebo| >= |real Tambora|) = {quant_below:.4f}")

    # Figure: density of placebo coefficients with real value marked
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(res['beta'], bins=40, color="#909090", edgecolor="#202020",
            linewidth=0.4, alpha=0.85)
    ax.axvline(r_real['beta'], color="#000000", linewidth=1.5, linestyle="-")
    ax.text(r_real['beta'], ax.get_ylim()[1] * 0.92, f"Tambora 1815\n$\\hat\\beta = {r_real['beta']:+.4f}$",
            ha="center", va="top", fontsize=9, color="#202020",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.5))
    ax.set_xlabel(r"High-density intensive $\times$ $T$-shock interaction coefficient")
    ax.set_ylabel("Placebo draws (out of 1000)")
    ax.set_title("Tambora 1815 vs 1000 placebo single-year events, 1755–1894",
                 loc="left")
    plt.tight_layout()
    fig.savefig(FIG / "fig11_tambora_placebo.pdf")
    fig.savefig(FIG / "fig11_tambora_placebo.png")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig11_tambora_placebo.pdf'}")


if __name__ == "__main__":
    main()
