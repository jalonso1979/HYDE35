"""Individual-eruption event studies on cropland-share growth for the
crop-dominant late pathway.

The continuous-VSSI Boserupian regression of Section 4.1 identifies the
average pathway-by-decade response to volcanic forcing.  A reviewer's
natural follow-up is to look at the response to specific, well-identified
single eruptions.  This script runs three:

  - Laki        1783 (NH extra-tropical, ~120 Tg VSSI)
  - Tambora     1815 (tropical, ~60 Tg VSSI)
  - Krakatoa    1883 (tropical, ~25 Tg VSSI)

All three fall inside HYDE's decadal post-1700 portion of the panel, so
their event-decade can be assigned cleanly.

Event-time horizons (decades relative to the eruption decade):
  h = -3, -2  pre-trends
  h = -1      reference (omitted)
  h =  0      eruption decade
  h = +1, +2, +3, +4  post-eruption

Specification: OLS on the crop-dominant late panel with country FE and
horizon dummies, country-clustered SEs.

Outputs:
    analysis/data/boserup_event_study.parquet
    analysis/figures/paper4_v2/figK_boserup_eventstudy.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

ERUPTIONS = [
    {"name": "Laki",     "year": 1783, "decade": 1780, "vssi_tg": 120,
     "lat_class": "NH extra-tropical"},
    {"name": "Tambora",  "year": 1815, "decade": 1810, "vssi_tg": 60,
     "lat_class": "tropical"},
    {"name": "Krakatoa", "year": 1883, "decade": 1880, "vssi_tg": 25,
     "lat_class": "tropical"},
]

HORIZONS = [-3, -2, -1, 0, 1, 2, 3, 4]  # h=-1 dropped as reference
REF_H = -1


def _event_panel(panel: pd.DataFrame, decade: int) -> pd.DataFrame:
    """For a given eruption decade, return a country-decade panel with
    event-time h = (year - decade) / 10 for h in [-3, 4]."""
    df = panel.copy()
    df["h"] = ((df["year"] - decade) // 10).astype(int)
    df = df[df["h"].isin(HORIZONS)].copy()
    return df


def _run_event_study(df: pd.DataFrame, eruption: str) -> pd.DataFrame:
    """Country-FE OLS with horizon dummies; ref = h=-1."""
    d = df.dropna(subset=["g_crop_ann", "iso3"]).copy()
    if len(d) < 20:
        return None

    # Build horizon dummies (drop reference)
    for h in HORIZONS:
        if h == REF_H:
            continue
        d[f"h_{h:+d}"] = (d["h"] == h).astype(float)

    # Within-country demeaning
    dum_cols = [f"h_{h:+d}" for h in HORIZONS if h != REF_H]
    g = d.groupby("iso3")
    for c in ["g_crop_ann"] + dum_cols:
        d[c] = d[c] - g[c].transform("mean")

    X = sm.add_constant(d[dum_cols])
    res = sm.OLS(d["g_crop_ann"], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"]}
    )

    out = []
    for h in HORIZONS:
        if h == REF_H:
            out.append({"eruption": eruption, "h": h, "beta": 0.0,
                        "se": 0.0, "p": 1.0,
                        "n_iso": d["iso3"].nunique()})
        else:
            col = f"h_{h:+d}"
            out.append({"eruption": eruption, "h": h,
                        "beta": float(res.params.get(col, np.nan)),
                        "se":   float(res.bse.get(col,    np.nan)),
                        "p":    float(res.pvalues.get(col, np.nan)),
                        "n_iso": d["iso3"].nunique()})
    return pd.DataFrame(out)


def _pooled_eruption_test(results: list[pd.DataFrame]) -> dict:
    """Pool across three eruptions: average h=+1 and h=+2 coefficients,
    inverse-variance weighted."""
    pooled = {}
    for h in [0, 1, 2, 3]:
        betas, ses, names = [], [], []
        for r in results:
            if r is None: continue
            row = r[r["h"] == h]
            if len(row) == 0: continue
            row = row.iloc[0]
            betas.append(row["beta"]); ses.append(row["se"]); names.append(row["eruption"])
        if len(betas) < 2:
            continue
        betas = np.array(betas); ses = np.array(ses)
        w = 1.0 / ses**2
        pool_b = float(np.sum(w * betas) / np.sum(w))
        pool_se = float(np.sqrt(1.0 / np.sum(w)))
        from scipy.stats import norm
        pool_p = float(2 * (1 - norm.cdf(abs(pool_b / pool_se))))
        # Cochran Q
        Q = float(np.sum(w * (betas - pool_b)**2))
        from scipy.stats import chi2
        q_df = len(betas) - 1
        q_p = float(1 - chi2.cdf(Q, q_df)) if q_df > 0 else np.nan
        pooled[h] = {
            "h": h, "beta": pool_b, "se": pool_se, "p": pool_p,
            "Q": Q, "Q_df": q_df, "Q_p": q_p,
            "n_eruptions": len(betas),
            "eruptions": ",".join(names),
        }
    return pooled


def main() -> None:
    panel = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    cdl = panel[panel["cluster"] == 0].copy()
    print(f"Crop-dominant late panel: {len(cdl):,} rows, "
          f"{cdl['iso3'].nunique()} countries\n")

    all_rows = []
    eruption_results = []

    for eruption in ERUPTIONS:
        name, decade, vssi = eruption["name"], eruption["decade"], eruption["vssi_tg"]
        print(f"=== {name} {eruption['year']} (decade {decade}, "
              f"{vssi} Tg, {eruption['lat_class']}) ===")
        ep = _event_panel(cdl, decade)
        n_iso = ep["iso3"].nunique()
        print(f"  N (panel) = {len(ep)}, n_countries = {n_iso}")

        if len(ep) < 20:
            print("  Too few observations, skipping.\n")
            continue

        res = _run_event_study(ep, name)
        if res is None:
            print("  Regression failed.\n")
            continue

        # Pretty print
        for _, r in res.iterrows():
            sig = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
            print(f"  h = {int(r['h']):+d}: β = {r['beta']:+.5f}  "
                  f"(SE {r['se']:.5f}) p = {r['p']:.3g}  {sig}")
        print()
        all_rows.append(res)
        eruption_results.append(res)

    combined = pd.concat(all_rows, ignore_index=True)

    # Pooled tests
    pooled = _pooled_eruption_test(eruption_results)
    print("=== Pooled (inverse-variance weighted across the three eruptions) ===")
    for h, p in pooled.items():
        sig = "***" if p["p"] < 0.01 else "**" if p["p"] < 0.05 else "*" if p["p"] < 0.10 else ""
        print(f"  h = +{int(h):d}: pooled β = {p['beta']:+.5f}  "
              f"(SE {p['se']:.5f})  p = {p['p']:.3g}  {sig}  "
              f"[Q = {p['Q']:.2f}, p = {p['Q_p']:.3g}]")
    pool_df = pd.DataFrame(pooled.values())
    pool_df["eruption"] = "POOLED"
    combined_with_pool = pd.concat([combined, pool_df], ignore_index=True)
    combined_with_pool.to_parquet(DATA / "boserup_event_study.parquet", index=False)
    print(f"\nSaved {DATA/'boserup_event_study.parquet'}")

    # Figure: per-eruption event-study horizon plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, res, eruption in zip(axes, all_rows, ERUPTIONS):
        name = eruption["name"]
        h_vals = res["h"].astype(float).values
        b = res["beta"].values * 1000   # to per-mille for readability
        se = res["se"].values * 1000
        ax.errorbar(h_vals, b, yerr=1.96 * se, fmt="o-", color="#A02020",
                    markerfacecolor="white", markersize=5, capsize=2.5,
                    elinewidth=0.8, linewidth=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name} {eruption['year']} ({eruption['vssi_tg']} Tg)",
                      fontsize=10)
        ax.set_xlabel("Decades relative to eruption")
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4])
        if ax is axes[0]:
            ax.set_ylabel(r"$\hat\beta_h$ on $\Delta \log s$  ($\times 10^{-3}$)")
    fig.suptitle("Boserupian cropland-share response: individual-eruption event studies "
                 "(crop-dominant late pathway)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "figK_boserup_eventstudy.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figK_boserup_eventstudy.png", dpi=160, bbox_inches="tight")
    print(f"Saved {FIG/'figK_boserup_eventstudy.pdf'}")


if __name__ == "__main__":
    main()
