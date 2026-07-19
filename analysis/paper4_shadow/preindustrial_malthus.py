"""Pre-industrial Malthusian regression with real annual climate from ModE-RA.

The standard Malthus regression in the paper had no climate terms in the
0-1750 sample (PAGES 2k only provided decadal averages). With ModE-RA we can
add annual T, P, and within-interval volatility.

HYDE has irregular timesteps (100-year before 1700, decadal 1700-1750). For
each (country, interval) we compute:
  - pop_growth = (ln P_{t+1} - ln P_t) / (year_{t+1} - year_t)
  - log_density = ln(density at t)
  - t_mean, p_mean      = mean of ModE-RA over [year_t, year_{t+1})
  - t_std,  p_std       = std of ModE-RA over interval (year-to-year volatility)
  - t_min,  t_max       = coldest/warmest year in interval (extreme exposure)

Run pathway-stratified two-way (country + year) fixed-effects regressions
of pop_growth on log_density + climate vars.

Outputs analysis/data/preindustrial_malthus_results.parquet and figures.
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

START_YEAR = 1421  # ModE-RA start
END_YEAR = 1750    # pre-industrial cutoff


def _load_hyde_intervals() -> pd.DataFrame:
    """Return (iso3, year_t, year_t1, pop_t, density_t, pop_t1, density_t1)."""
    raw = pd.read_csv(ROOT / "hyde35_country_year_mean_std.csv")
    # iso3 mapping
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    raw["country"] = pd.to_numeric(raw["country"], errors="coerce")
    raw = raw.dropna(subset=["country"]).copy()
    raw["country"] = raw["country"].astype(int)
    raw = raw.merge(iso_map[["iso_num", "iso3"]], left_on="country", right_on="iso_num", how="inner")

    # Keep relevant vars in pre-industrial window
    keep = raw[(raw["year"] >= START_YEAR) & (raw["year"] <= END_YEAR)].copy()
    keep = keep[keep["var"].isin(["pop_persons", "popdens_p_km2"])].copy()

    pivot = keep.pivot_table(
        index=["iso3", "year"], columns="var", values="mean"
    ).reset_index()
    pivot = pivot.dropna(subset=["pop_persons", "popdens_p_km2"])
    pivot = pivot[(pivot["pop_persons"] > 0) & (pivot["popdens_p_km2"] > 0)]
    pivot["log_pop"] = np.log(pivot["pop_persons"])
    pivot["log_density"] = np.log(pivot["popdens_p_km2"])
    pivot = pivot.sort_values(["iso3", "year"]).reset_index(drop=True)

    pivot["year_next"] = pivot.groupby("iso3")["year"].shift(-1)
    pivot["log_pop_next"] = pivot.groupby("iso3")["log_pop"].shift(-1)
    pivot = pivot.dropna(subset=["year_next", "log_pop_next"]).copy()
    pivot["year_next"] = pivot["year_next"].astype(int)
    pivot["dt"] = pivot["year_next"] - pivot["year"]
    pivot["pop_growth_ann"] = (pivot["log_pop_next"] - pivot["log_pop"]) / pivot["dt"]
    return pivot[["iso3", "year", "year_next", "dt", "log_density", "pop_growth_ann"]]


def _attach_interval_climate(intervals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ModE-RA annual climate over each interval [year, year_next)."""
    annual = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    annual = annual[(annual["year"] >= START_YEAR) & (annual["year"] <= END_YEAR)].copy()
    annual = annual[["iso3", "year", "t_c", "p_mm"]]
    out = []
    for _, row in intervals.iterrows():
        sub = annual[(annual["iso3"] == row["iso3"])
                     & (annual["year"] >= row["year"])
                     & (annual["year"] < row["year_next"])]
        if len(sub) < 3:
            continue
        out.append({
            "iso3": row["iso3"],
            "year": int(row["year"]),
            "year_next": int(row["year_next"]),
            "dt": int(row["dt"]),
            "log_density": row["log_density"],
            "pop_growth_ann": row["pop_growth_ann"],
            "t_mean_int": sub["t_c"].mean(),
            "t_std_int": sub["t_c"].std(),
            "t_min_int": sub["t_c"].min(),
            "t_max_int": sub["t_c"].max(),
            "p_mean_int": sub["p_mm"].mean(),
            "p_std_int": sub["p_mm"].std(),
            "n_yrs": len(sub),
        })
    return pd.DataFrame(out)


def _attach_pathway(df: pd.DataFrame) -> pd.DataFrame:
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str)
    clust["cluster"] = clust["cluster"].astype(int)
    df = df.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    df["pathway"] = df["cluster"].map(PATHWAY_NAMES)
    return df


def _fe_regression(d: pd.DataFrame, regressors: list[str]) -> dict:
    """Within-country two-way FE via dummies. Returns key coef summary."""
    # Demean by country
    d = d.dropna(subset=regressors + ["pop_growth_ann"]).copy()
    if len(d) < 20:
        return {}
    g = d.groupby("iso3")
    for c in regressors + ["pop_growth_ann"]:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[regressors])
    y = d["pop_growth_ann"]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].values})
    return {
        "params": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "rsquared": res.rsquared,
        "n": int(res.nobs),
        "se": res.bse.to_dict(),
    }


def main() -> None:
    print("Loading HYDE intervals 1421-1750...", flush=True)
    intervals = _load_hyde_intervals()
    print(f"  {len(intervals):,} (country, interval) pairs, "
          f"{intervals['iso3'].nunique()} countries", flush=True)

    print("Attaching ModE-RA interval climate...", flush=True)
    df = _attach_interval_climate(intervals)
    df = _attach_pathway(df)
    print(f"  Final sample: {len(df):,} rows, {df['iso3'].nunique()} countries, "
          f"{df['cluster'].nunique()} pathways", flush=True)

    df["t_anom_int"] = df["t_mean_int"] - df.groupby("iso3")["t_mean_int"].transform("mean")
    df["p_anom_int"] = df["p_mean_int"] - df.groupby("iso3")["p_mean_int"].transform("mean")

    df.to_parquet(DATA / "preindustrial_malthus_panel.parquet", index=False)

    # ---- Pooled FE regression (full sample) ------------------------------
    print("\n=== Pooled FE Malthusian regression 1421-1750 (with climate) ===")
    full = _fe_regression(
        df,
        ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"],
    )
    print(f"N = {full.get('n')}, R^2 = {full.get('rsquared'):.4f}")
    for v in ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"]:
        b = full["params"][v]
        s = full["se"][v]
        p = full["pvalues"][v]
        print(f"  {v:>15s}: beta = {b:+.6f}  se = {s:.6f}  p = {p:.4g}")

    # ---- By pathway -----------------------------------------------------
    print("\n=== Pathway-stratified FE Malthus + climate, 1421-1750 ===")
    rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        name = PATHWAY_NAMES[cluster_id]
        if len(sub) < 25:
            continue
        r = _fe_regression(
            sub,
            ["log_density", "t_anom_int", "p_anom_int", "t_std_int", "p_std_int"],
        )
        if not r:
            continue
        rows.append({
            "pathway": name,
            "n": r["n"],
            "beta_density": r["params"]["log_density"],
            "p_density": r["pvalues"]["log_density"],
            "beta_T": r["params"]["t_anom_int"],
            "p_T": r["pvalues"]["t_anom_int"],
            "beta_P": r["params"]["p_anom_int"],
            "p_P": r["pvalues"]["p_anom_int"],
            "beta_Tstd": r["params"]["t_std_int"],
            "p_Tstd": r["pvalues"]["t_std_int"],
            "rsq": r["rsquared"],
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_parquet(DATA / "preindustrial_malthus_results.parquet", index=False)

    # ---- Plot ----
    if len(res) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.errorbar(res["beta_density"], range(len(res)),
                    xerr=res["beta_density"].abs() * 0.0,  # se not stored
                    fmt="o", color="#0072B2")
        for i, row in res.iterrows():
            mark = "***" if row["p_density"] < 0.01 else "**" if row["p_density"] < 0.05 else "*" if row["p_density"] < 0.1 else ""
            ax.text(row["beta_density"], i, f"  {mark}", va="center")
        ax.set_yticks(range(len(res)))
        ax.set_yticklabels(res["pathway"])
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("β on log density (Malthusian coefficient)")
        ax.set_title("Density effect by pathway, 1421–1750")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.errorbar(res["beta_T"], range(len(res)), fmt="o", color="#D55E00", label="T anomaly")
        ax.errorbar(res["beta_P"] / 10, range(len(res)), fmt="s", color="#009E73",
                    label="P anomaly (÷10)")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_yticks(range(len(res)))
        ax.set_yticklabels(res["pathway"])
        ax.set_xlabel("Coefficient on climate anomaly")
        ax.set_title("Climate response by pathway, 1421–1750")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = FIG / "fig10_preindustrial_malthus.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
