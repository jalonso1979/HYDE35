"""Joint panel VAR of population, cropland share, and urban share, 1500--1900.

The headline Malthusian regression treats population growth as the dependent
variable and lets land use sit inside the pathway fixed effect. This script
estimates instead a three-equation system

    Δ_ann log Pop_{i,t}        = α^P_i + φ^P_p log Pop_{i,t-1}
                                + φ^P_c log CropShare_{i,t-1}
                                + φ^P_u log UrbanShare_{i,t-1}
                                + γ^P_T T_{it} + γ^P_P P_{it}
                                + δ^P_T σ^T_{it} + β^P V_{it} + ε^P_{it}

    Δ_ann log CropShare_{i,t}  = ... (analogous)

    Δ_ann log UrbanShare_{i,t} = ... (analogous)

estimated by within-country FE OLS with country-clustered SEs. The three
equations trace the three margins of pre-industrial structural response to
climate forcing: the demographic margin (pop), the intensive-agricultural
margin (cropland vs. grazing composition), and the exit-from-agriculture
margin (urbanization, the unified-growth-theory escape channel). V_{it} is
the decade-summed Sigl-Toohey VSSI exposure that identifies all three
equations off exogenous climate variation.

The exercise is the right slot for the joint-modelling question: the volcanic
exercise provides exogenous climate forcing that identifies the two equations
cleanly, and pathway fixed effects partial out the pre-industrial selection
issue that contaminates a pure between-country specification.

Outputs:
    analysis/data/joint_landuse_var_panel.parquet  -- the assembled panel
    analysis/data/joint_landuse_var_results.parquet -- pooled and pathway slopes
    analysis/figures/paper4_v2/figJ_joint_landuse.{pdf,png}
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

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}


def _hyde_wide(path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "gbc2025_7apr_base" / path)
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    df = df.dropna(subset=["ISO1"]).copy()
    df["iso3"] = df["ISO1"].astype(int).map(num_to_iso3)
    df = df.dropna(subset=["iso3"]).copy()
    return df


def _build_panel(year_min: int = 1500, year_max: int = 1900) -> pd.DataFrame:
    pop_w = _hyde_wide("his_crop_4apr2025.csv")  # placeholder, use subpop instead
    # Country-level pop comes from summing subpop or from his_pop... but we
    # have country-level summed pop already inside HYDE 3.5 distributable.
    # Use the existing volcanic panel build which sums subpop.
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
    pop = sub.groupby("iso3", as_index=False)[ycols].sum(min_count=1)

    crop = _hyde_wide("his_crop_4apr2025.csv")
    past = _hyde_wide("his_past_4apr2025.csv")
    crop = crop.groupby("iso3", as_index=False)[ycols].sum(min_count=1)
    past = past.groupby("iso3", as_index=False)[ycols].sum(min_count=1)

    # Country urban population from HYDE country-level txt; columns are year
    # strings without the 'y' prefix, so we map onto the same ycols list.
    urb_raw = pd.read_csv(ROOT / "gbc2025_7apr_base" / "txt" / "urbc_c.txt",
                            sep=r"\s+")
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    urb_raw = urb_raw[urb_raw["region"].astype(str).str.isdigit()].copy()
    urb_raw["iso3"] = urb_raw["region"].astype(int).map(num_to_iso3)
    urb_raw = urb_raw.dropna(subset=["iso3"]).copy()
    # Reshape: detect year cols (integer-castable strings)
    yr_cols = [c for c in urb_raw.columns if c not in ("region", "iso3")
                and c.lstrip("-").isdigit()]
    urb_long = urb_raw.melt(id_vars="iso3", value_vars=yr_cols,
                              var_name="year", value_name="urban_pop")
    urb_long["year"] = urb_long["year"].astype(int)
    urb_long["ycol"] = urb_long["year"].apply(lambda y: f"y{y}")
    urb_long = urb_long[["iso3", "ycol", "urban_pop"]]

    pop_l = pop.melt(id_vars="iso3", var_name="ycol", value_name="pop")
    crop_l = crop.melt(id_vars="iso3", var_name="ycol", value_name="crop_km2")
    past_l = past.melt(id_vars="iso3", var_name="ycol", value_name="past_km2")
    for d, c in [(pop_l, "pop"), (crop_l, "crop_km2"), (past_l, "past_km2")]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    urb_long["urban_pop"] = pd.to_numeric(urb_long["urban_pop"], errors="coerce")
    panel = pop_l.merge(crop_l, on=["iso3", "ycol"], how="inner") \
                 .merge(past_l, on=["iso3", "ycol"], how="inner") \
                 .merge(urb_long, on=["iso3", "ycol"], how="left")
    panel = panel.dropna(subset=["pop", "crop_km2", "past_km2"])
    panel["year"] = panel["ycol"].str.lstrip("y").astype(int)
    panel = panel[(panel["year"] >= year_min) & (panel["year"] <= year_max)
                  & (panel["pop"] > 0)
                  & ((panel["crop_km2"] + panel["past_km2"]) > 0)].copy()
    panel["crop_share"] = panel["crop_km2"] / (panel["crop_km2"] + panel["past_km2"])
    # Urban share: urban_pop / total_pop, with a tiny floor to avoid log(0).
    panel["urban_share"] = (panel["urban_pop"] / panel["pop"]).clip(lower=1e-6)
    panel["log_pop"] = np.log(panel["pop"])
    panel["log_crop_share"] = np.log(panel["crop_share"].clip(1e-4))
    panel["log_urban_share"] = np.log(panel["urban_share"])
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    # Build interval-rate growth (annualised)
    panel["next_year"] = panel.groupby("iso3")["year"].shift(-1)
    panel["next_log_pop"] = panel.groupby("iso3")["log_pop"].shift(-1)
    panel["next_log_crop_share"] = panel.groupby("iso3")["log_crop_share"].shift(-1)
    panel["next_log_urban_share"] = panel.groupby("iso3")["log_urban_share"].shift(-1)
    panel = panel.dropna(subset=["next_year"]).copy()
    panel["dt"] = (panel["next_year"] - panel["year"]).astype(int)
    panel["g_pop_ann"] = (panel["next_log_pop"] - panel["log_pop"]) / panel["dt"]
    panel["g_crop_ann"] = (panel["next_log_crop_share"] - panel["log_crop_share"]) / panel["dt"]
    panel["g_urb_ann"] = (panel["next_log_urban_share"] - panel["log_urban_share"]) / panel["dt"]

    # Attach climate (annual T, P) averaged over the interval
    clim = pd.read_parquet(DATA / "country_climate_1421_2025.parquet")
    clim = clim[["iso3", "year", "t_c", "p_mm"]]
    # Interval-mean T, P and within-interval sigma^T, sigma^P
    rows = []
    for i, r in panel.iterrows():
        sub = clim[(clim["iso3"] == r["iso3"]) &
                    (clim["year"] >= r["year"]) &
                    (clim["year"] < r["next_year"])]
        if len(sub) < 3:
            rows.append({"t_bar": np.nan, "p_bar": np.nan,
                          "t_sd": np.nan, "p_sd": np.nan})
            continue
        rows.append({"t_bar": sub["t_c"].mean(), "p_bar": sub["p_mm"].mean(),
                      "t_sd": sub["t_c"].std(), "p_sd": sub["p_mm"].std()})
    panel = pd.concat([panel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    # Within-country demean (interval-mean climate becomes anomaly)
    for c in ["t_bar", "p_bar", "t_sd", "p_sd"]:
        g = panel.groupby("iso3")[c]
        panel[c + "_dev"] = panel[c] - g.transform("mean")

    # VSSI exposure per interval
    with open(DATA / "eVolv2k_sigl_toohey_2024.tab") as f:
        lines = f.read().splitlines()
    data_start = next(idx + 1 for idx, l in enumerate(lines) if l.startswith("*/"))
    erows = []
    for line in lines[data_start + 1:]:
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < 13: continue
        erows.append({"year": parts[0], "vssi": parts[7]})
    e = pd.DataFrame(erows)
    e["year"] = pd.to_numeric(e["year"], errors="coerce")
    e["vssi"] = pd.to_numeric(e["vssi"], errors="coerce")
    e = e.dropna(subset=["year", "vssi"])
    annual = e.groupby("year", as_index=False)["vssi"].sum()
    annual["year"] = annual["year"].astype(int)
    panel["vssi_int"] = 0.0
    for i, r in panel.iterrows():
        m = (annual["year"] >= r["year"]) & (annual["year"] < r["next_year"])
        if m.any():
            panel.at[i, "vssi_int"] = float(annual.loc[m, "vssi"].sum())

    # Attach pathway
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"]).copy()
    clust["iso3"] = clust["iso3"].astype(str); clust["cluster"] = clust["cluster"].astype(int)
    panel = panel.merge(clust[["iso3", "cluster"]], on="iso3", how="inner")
    panel["pathway"] = panel["cluster"].map(PATHWAY_NAMES)
    return panel


def _run_one(d: pd.DataFrame, lhs: str, controls: list[str]) -> dict:
    """Within-country FE OLS with country-clustered SEs."""
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}


def main() -> None:
    print("=== Joint land-use + population panel VAR, 1500--1900 ===\n")
    panel = _build_panel()
    panel.to_parquet(DATA / "joint_landuse_var_panel.parquet", index=False)
    print(f"Panel: {len(panel):,} (country, interval) cells, "
          f"{panel['iso3'].nunique()} countries, "
          f"{panel['cluster'].nunique()} pathways")
    print(f"Year coverage: {panel['year'].min()}-{panel['next_year'].max()}")

    # Canonical control set: matches joint_var_bootstrap.py and the reported
    # estimating equation (temperature volatility t_sd_dev only; no p_sd_dev).
    controls = ["log_pop", "log_crop_share", "log_urban_share",
                "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]

    eqs = [("Δ_ann log Pop", "g_pop_ann"),
           ("Δ_ann log CropShare", "g_crop_ann"),
           ("Δ_ann log UrbanShare", "g_urb_ann")]
    for name, dep in eqs:
        print(f"\n=== Equation: {name} ~ joint regressors (country FE) ===")
        r = _run_one(panel, dep, controls)
        print(f"N={r['n']}, R²={r['r2']:.4f}")
        for c in controls:
            if c in r["params"]:
                stars = ("***" if r['p'][c] < 0.01 else "**" if r['p'][c] < 0.05
                         else "*" if r['p'][c] < 0.10 else "")
                print(f"  {c:>20}: β = {r['params'][c]:+.6f}  "
                      f"SE = {r['bse'][c]:.6f}  p = {r['p'][c]:.3g} {stars}")

    # --- Pathway-stratified VSSI slopes on each equation ---
    print("\n=== Pathway-stratified VSSI slopes on each equation ===")
    rows = []
    for cl in sorted(panel["cluster"].unique()):
        sub = panel[panel["cluster"] == cl]
        if len(sub) < 20 or sub["iso3"].nunique() < 2:
            continue
        rp = _run_one(sub, "g_pop_ann", controls)
        rc = _run_one(sub, "g_crop_ann", controls)
        ru = _run_one(sub, "g_urb_ann", controls)
        rows.append({
            "pathway": PATHWAY_NAMES[cl], "n": rp["n"],
            "pop_vssi_beta": rp["params"].get("vssi_int", np.nan),
            "pop_vssi_se":   rp["bse"].get("vssi_int", np.nan),
            "pop_vssi_p":    rp["p"].get("vssi_int", np.nan),
            "crop_vssi_beta": rc["params"].get("vssi_int", np.nan),
            "crop_vssi_se":   rc["bse"].get("vssi_int", np.nan),
            "crop_vssi_p":    rc["p"].get("vssi_int", np.nan),
            "urb_vssi_beta": ru["params"].get("vssi_int", np.nan),
            "urb_vssi_se":   ru["bse"].get("vssi_int", np.nan),
            "urb_vssi_p":    ru["p"].get("vssi_int", np.nan),
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    res.to_parquet(DATA / "joint_landuse_var_results.parquet", index=False)

    # --- Figure: pathway slopes on (a) pop, (b) crop share, (c) urban share ---
    if len(res):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
        panel_defs = [
            ("pop_vssi", "(a) Demographic equation",
             r"$\beta$ per Tg of decadal VSSI, log pop growth", "#202020"),
            ("crop_vssi", "(b) Crop-share equation",
             r"$\beta$ per Tg of decadal VSSI, log crop-share growth", "#A02020"),
            ("urb_vssi", "(c) Urban-share equation",
             r"$\beta$ per Tg of decadal VSSI, log urban-share growth", "#0072B2"),
        ]
        for ax, (col, title, xlab, c) in zip(axes, panel_defs):
            order = res.sort_values(f"{col}_beta").reset_index(drop=True)
            y = np.arange(len(order))
            ax.errorbar(order[f"{col}_beta"], y,
                        xerr=1.96 * order[f"{col}_se"],
                        fmt="o", color=c, markerfacecolor="white",
                        markeredgewidth=1, ecolor=c, elinewidth=0.7,
                        capsize=2.5)
            for i, r in order.iterrows():
                s = ("***" if r[f"{col}_p"] < 0.01 else "**" if r[f"{col}_p"] < 0.05
                     else "*" if r[f"{col}_p"] < 0.10 else "")
                ax.text(r[f"{col}_beta"], i + 0.18,
                        f"$N={int(r['n'])}$  {s}", ha="center", fontsize=8.5)
            ax.axvline(0, color="#404040", linewidth=0.6)
            ax.set_yticks(y); ax.set_yticklabels(order["pathway"])
            ax.set_xlabel(xlab)
            ax.set_title(title, loc="left", fontsize=10.5)
            ax.grid(alpha=0.3)

        fig.suptitle("Joint three-equation reduced-form panel system: response to volcanic forcing by pathway, 1500--1900",
                     y=1.04, x=0.04, ha="left", fontsize=12)
        plt.tight_layout()
        fig.savefig(FIG / "figJ_joint_landuse.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figJ_joint_landuse.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"\nSaved {FIG/'figJ_joint_landuse.pdf'}")


if __name__ == "__main__":
    main()
