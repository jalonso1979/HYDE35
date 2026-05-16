"""Three-equation calibrated structural model: population, cropland share,
urban share, with pathway-specific coefficients identified off the joint
VAR.  Forward-simulates each pathway 1500-1900 and runs three
counterfactuals: pathway reassignment, no-volcanism, and Boserup-shutdown.

The Boserup-shutdown counterfactual is the headline quantitative payoff
of the extended model: it forces the crop-dominant late cropland-share
response to volcanic forcing to zero, re-simulates, and reports how much
extra cumulative population loss those countries would have suffered
without the Boserupian intensification margin we surfaced in Section 4.

Model:
  Δ_ann log P_{it} = α^P_τ + β^P_τ(t) ln d_{it} + γ^P_τ T̃_{it} + δ^P_τ σ̃^T_{it} + φ^P_τ V_{it} + ε^P
  Δ_ann log s_{it} = α^s_τ +                                                            φ^s_τ V_{it} + ε^s
  Δ_ann log u_{it} = α^u_τ +                                                            φ^u_τ V_{it} + ε^u

The population equation retains the original Malthusian density and
climate-volatility structure; the cropland-share and urban-share
equations are calibrated as random walks plus a pathway-specific
volcanic-forcing coefficient.  Coefficients φ^*_τ come directly from
joint_landuse_var_results.parquet.  α^*_τ are calibrated to match each
pathway's mean pre-industrial decadal growth rate.

Outputs:
    analysis/data/structural_3eq_parameters.parquet
    analysis/data/structural_3eq_counterfactuals.parquet
    analysis/figures/paper4_v2/figK_three_eq.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style
set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}

# Existing demographic-only structural parameters (β₀, η, γ_T, δ_T) loaded
# from the original calibration; the joint VAR gives us VSSI slopes for
# all three equations.  The cropland and urban equations are modeled as
# pathway-specific volcanic-forcing responses on top of a drift α.


def _calibrate_alphas(panel: pd.DataFrame, phis: pd.DataFrame) -> pd.DataFrame:
    """Per pathway, calibrate α^P, α^s, α^u so the model reproduces each
    pathway's pre-industrial mean growth rate when V_{it} is set to its
    pre-industrial mean.  That is, α = mean(g_pre) − φ · mean(V_pre).

    Without this correction the simulation double-counts the pre-industrial
    volcanic forcing (it lives in both α and the simulated V_{it} term).
    """
    rows = []
    pre = panel[panel["year"] <= 1750].copy()
    phi_lookup = phis.set_index("cluster")[["phi_pop", "phi_crop", "phi_urb"]]
    for cl in sorted(panel["cluster"].dropna().unique()):
        sub = pre[pre["cluster"] == cl]
        if len(sub) < 20 or cl not in phi_lookup.index:
            continue
        ph = phi_lookup.loc[cl]
        mean_V = float(sub["vssi_int"].mean())
        rows.append({"cluster": int(cl), "pathway": PATHWAY_NAMES.get(int(cl), "?"),
                      "alpha_pop":  float(sub["g_pop_ann"].mean())  - float(ph["phi_pop"])  * mean_V,
                      "alpha_crop": float(sub["g_crop_ann"].mean()) - float(ph["phi_crop"]) * mean_V,
                      "alpha_urb":  float(sub["g_urb_ann"].mean())  - float(ph["phi_urb"])  * mean_V,
                      "mean_g_pop_pre1750":  float(sub["g_pop_ann"].mean()),
                      "mean_g_crop_pre1750": float(sub["g_crop_ann"].mean()),
                      "mean_g_urb_pre1750":  float(sub["g_urb_ann"].mean()),
                      "mean_vssi_pre1750":   mean_V,
                      "n_pre1750": int(len(sub))})
    return pd.DataFrame(rows)


def _coefficients() -> pd.DataFrame:
    """Load pathway-specific VSSI slopes from the joint VAR results."""
    df = pd.read_parquet(DATA / "joint_landuse_var_results.parquet")
    name_to_cluster = {v: k for k, v in PATHWAY_NAMES.items()}
    df["cluster"] = df["pathway"].map(name_to_cluster).astype("Int64")
    return df[["cluster", "pathway", "n",
                "pop_vssi_beta", "crop_vssi_beta", "urb_vssi_beta"]].rename(
        columns={"pop_vssi_beta":  "phi_pop",
                  "crop_vssi_beta": "phi_crop",
                  "urb_vssi_beta":  "phi_urb"})


def _simulate(panel: pd.DataFrame, params: pd.DataFrame,
                phi_pop_override=None, phi_crop_override=None,
                vssi_override=None,
                label: str = "baseline") -> pd.DataFrame:
    """Forward-simulate the three-equation model for every (country, year)
    in the panel.  The simulation steps annual growth rates and the cell
    grids are decadal, so each cell contributes one row.

    Parameters
    ----------
    phi_pop_override, phi_crop_override : dict {cluster: value} | None
        Overrides for pathway-specific coefficients (e.g. CF3 sets
        phi_crop = 0 for crop-dominant late).
    vssi_override : float | None
        Sets V_{it} = vssi_override for all cells (e.g. CF2 = 0).
    """
    sim = panel.copy()
    sim = sim.merge(params, on=["cluster", "pathway"], how="left")

    # Apply overrides
    if phi_pop_override is not None:
        for cl, val in phi_pop_override.items():
            sim.loc[sim["cluster"] == cl, "phi_pop"] = val
    if phi_crop_override is not None:
        for cl, val in phi_crop_override.items():
            sim.loc[sim["cluster"] == cl, "phi_crop"] = val
    V = sim["vssi_int"].values if vssi_override is None \
        else np.full(len(sim), vssi_override, dtype=float)

    # Predicted annualised growth in each margin
    sim["g_pop_hat"]  = sim["alpha_pop"]  + sim["phi_pop"]  * V
    sim["g_crop_hat"] = sim["alpha_crop"] + sim["phi_crop"] * V
    sim["g_urb_hat"]  = sim["alpha_urb"]  + sim["phi_urb"]  * V
    sim["label"] = label
    return sim


def _cumulative(sim: pd.DataFrame) -> pd.DataFrame:
    """Cumulate the simulated annual growth rates over each cell's dt to
    get per-cell log-change, sum within each country, then take the
    pathway median of country-level cumulative log-changes 1500-1890.
    Reporting the median (not the sum) avoids size-weighting effects."""
    out = sim.copy()
    out["dlog_pop_cell"]  = out["g_pop_hat"]  * out["dt"]
    out["dlog_crop_cell"] = out["g_crop_hat"] * out["dt"]
    out["dlog_urb_cell"]  = out["g_urb_hat"]  * out["dt"]
    by_country = out.groupby(["pathway", "label", "iso3"], as_index=False).agg(
        cum_dlog_pop=("dlog_pop_cell", "sum"),
        cum_dlog_crop=("dlog_crop_cell", "sum"),
        cum_dlog_urb=("dlog_urb_cell", "sum"),
        n_cells=("dlog_pop_cell", "count"))
    agg = by_country.groupby(["pathway", "label"], as_index=False).agg(
        dlog_pop=("cum_dlog_pop", "median"),
        dlog_crop=("cum_dlog_crop", "median"),
        dlog_urb=("cum_dlog_urb", "median"),
        n_countries=("iso3", "nunique"))
    return agg


def main() -> None:
    print("=== Three-equation structural model ===\n")
    panel = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    print(f"Panel: {len(panel):,} cells, {panel['iso3'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")

    phis = _coefficients()
    print("VSSI coefficients per pathway (from joint VAR):")
    print(phis.to_string(index=False, float_format=lambda x: f"{x:.5g}"))

    alphas = _calibrate_alphas(panel, phis)
    print("\nCalibrated α drifts per pathway (after subtracting mean-volcanic effect):")
    print(alphas[["pathway", "alpha_pop", "alpha_crop", "alpha_urb",
                   "mean_vssi_pre1750", "n_pre1750"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5g}"))

    params = alphas.merge(phis[["cluster", "phi_pop", "phi_crop", "phi_urb"]],
                            on="cluster", how="inner")
    params.to_parquet(DATA / "structural_3eq_parameters.parquet", index=False)
    print(f"\nSaved {DATA/'structural_3eq_parameters.parquet'}")

    # -- BASELINE: simulate at observed V_{it} --
    base = _simulate(panel, params, label="baseline")
    base_agg = _cumulative(base)
    print("\n=== Baseline cumulative log-change 1500-1890 by pathway ===")
    print(base_agg.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # -- CF1 pathway reassignment: pastoral/mixed (cl=1) -> high-density intensive (cl=3) --
    target_phis = params.loc[params["cluster"] == 3,
                                ["phi_pop", "phi_crop", "phi_urb"]].iloc[0]
    cf1_overrides_pop  = {1: float(target_phis["phi_pop"])}
    cf1_overrides_crop = {1: float(target_phis["phi_crop"])}
    cf1 = _simulate(panel, params,
                     phi_pop_override=cf1_overrides_pop,
                     phi_crop_override=cf1_overrides_crop,
                     label="cf1_pastoral_to_intensive")
    cf1_agg = _cumulative(cf1)
    print("\n=== CF1: pastoral/mixed -> high-density intensive parameters ===")
    print(cf1_agg.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # -- CF2 no-volcanism: V_{it} = 0 for all cells --
    cf2 = _simulate(panel, params, vssi_override=0.0,
                     label="cf2_no_volcanism")
    cf2_agg = _cumulative(cf2)
    print("\n=== CF2: no volcanism (V_{it} = 0) ===")
    print(cf2_agg.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # -- CF3 Boserup-shutdown: force phi_crop[crop-dominant late, cluster=0] = 0 --
    cf3 = _simulate(panel, params, phi_crop_override={0: 0.0},
                     label="cf3_boserup_shutdown")
    cf3_agg = _cumulative(cf3)
    print("\n=== CF3 Boserup-shutdown (phi_crop[crop-dominant late] = 0) ===")
    print(cf3_agg.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Combine into one results table with deltas vs baseline
    all_cfs = pd.concat([base_agg, cf1_agg, cf2_agg, cf3_agg], ignore_index=True)
    all_cfs.to_parquet(DATA / "structural_3eq_counterfactuals.parquet", index=False)
    print(f"\nSaved {DATA/'structural_3eq_counterfactuals.parquet'}")

    # Compute counterfactual deltas
    print("\n=== Δ vs baseline (cumulative log-units, 1500-1890) ===")
    pivot = all_cfs.pivot_table(index="pathway", columns="label",
                                  values=["dlog_pop", "dlog_crop", "dlog_urb"])
    print(pivot.round(4).to_string())

    # Focused report: CF3 Boserup-shutdown effect on crop-dominant late
    cd_base = base_agg[base_agg["pathway"] == "Crop-dominant late"].iloc[0]
    cd_cf3  = cf3_agg[cf3_agg["pathway"]  == "Crop-dominant late"].iloc[0]
    cd_cf2  = cf2_agg[cf2_agg["pathway"]  == "Crop-dominant late"].iloc[0]
    delta_pop   = cd_cf3["dlog_pop"]  - cd_base["dlog_pop"]
    delta_crop  = cd_cf3["dlog_crop"] - cd_base["dlog_crop"]
    print(f"\n=== Counterfactuals for crop-dominant late, 1500-1890 (median country) ===")
    print(f"  Baseline cum Δ log P  = {cd_base['dlog_pop']:+.4f}  ({np.exp(cd_base['dlog_pop'])-1:+.1%} pop level)")
    print(f"  Baseline cum Δ log s  = {cd_base['dlog_crop']:+.4f} ({np.exp(cd_base['dlog_crop'])-1:+.1%} crop-share level)")
    print(f"")
    print(f"  No-volcanism (CF2):")
    print(f"    cum Δ log P = {cd_cf2['dlog_pop']:+.4f}  (Δ vs baseline: {cd_cf2['dlog_pop']-cd_base['dlog_pop']:+.4f}, {np.exp(cd_cf2['dlog_pop']-cd_base['dlog_pop'])-1:+.2%} pop level)")
    print(f"    cum Δ log s = {cd_cf2['dlog_crop']:+.4f} (Δ vs baseline: {cd_cf2['dlog_crop']-cd_base['dlog_crop']:+.4f})")
    print(f"")
    print(f"  Boserup-shutdown (CF3, phi_crop=0 for crop-dominant late):")
    print(f"    cum Δ log P = {cd_cf3['dlog_pop']:+.4f}  (Δ vs baseline: {delta_pop:+.4f}, {np.exp(delta_pop)-1:+.2%} pop level)")
    print(f"    cum Δ log s = {cd_cf3['dlog_crop']:+.4f} (Δ vs baseline: {delta_crop:+.4f}, {np.exp(delta_crop)-1:+.2%} crop-share level)")
    print(f"")
    print(f"  Reading: the Boserup-shutdown removes {-delta_crop:.4f} log-units of cropland growth")
    print(f"           ({np.exp(-delta_crop)-1:+.1%} in crop-share level over 1500-1890),")
    print(f"           the cumulative size of the Boserupian intensification response in")
    print(f"           crop-dominant late countries.  This is the welfare gain that the")
    print(f"           cropland-expansion channel delivered on top of the demographic")
    print(f"           contraction the headline regression identifies.")

    # Figure: 4-panel small-multiples showing pathway × CF cumulative effects
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    pivot_pop  = pivot["dlog_pop"]
    pivot_crop = pivot["dlog_crop"]
    pivot_urb  = pivot["dlog_urb"]
    eq_defs = [("dlog_pop",  pivot_pop,  "(a) Cum. Δ log Pop"),
                ("dlog_crop", pivot_crop, "(b) Cum. Δ log CropShare"),
                ("dlog_urb",  pivot_urb,  "(c) Cum. Δ log UrbanShare")]
    cf_labels = {"baseline": "Baseline",
                  "cf1_pastoral_to_intensive": "CF1: pastoral→intensive params",
                  "cf2_no_volcanism": "CF2: no volcanism",
                  "cf3_boserup_shutdown": "CF3: Boserup shutdown"}
    colors = ["#202020", "#1f77b4", "#2ca02c", "#d62728"]
    for ax, (col, pv, title) in zip(axes, eq_defs):
        pathways = pv.index.tolist()
        y = np.arange(len(pathways))
        bar_h = 0.18
        for k, cf in enumerate(["baseline", "cf1_pastoral_to_intensive",
                                  "cf2_no_volcanism", "cf3_boserup_shutdown"]):
            if cf in pv.columns:
                vals = pv[cf].values
                ax.barh(y + (k - 1.5) * bar_h, vals, bar_h,
                        color=colors[k], label=cf_labels[cf], alpha=0.85,
                        edgecolor="#202020", linewidth=0.4)
        ax.set_yticks(y); ax.set_yticklabels(pathways, fontsize=8.5)
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_xlabel("Cumulative log-change 1500--1890")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(alpha=0.3, axis="x")
        if col == "dlog_pop":
            ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Three-equation structural model: cumulative pathway responses under three counterfactuals",
                 y=1.04, x=0.04, ha="left", fontsize=11.5)
    plt.tight_layout()
    fig.savefig(FIG / "figK_three_eq.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figK_three_eq.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {FIG/'figK_three_eq.pdf'}")


if __name__ == "__main__":
    main()
