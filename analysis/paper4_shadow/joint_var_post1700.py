"""Joint VAR robustness exercise (c): drop pre-1700 century-resolution HYDE
cells and re-run the three-equation joint VAR.

HYDE 3.5 uses 100-year intervals before 1700 and decadal afterwards, so the
1500 and 1600 cells in the joint VAR panel represent century-aggregated
demographic and land-use back-projections from a much sparser historical
source base than the decadal cells from 1700 onward.  HYDE's pre-1850
population back-projection also uses cropland as an input variable, so the
pre-1700 portion of the joint cropland-and-population panel is partly
mechanical.

This script restricts the joint VAR sample to year >= 1700 (drops the 1500
and 1600 century cells) and re-runs the pathway-stratified specification.
We also report a year >= 1750 cutoff for comparison.

Outputs:
    analysis/data/joint_var_post1700_results.parquet
    analysis/figures/paper4_v2/figJ_post1700.{pdf,png}
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

CONTROLS = ["log_pop", "log_crop_share", "log_urban_share",
            "t_bar_dev", "p_bar_dev", "t_sd_dev", "vssi_int"]


def _run_one(d: pd.DataFrame, lhs: str, controls: list[str]) -> dict:
    d = d.dropna(subset=[lhs] + controls + ["iso3"]).copy()
    if len(d) < 20 or d["iso3"].nunique() < 2:
        return None
    g = d.groupby("iso3")
    for c in [lhs] + controls:
        d[c] = d[c] - g[c].transform("mean")
    X = sm.add_constant(d[controls])
    res = sm.OLS(d[lhs], X).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["iso3"]})
    return {"params": res.params, "bse": res.bse, "p": res.pvalues,
            "n": int(res.nobs), "r2": float(res.rsquared)}


def _stratified(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"\n=== {label}: pathway-stratified VSSI slopes ===")
    rows = []
    for cl in sorted(panel["cluster"].dropna().unique()):
        sub = panel[panel["cluster"] == cl]
        if len(sub) < 20 or sub["iso3"].nunique() < 2:
            print(f"  Cluster {cl} ({PATHWAY_NAMES.get(cl, '?')}): N<20 or <2 countries, skip")
            continue
        rp = _run_one(sub, "g_pop_ann", CONTROLS)
        rc = _run_one(sub, "g_crop_ann", CONTROLS)
        ru = _run_one(sub, "g_urb_ann", CONTROLS)
        if rp is None or rc is None or ru is None:
            continue
        rows.append({"sample": label, "cluster": int(cl),
                      "pathway": PATHWAY_NAMES.get(cl, "?"),
                      "n": rp["n"], "n_countries": sub["iso3"].nunique(),
                      "pop_beta": rp["params"].get("vssi_int", np.nan),
                      "pop_se":   rp["bse"].get("vssi_int", np.nan),
                      "pop_p":    rp["p"].get("vssi_int", np.nan),
                      "crop_beta": rc["params"].get("vssi_int", np.nan),
                      "crop_se":   rc["bse"].get("vssi_int", np.nan),
                      "crop_p":    rc["p"].get("vssi_int", np.nan),
                      "urb_beta": ru["params"].get("vssi_int", np.nan),
                      "urb_se":   ru["bse"].get("vssi_int", np.nan),
                      "urb_p":    ru["p"].get("vssi_int", np.nan)})
    out = pd.DataFrame(rows)
    if len(out):
        cols = ["pathway", "n", "n_countries",
                "pop_beta", "pop_p", "crop_beta", "crop_p", "urb_beta", "urb_p"]
        print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return out


def _pooled(panel: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: pooled joint VAR ===")
    for tag, lhs in [("Δ_ann log Pop", "g_pop_ann"),
                       ("Δ_ann log CropShare", "g_crop_ann"),
                       ("Δ_ann log UrbanShare", "g_urb_ann")]:
        r = _run_one(panel, lhs, CONTROLS)
        if r is None: continue
        b = r["params"].get("vssi_int", np.nan)
        se = r["bse"].get("vssi_int", np.nan)
        p = r["p"].get("vssi_int", np.nan)
        print(f"  {tag}: VSSI β={b:+.6f} (SE {se:.6f}) p={p:.3g}, "
              f"R²={r['r2']:.4f}, N={r['n']}")


def main() -> None:
    print("=== Joint VAR robustness (c): drop pre-1700 century cells ===\n")

    panel = pd.read_parquet(DATA / "joint_landuse_var_panel.parquet")
    print(f"Full panel: {len(panel):,} rows, {panel['iso3'].nunique()} countries")
    print(f"  year range: {panel['year'].min()}-{panel['year'].max()}")
    print(f"  pre-1700 cells (century resolution): "
          f"{(panel['year'] < 1700).sum():,}")
    print(f"  post-1700 cells (decadal): {(panel['year'] >= 1700).sum():,}")

    samples = [
        ("1500--1900 (full)", panel),
        ("1700--1900 (drop century cells)", panel[panel["year"] >= 1700]),
        ("1750--1900 (drop pre-W-S era)", panel[panel["year"] >= 1750]),
        ("1800--1900 (modern instrumental era)", panel[panel["year"] >= 1800]),
    ]

    all_rows = []
    for label, sub in samples:
        print(f"\n--- {label}: {len(sub):,} rows, {sub['iso3'].nunique()} countries ---")
        _pooled(sub, label)
        out = _stratified(sub, label)
        all_rows.append(out)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_parquet(DATA / "joint_var_post1700_results.parquet", index=False)
    print(f"\nSaved {DATA/'joint_var_post1700_results.parquet'}")

    # Compare full vs post-1700 in one summary table
    print("\n=== Summary: pathway × equation coefficient sign and significance ===")
    summary_rows = []
    for cl in sorted(panel["cluster"].dropna().unique()):
        path = PATHWAY_NAMES.get(cl, "?")
        for sample_label, _ in samples:
            sub = combined[(combined["pathway"] == path)
                            & (combined["sample"] == sample_label)]
            if len(sub) == 0: continue
            r = sub.iloc[0]
            def _sig(p):
                return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
            summary_rows.append({
                "pathway": path, "sample": sample_label, "n": int(r["n"]),
                "pop": f"{r['pop_beta']*1e5:+.2f}×10⁻⁵ {_sig(r['pop_p'])}",
                "crop": f"{r['crop_beta']*1e5:+.2f}×10⁻⁵ {_sig(r['crop_p'])}",
                "urb": f"{r['urb_beta']*1e5:+.2f}×10⁻⁵ {_sig(r['urb_p'])}",
            })
    sdf = pd.DataFrame(summary_rows)
    print(sdf.to_string(index=False))

    # Figure: forest plot comparing full vs post-1700
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    eq_defs = [("pop", "(a) Δ log Pop", "#202020"),
                ("crop", "(b) Δ log CropShare", "#A02020"),
                ("urb", "(c) Δ log UrbanShare", "#0072B2")]
    # Two subsamples per pathway, side by side
    sub_labels = ["1500--1900 (full)", "1700--1900 (drop century cells)"]
    for ax, (eq, title, color) in zip(axes, eq_defs):
        d = combined[combined["sample"].isin(sub_labels)].copy()
        pathways = sorted(d["pathway"].unique())
        y_positions = []
        labels = []
        for i, path in enumerate(pathways):
            for j, samp in enumerate(sub_labels):
                row = d[(d["pathway"] == path) & (d["sample"] == samp)]
                if len(row) == 0: continue
                row = row.iloc[0]
                y = i * 2 + j * 0.4
                y_positions.append(y)
                labels.append(f"{path[:18]}\n{samp[:18]}")
                marker = "o" if samp == sub_labels[0] else "s"
                ax.errorbar(row[f"{eq}_beta"], y,
                            xerr=1.96 * row[f"{eq}_se"], fmt=marker,
                            color=color, markerfacecolor="white" if samp == sub_labels[0] else color,
                            markeredgewidth=1, ecolor=color, elinewidth=0.7,
                            capsize=2.5)
                s = ("***" if row[f"{eq}_p"] < 0.01 else
                     "**" if row[f"{eq}_p"] < 0.05 else
                     "*" if row[f"{eq}_p"] < 0.10 else "")
                if s:
                    ax.text(row[f"{eq}_beta"], y + 0.15, s, ha="center",
                            fontsize=9)
        ax.set_yticks(y_positions); ax.set_yticklabels(labels, fontsize=7.5)
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_xlabel(r"$\beta$ on VSSI per Tg")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Joint VAR pathway coefficients: full panel vs. post-1700 (drop century cells)",
                 y=1.02, x=0.04, ha="left", fontsize=11.5)
    plt.tight_layout()
    fig.savefig(FIG / "figJ_post1700.pdf", bbox_inches="tight")
    fig.savefig(FIG / "figJ_post1700.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\nSaved {FIG/'figJ_post1700.pdf'}")


if __name__ == "__main__":
    main()
