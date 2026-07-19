"""Dynamic event-study for the Sigl-Toohey volcanic forcing.

Currently the Sigl regression projects pop growth in interval t on VSSI
exposure in the same interval. A standard top-five-grade presentation
adds leads and lags: project pop growth at decade t on VSSI at decades
[t-2, t-1, t, t+1, t+2], producing an impulse-response function (IRF)
across event time. Pre-event coefficients test parallel trends; post-
event coefficients trace the demographic response trajectory.

We run two variants:
  1. Pooled (one IRF for all pathways)
  2. Per-pathway IRFs, contrasted

Output: structural IRF table + figure with pathway-by-pathway lead-lag plot.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette, LINESTYLES

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"

PATHWAY_NAMES = {0: "Crop-dominant late", 1: "Pastoral/mixed late",
                 2: "Irrigation pioneer", 3: "High-density intensive",
                 4: "Early extensifiers"}

LEADS_LAGS = [-2, -1, 0, 1, 2]   # decades relative to event


def _build_irf_panel() -> pd.DataFrame:
    """For each (country, decade), attach VSSI exposures at h = -2..+2."""
    sigl_panel = pd.read_parquet(DATA / "sigl_volcanic_panel.parquet")
    # Build decade-level VSSI series per country (already present as vssi_int)
    p = sigl_panel.sort_values(["iso3", "year"]).copy()
    # Horizon-h response to an eruption requires VSSI from h periods earlier:
    # vssi_h{+h} must hold V_{t-h}, i.e. shift(+h). (Negative h => lead/pre-event.)
    for h in LEADS_LAGS:
        p[f"vssi_h{h:+d}"] = p.groupby("iso3")["vssi_int"].shift(h)
    p = p.dropna(subset=[f"vssi_h{h:+d}" for h in LEADS_LAGS])
    return p


def estimate_irf(df: pd.DataFrame, cluster_filter: int | None = None) -> pd.DataFrame:
    if cluster_filter is not None:
        df = df[df["cluster"] == cluster_filter].copy()
    if len(df) < 30:
        return pd.DataFrame()
    # Within-country demean
    g = df.groupby("iso3")
    df["pop_growth_ann_w"] = df["pop_growth_ann"] - g["pop_growth_ann"].transform("mean")
    df["t_anom_int_w"]    = df["t_anom_int"]    - g["t_anom_int"].transform("mean")
    for h in LEADS_LAGS:
        col = f"vssi_h{h:+d}"
        df[col + "_w"] = df[col] - g[col].transform("mean")
    X = sm.add_constant(df[[f"vssi_h{h:+d}_w" for h in LEADS_LAGS] + ["t_anom_int_w"]])
    y = df["pop_growth_ann_w"]
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["iso3"].values})
    rows = []
    for h in LEADS_LAGS:
        col = f"vssi_h{h:+d}_w"
        rows.append({
            "lag": h,
            "beta": r.params[col],
            "se": r.bse[col],
            "p": r.pvalues[col],
        })
    out = pd.DataFrame(rows)
    out["ci_lo"] = out["beta"] - 1.96 * out["se"]
    out["ci_hi"] = out["beta"] + 1.96 * out["se"]
    out["N"] = int(r.nobs)
    return out


def main() -> None:
    p = _build_irf_panel()
    print(f"IRF panel: {len(p):,} cells, "
          f"{p['iso3'].nunique()} countries, "
          f"{p['cluster'].nunique()} pathways")

    print("\n=== Pooled IRF (all pathways, country FE) ===")
    pooled = estimate_irf(p)
    print(pooled.round(7).to_string(index=False))

    pooled.to_parquet(DATA / "sigl_irf_pooled.parquet", index=False)

    irfs_path = {}
    print("\n=== Per-pathway IRFs ===")
    for cl in sorted(p["cluster"].unique()):
        if cl not in PATHWAY_NAMES: continue
        irf = estimate_irf(p, cluster_filter=cl)
        if len(irf) == 0: continue
        irf["pathway"] = PATHWAY_NAMES[cl]
        irf["cluster"] = cl
        irfs_path[cl] = irf
        print(f"\n  {PATHWAY_NAMES[cl]}:")
        print(irf.round(7).to_string(index=False))
    all_path = pd.concat(list(irfs_path.values()), ignore_index=True)
    all_path.to_parquet(DATA / "sigl_irf_pathway.parquet", index=False)

    # Parallel-trends test: F-test that lead coefficients (h<0) are jointly zero
    print("\n=== Parallel-trends F-test (pooled, lead coefficients h=-2,-1) ===")
    g = p.groupby("iso3")
    pw = p.copy()
    for c in ["pop_growth_ann"] + [f"vssi_h{h:+d}" for h in LEADS_LAGS] + ["t_anom_int"]:
        pw[c + "_w"] = pw[c] - g[c].transform("mean")
    X = sm.add_constant(pw[[f"vssi_h{h:+d}_w" for h in LEADS_LAGS] + ["t_anom_int_w"]])
    y = pw["pop_growth_ann_w"]
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pw["iso3"].values})
    lead_cols = [f"vssi_h-2_w", f"vssi_h-1_w"]
    f_stat = r.f_test([f"{c} = 0" for c in lead_cols])
    print(f"  F = {float(f_stat.fvalue):.3f}, p = {float(f_stat.pvalue):.4g}, "
          f"df = ({lead_cols.__len__()}, {int(r.df_resid)})")
    print(f"  → {'PASS' if float(f_stat.pvalue) > 0.10 else 'FAIL'} parallel trends "
          f"(want p > 0.10)")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    ax.errorbar(pooled["lag"], pooled["beta"] * 1e4,
                yerr=1.96 * pooled["se"] * 1e4,
                fmt="o-", color="#202020", markerfacecolor="white",
                markeredgewidth=1, ecolor="#606060", linewidth=1.0, capsize=2)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.axvline(0, color="#606060", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Decade relative to VSSI exposure")
    ax.set_ylabel(r"$\hat{\beta}$ × 10$^4$ on pop growth")
    ax.set_title("(a) Pooled IRF, all pathways", loc="left")
    ax.set_xticks(LEADS_LAGS)

    ax = axes[1]
    pal = gray_palette(len(irfs_path))
    for i, (cl, irf) in enumerate(sorted(irfs_path.items())):
        ax.errorbar(irf["lag"], irf["beta"] * 1e4,
                    yerr=1.96 * irf["se"] * 1e4,
                    fmt="o-", color=pal[i], linewidth=1.0,
                    markersize=4, capsize=2,
                    label=PATHWAY_NAMES[cl].split()[0], alpha=0.85,
                    linestyle=LINESTYLES[i % len(LINESTYLES)])
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.axvline(0, color="#606060", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Decade relative to VSSI exposure")
    ax.set_ylabel(r"$\hat{\beta}$ × 10$^4$")
    ax.set_title("(b) Per-pathway IRF", loc="left")
    ax.set_xticks(LEADS_LAGS)
    ax.legend(fontsize=7.5, loc="best", ncol=2)

    plt.tight_layout()
    fig.savefig(FIG / "fig15_sigl_irf.pdf")
    fig.savefig(FIG / "fig15_sigl_irf.png")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig15_sigl_irf.pdf'}")


if __name__ == "__main__":
    main()
