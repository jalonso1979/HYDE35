"""Three structural counterfactuals on the calibrated quantitative model.

(i)  Pathway reassignment: take each pastoral/mixed-late country and re-run
     dynamics with high-density intensive parameters. Quantifies the
     "what if Russia had been Japan" question.
(ii) No volcanism: set σ_T contributions from intervals containing a major
     eruption (Sigl-Toohey VSSI ≥ 5 Tg) to zero. Quantifies cumulative
     demographic effect of 1500-1900 volcanism by pathway.
(iii) Climate decoupling: progressively shrink within-interval σ_T toward
     zero and recompute cumulative growth. Quantifies how much pre-industrial
     volatility cost / gave in each pathway.
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figstyle import set_style, gray_palette
from structural_simulate import simulate, load_panel_and_params, PATHWAY_NAMES

set_style()

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
FIG = ROOT / "analysis" / "figures" / "paper4_v2"


def get_volcanic_years() -> set:
    """Years with Sigl-Toohey VSSI ≥ 5 Tg, 1500-1900."""
    sigl = pd.read_csv(DATA / "eVolv2k_sigl_toohey_2024.tab", sep="\t",
                        skiprows=48, header=None,
                        usecols=[0, 7], names=["year", "vssi"],
                        on_bad_lines="skip", engine="python")
    sigl["year"] = pd.to_numeric(sigl["year"], errors="coerce")
    sigl["vssi"] = pd.to_numeric(sigl["vssi"], errors="coerce")
    sigl = sigl.dropna()
    big = sigl[(sigl["vssi"] >= 5.0) & (sigl["year"].between(1500, 1900))]
    return set(int(y) for y in big["year"])


def main() -> None:
    panel, params, _ = load_panel_and_params()

    # ── Counterfactual 1: pathway reassignment ─────────────────────────
    print("=== CF1: reassign pastoral/mixed countries to high-density intensive ===")
    pastoral_isos = panel[panel["cluster"] == 1]["iso3"].unique()
    override = {iso: 3 for iso in pastoral_isos}
    sim_base = simulate(panel, params, scenario="baseline")
    sim_cf1 = simulate(panel, params, scenario="cf_intensive_reassign",
                        counterfactual={"pathway_override": override})

    # Compare cumulative growth deviations 1421-1750 for pastoral countries
    pre_final_base = (sim_base[sim_base["year"] < 1750]
                       .sort_values(["iso3", "year"])
                       .groupby("iso3").tail(1)[["iso3", "cum_growth_sim_dev"]]
                       .rename(columns={"cum_growth_sim_dev": "cum_base"}))
    pre_final_cf1 = (sim_cf1[sim_cf1["year"] < 1750]
                      .sort_values(["iso3", "year"])
                      .groupby("iso3").tail(1)[["iso3", "cum_growth_sim_dev"]]
                      .rename(columns={"cum_growth_sim_dev": "cum_cf"}))
    cmp = pre_final_base.merge(pre_final_cf1, on="iso3")
    cmp["cum_diff"] = cmp["cum_cf"] - cmp["cum_base"]
    cmp = cmp[cmp["iso3"].isin(pastoral_isos)]
    print(f"  Pastoral countries N = {len(cmp)}")
    print(f"  Cumulative-growth lift if reassigned to intensive:")
    print(f"    median: {cmp['cum_diff'].median():+.4f} log-units (≈ "
          f"{(np.exp(cmp['cum_diff'].median()) - 1)*100:+.1f}% population)")
    print(f"    IQR:    [{cmp['cum_diff'].quantile(0.25):+.4f}, "
          f"{cmp['cum_diff'].quantile(0.75):+.4f}]")

    # ── Counterfactual 2: no volcanism ─────────────────────────────────
    print("\n=== CF2: no volcanism 1500-1900 (zero σ_T contribution in eruption years) ===")
    volc = get_volcanic_years()
    print(f"  Identified {len(volc)} eruption years with VSSI≥5 Tg in 1500-1900")
    sim_cf2 = simulate(panel, params, scenario="cf_no_volc",
                        counterfactual={"zero_volc_years": volc})
    final_base = (sim_base[sim_base["year"].between(1500, 1900)]
                   .sort_values(["iso3", "year"]).groupby("iso3").tail(1)
                   [["iso3", "cum_growth_sim_dev"]]
                   .rename(columns={"cum_growth_sim_dev": "cum_base"}))
    final_cf2 = (sim_cf2[sim_cf2["year"].between(1500, 1900)]
                  .sort_values(["iso3", "year"]).groupby("iso3").tail(1)
                  [["iso3", "cum_growth_sim_dev"]]
                  .rename(columns={"cum_growth_sim_dev": "cum_cf"}))
    cmp2 = final_base.merge(final_cf2, on="iso3")
    cmp2["cum_diff"] = cmp2["cum_cf"] - cmp2["cum_base"]
    cmp2 = cmp2.merge(panel[["iso3", "cluster"]].drop_duplicates(), on="iso3")
    print(f"  Cumulative growth gained by removing volcanic years, by pathway:")
    for cl, g in cmp2.groupby("cluster"):
        if cl not in PATHWAY_NAMES: continue
        m = g["cum_diff"].median()
        print(f"    {PATHWAY_NAMES[cl]:<25s} N={len(g):>3}  median "
              f"Δ = {m:+.4f} log-units ({(np.exp(m)-1)*100:+.2f}% pop)")

    # ── Counterfactual 3: climate decoupling ───────────────────────────
    print("\n=== CF3: climate decoupling (shrink σ_T toward zero) ===")
    results_cf3 = []
    for w in [1.0, 0.75, 0.5, 0.25, 0.0]:
        s = simulate(panel, params,
                      scenario=f"cf_decoupl_{w}",
                      counterfactual={"climate_decoupl": w})
        f = (s[s["year"] < 1900].sort_values(["iso3", "year"])
              .groupby("iso3").tail(1)[["iso3", "cum_growth_sim_dev"]])
        f["weight"] = w
        results_cf3.append(f)
    cf3 = pd.concat(results_cf3, ignore_index=True)
    cf3 = cf3.merge(panel[["iso3", "cluster"]].drop_duplicates(), on="iso3")
    print(f"  Mean cumulative simulated growth deviation by σ_T weight:")
    for w, g in cf3.groupby("weight"):
        for cl in sorted(g["cluster"].unique()):
            if cl not in PATHWAY_NAMES: continue
            sub = g[g["cluster"] == cl]
            print(f"    weight={w:.2f}  {PATHWAY_NAMES[cl]:<25s} "
                  f"mean cum dev = {sub['cum_growth_sim_dev'].mean():+.5f}")

    # Save and figure
    sim_base.to_parquet(DATA / "structural_sim_baseline.parquet", index=False)
    sim_cf1.to_parquet(DATA / "structural_sim_cf_intensive.parquet", index=False)
    sim_cf2.to_parquet(DATA / "structural_sim_cf_no_volc.parquet", index=False)
    cf3.to_parquet(DATA / "structural_sim_cf_decoupling.parquet", index=False)

    # Make CF figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    ax = axes[0]
    pal = gray_palette(4)
    for i, (cl, g) in enumerate(cmp2.groupby("cluster")):
        if cl not in PATHWAY_NAMES: continue
        ax.boxplot(g["cum_diff"].values, positions=[i], widths=0.6,
                    showmeans=True, patch_artist=True,
                    boxprops=dict(facecolor=pal[i], edgecolor="#202020", linewidth=0.5),
                    medianprops=dict(color="#202020", linewidth=0.7),
                    whiskerprops=dict(color="#202020", linewidth=0.6),
                    capprops=dict(color="#202020", linewidth=0.6),
                    flierprops=dict(marker="+", markersize=3, markeredgecolor="#606060"),
                    meanprops=dict(marker="o", markerfacecolor="white",
                                    markeredgecolor="#202020", markersize=4))
    ax.set_xticks(range(len([c for c in cmp2["cluster"].unique() if c in PATHWAY_NAMES])))
    ax.set_xticklabels([PATHWAY_NAMES[c].split()[0]
                        for c in sorted(cmp2["cluster"].unique()) if c in PATHWAY_NAMES],
                       rotation=10, ha="right", fontsize=8)
    ax.axhline(0, color="#404040", linewidth=0.6)
    ax.set_ylabel("Cumulative growth gained\nfrom removing volcanism")
    ax.set_title("(a) CF2: no 1500–1900 volcanism", loc="left", fontsize=10)

    ax = axes[1]
    weights = [1.0, 0.75, 0.5, 0.25, 0.0]
    for i, cl in enumerate(sorted(set(cf3["cluster"]) & set(PATHWAY_NAMES))):
        means = [cf3[(cf3["cluster"] == cl) & (cf3["weight"] == w)]
                  ["cum_growth_sim_dev"].mean() for w in weights]
        ax.plot(weights, means, marker="o", linewidth=1.0,
                color=pal[i], label=PATHWAY_NAMES[cl].split()[0])
    ax.set_xlabel("σ_T weight (1=baseline, 0=full decoupling)")
    ax.set_ylabel("Mean cumulative growth deviation")
    ax.set_title("(b) CF3: climate decoupling", loc="left", fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(FIG / "fig14_counterfactuals.pdf")
    fig.savefig(FIG / "fig14_counterfactuals.png")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig14_counterfactuals.pdf'}")


if __name__ == "__main__":
    main()
