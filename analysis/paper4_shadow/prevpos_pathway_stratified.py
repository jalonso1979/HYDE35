"""Section 4.3.3 — Pathway-stratified preventive/positive check.

For each pathway p in {crop-dominant late, pastoral/mixed late, irrigation
pioneer, high-density intensive, early extensifiers}, run the four prevpos
regressions on the within-pathway sub-sample of the 1751-1900 annual European
panel. Output one parquet of per-pathway-per-outcome coefficients and one
figure of the headline T-anomaly coefficient on Δlog m0 by pathway.
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


def main() -> None:
    panel = pd.read_parquet(DATA / "prevpos_panel.parquet")
    clust = pd.read_parquet(DATA / "paper1_clustered_features.parquet")
    clust = clust.dropna(subset=["iso3", "cluster"])[["iso3", "cluster"]]
    clust["cluster"] = clust["cluster"].astype(int)
    panel = panel.merge(clust, on="iso3", how="inner")
    panel["pathway"] = panel["cluster"].map(PATHWAY_NAMES)

    controls = ["t_anom", "p_anom", "t_roll_sd"]
    outcomes = ["d_log_fert", "d_log_m0", "d_log_m5", "d_log_m10"]

    rows = []
    for cl, name in PATHWAY_NAMES.items():
        sub = panel[panel["cluster"] == cl]
        if sub["iso3"].nunique() < 2 or len(sub) < 30:
            print(f"  skip {name}: countries={sub['iso3'].nunique()}, N={len(sub)}")
            continue
        for lhs in outcomes:
            d = sub.dropna(subset=[lhs] + controls).copy()
            if len(d) < 30 or d["iso3"].nunique() < 2:
                continue
            g = d.groupby("iso3")
            for c in controls + [lhs]:
                d[c] = d[c] - g[c].transform("mean")
            d["trend"] = d["year"] - d.groupby("iso3")["year"].transform("mean")
            X = sm.add_constant(d[controls + ["trend"]])
            try:
                res = sm.OLS(d[lhs], X).fit(cov_type="cluster", cov_kwds={"groups": d["iso3"]})
            except Exception as e:
                print(f"  {name}/{lhs} regression failed: {e}")
                continue
            for c in controls:
                rows.append({"pathway": name, "outcome": lhs, "regressor": c,
                              "beta": res.params.get(c, np.nan),
                              "se": res.bse.get(c, np.nan),
                              "p": res.pvalues.get(c, np.nan),
                              "n": int(res.nobs),
                              "n_countries": d["iso3"].nunique()})
            print(f"  {name} / {lhs}: N={int(res.nobs)}, "
                  f"β_T={res.params.get('t_anom', np.nan):+.4f} "
                  f"(p={res.pvalues.get('t_anom', np.nan):.3g})")

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "prevpos_pathway.parquet", index=False)
    print(f"\nSaved {DATA/'prevpos_pathway.parquet'}: {len(out)} rows")

    # Figure: per-pathway T anomaly coefficient on m0
    m0 = out[(out["outcome"] == "d_log_m0") & (out["regressor"] == "t_anom")]
    if len(m0) >= 2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        m0 = m0.sort_values("beta").reset_index(drop=True)
        y = np.arange(len(m0))
        ax.errorbar(m0["beta"], y, xerr=1.96 * m0["se"], fmt="o",
                    color="#202020", markerfacecolor="white", markeredgewidth=1.0,
                    ecolor="#404040", elinewidth=0.8, capsize=2.5)
        for i, r in m0.iterrows():
            s = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
                 else "*" if r["p"] < 0.10 else "")
            ax.text(r["beta"], i + 0.18, f"$N={int(r['n'])}$  {s}",
                    ha="center", fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels(m0["pathway"])
        ax.axvline(0, color="#404040", linewidth=0.6)
        ax.set_xlabel(r"Coefficient on $T$ anomaly ($\Delta \log m_0$, per °C)")
        ax.set_title("Pathway-stratified summer-mortality coefficient",
                     loc="left", fontsize=10.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG / "figK_prevpos_pathway.pdf", bbox_inches="tight")
        fig.savefig(FIG / "figK_prevpos_pathway.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"Saved {FIG/'figK_prevpos_pathway.pdf'}")


if __name__ == "__main__":
    main()
