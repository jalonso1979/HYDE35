"""Exercise 3 (new): H_pred coefficient-evolution test.

For the FWER-surviving outcomes where H_pred was a significant predictor in
the v2 paper (log GDPpc, log pop growth), document how the H_pred coefficient
evolves as the new modern-genetics substrates are progressively added to the
regression.

Three specifications per outcome:
  (1) Baseline: y ~ H_pred_pwadj + geography controls
  (2) + Neolithic fraction
  (3) + functional-allele bundle (all 8 SNPs)

The empirical question: does H_pred's signal collapse globally (extending the
§6.7 Lazaridis Eurasian-sub-sample result to all 162 countries)?

Output: analysis/data/deep_determinants/h_pred_evolution.parquet
        analysis/figures/paper5_horserace/tab_h_pred_evolution.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper5_horserace.exercise1_shapley import (
    CLIMATE_BUNDLE,
    FUNCTIONAL_BUNDLE,
    CONTROLS,
)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/h_pred_evolution.parquet"
TEX = ROOT / "analysis/figures/paper5_horserace/tab_h_pred_evolution.tex"

TARGET_OUTCOMES = ["log_gdppc_2015", "log_pop_growth_1950_2025"]


def _fit(df: pd.DataFrame, y_col: str, regressors: list[str]) -> dict:
    needed = [y_col, "H_pred_pwadj"] + regressors
    sub = df.dropna(subset=needed).copy()
    X = sm.add_constant(sub[["H_pred_pwadj"] + regressors])
    res = sm.OLS(sub[y_col], X).fit(cov_type="HC3")
    return {
        "coef": float(res.params["H_pred_pwadj"]),
        "se": float(res.bse["H_pred_pwadj"]),
        "t": float(res.tvalues["H_pred_pwadj"]),
        "p": float(res.pvalues["H_pred_pwadj"]),
        "n": int(res.nobs),
        "r2": float(res.rsquared),
    }


def main() -> None:
    df = pd.read_parquet(PANEL)

    specs = {
        "baseline": [],
        "plus_neolithic": ["neolithic_frac"],
        "plus_functional": ["neolithic_frac"] + list(FUNCTIONAL_BUNDLE),
    }

    rows = []
    for outcome in TARGET_OUTCOMES:
        print(f"\n=== {outcome} ===")
        for spec_name, extra in specs.items():
            r = _fit(df, outcome, CONTROLS + extra)
            rows.append({"outcome": outcome, "spec": spec_name, **r})
            stars = ("***" if r["p"] < 0.01
                     else "**" if r["p"] < 0.05
                     else "*" if r["p"] < 0.10 else "")
            print(f"  {spec_name:18s} β_H={r['coef']:+8.3f} (se {r['se']:.2f}) "
                  f"|t|={abs(r['t']):.2f}{stars}  n={r['n']}, R²={r['r2']:.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")

    # LaTeX table
    spec_labels = {"baseline": "Baseline",
                   "plus_neolithic": "+ Neolithic frac",
                   "plus_functional": "+ Functional alleles"}
    outcome_labels = {
        "log_gdppc_2015": r"$\ln\text{GDPpc}_{2015}$",
        "log_pop_growth_1950_2025": r"$\Delta\ln\!P_{1950\to2025}$",
    }

    with open(TEX, "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Specification & $\\hat\\beta_{H}$ & SE & $|t|$ & $N$ \\\\\n")
        for outcome in TARGET_OUTCOMES:
            f.write("\\midrule\n")
            f.write("\\multicolumn{5}{l}{\\textit{" + outcome_labels[outcome] + "}} \\\\\n")
            for spec in ["baseline", "plus_neolithic", "plus_functional"]:
                row = out_df[(out_df["outcome"] == outcome) & (out_df["spec"] == spec)].iloc[0]
                stars = ("$^{***}$" if row["p"] < 0.01
                         else "$^{**}$" if row["p"] < 0.05
                         else "$^{*}$" if row["p"] < 0.10 else "")
                f.write(f"\\quad {spec_labels[spec]} & {row['coef']:+.2f}{stars} & "
                        f"{row['se']:.2f} & {abs(row['t']):.2f} & {int(row['n'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {TEX}")


if __name__ == "__main__":
    main()
