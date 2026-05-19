"""Exercise 1c: Within-functional-bundle Shapley for the 8 SNPs.

For each outcome, decompose the functional-allele bundle's joint contribution
across (LCT, ADH1B, AMY1, EDAR, DARC, SLC24A5, HBB, FADS), conditional on the
other 4 substrates (climate bundle, Neolithic fraction, ancestral crop yield,
pandemic intensity) plus geography controls. Sum of within-bundle Shapley
values equals the bundle's marginal R² gain in the full specification.

Output: analysis/data/deep_determinants/exercise1_functional_subshapley.parquet
        analysis/figures/paper5_horserace/tab07_functional_subshapley.tex
"""
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.shapley import shapley_r2_decomposition
from analysis.paper5_horserace.exercise1_shapley import (
    CLIMATE_BUNDLE,
    FUNCTIONAL_BUNDLE,
    OUTCOMES,
    CONTROLS,
)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise1_functional_subshapley.parquet"

NON_FUNCTIONAL_SUBSTRATES = list(CLIMATE_BUNDLE) + [
    "neolithic_frac",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]


def main() -> None:
    df = pd.read_parquet(PANEL)
    rows = []
    for outcome in OUTCOMES:
        result = shapley_r2_decomposition(
            df,
            y_col=outcome,
            substrates=list(FUNCTIONAL_BUNDLE),
            controls=CONTROLS + NON_FUNCTIONAL_SUBSTRATES,
        )
        for snp in FUNCTIONAL_BUNDLE:
            rows.append({
                "outcome": outcome,
                "snp": snp,
                "shapley_r2": result["shapley"][snp],
                "baseline_r2": result["baseline_r2"],
                "full_model_r2": result["full_model_r2"],
                "n_obs": result["n_obs"],
            })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out_df)} rows)")

    pivot = out_df.pivot(index="snp", columns="outcome", values="shapley_r2")
    pivot = pivot.reindex(index=list(FUNCTIONAL_BUNDLE), columns=OUTCOMES)
    print("\nWithin-functional-bundle Shapley R²:")
    print(pivot.round(4).to_string())

    _emit_table(out_df, ROOT / "analysis/figures/paper5_horserace/tab07_functional_subshapley.tex")


def _emit_table(out_df: pd.DataFrame, out: Path) -> None:
    pivot = out_df.pivot(index="snp", columns="outcome", values="shapley_r2")
    pivot = pivot.reindex(index=list(FUNCTIONAL_BUNDLE), columns=OUTCOMES)

    label_map = {
        "fa_lct": r"LCT",
        "fa_adh1b": r"ADH1B",
        "fa_amy1": r"AMY1",
        "fa_edar": r"EDAR",
        "fa_darc": r"DARC",
        "fa_slc24a5": r"SLC24A5",
        "fa_hbb": r"HBB",
        "fa_fads": r"FADS1/2",
    }
    outcome_labels = {
        "log_popd_1500": r"$\ln\!D_{1500}$",
        "log_popd_2025": r"$\ln\!D_{2025}$",
        "log_pop_growth_1950_2025": r"$\Delta\!\ln\!P$",
        "urban_change_1950_2025": r"$\Delta\text{Urb}$",
        "log_gdppc_2015": r"$\ln\!\text{GDPpc}$",
        "dt_timing_year": r"DT yr",
    }

    with open(out, "w") as f:
        col_spec = "l" + "r" * len(OUTCOMES)
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        f.write(" & " + " & ".join(outcome_labels.get(o, o) for o in OUTCOMES) + " \\\\\n")
        f.write("\\midrule\n")
        for snp in FUNCTIONAL_BUNDLE:
            row = " & ".join(f"{pivot.loc[snp, o]:.4f}" for o in OUTCOMES)
            f.write(label_map.get(snp, snp) + " & " + row + " \\\\\n")
        f.write("\\midrule\n")
        totals = " & ".join(f"{pivot[o].sum():.4f}" for o in OUTCOMES)
        f.write("Bundle total & " + totals + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
