"""Exercise 1b: Within-bundle Shapley decomposition for the four climate primitives.

For each outcome, decompose the climate bundle's R² contribution across
(T̄, P̄, σᵥᵀ, σᵥᴾ), conditioning on the non-climate substrates
(H, A, Π) plus the standard geography controls.

The baseline for this sub-decomposition is therefore richer than the headline
exercise: controls + 3 non-climate substrates. Under this baseline, sum of
the four sub-Shapley values equals R²(full) - R²(controls + 3 non-climate
substrates) = the marginal contribution of the climate bundle conditional
on everything else.

Output: analysis/data/deep_determinants/exercise1_climate_subshapley.parquet
        analysis/figures/paper5_horserace/tab06_climate_subshapley.tex
"""
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.shapley import shapley_r2_decomposition
from analysis.paper5_horserace.exercise1_shapley import (
    CLIMATE_BUNDLE,
    OUTCOMES,
    CONTROLS,
)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise1_climate_subshapley.parquet"

NON_CLIMATE_SUBSTRATES = [
    "H_pred_pwadj",
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
            substrates=list(CLIMATE_BUNDLE),
            controls=CONTROLS + NON_CLIMATE_SUBSTRATES,
        )
        for cvar in CLIMATE_BUNDLE:
            rows.append(
                {
                    "outcome": outcome,
                    "climate_var": cvar,
                    "shapley_r2": result["shapley"][cvar],
                    "baseline_r2": result["baseline_r2"],
                    "full_model_r2": result["full_model_r2"],
                    "n_obs": result["n_obs"],
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out_df)} rows)")

    pivot = out_df.pivot(index="climate_var", columns="outcome", values="shapley_r2")
    pivot = pivot.reindex(index=list(CLIMATE_BUNDLE), columns=OUTCOMES)
    print("\nWithin-bundle climate Shapley R² (conditional on H, A, Π + controls):")
    print(pivot.round(4).to_string())

    _emit_table(out_df, ROOT / "analysis/figures/paper5_horserace/tab06_climate_subshapley.tex")


def _emit_table(out_df: pd.DataFrame, out: Path) -> None:
    pivot = out_df.pivot(index="climate_var", columns="outcome", values="shapley_r2")
    pivot = pivot.reindex(index=list(CLIMATE_BUNDLE), columns=OUTCOMES)

    label_map = {
        "t_mean_pre1750": r"Mean T ($\bar T$)",
        "p_mean_pre1750": r"Mean P ($\bar P$)",
        "sigma_v_T_pre1750": r"T volatility ($\sigma_v^T$)",
        "sigma_v_P_pre1750": r"P volatility ($\sigma_v^P$)",
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
        for cvar in CLIMATE_BUNDLE:
            row = " & ".join(f"{pivot.loc[cvar, o]:.4f}" for o in OUTCOMES)
            f.write(label_map.get(cvar, cvar) + " & " + row + " \\\\\n")
        f.write("\\midrule\n")
        totals = " & ".join(f"{pivot[o].sum():.4f}" for o in OUTCOMES)
        f.write("Bundle total & " + totals + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
