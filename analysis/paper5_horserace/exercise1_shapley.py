"""Exercise 1: Shapley-Owen variance decomposition across four substrates,
for each of four modern demographic outcomes.

Output: analysis/data/deep_determinants/exercise1_shapley_results.parquet
        (long-form: 16 rows = 4 outcomes × 4 substrates)
        analysis/figures/paper5_horserace/tab03_full_ols.tex
"""
from pathlib import Path

import pandas as pd

from analysis.paper5_horserace.shapley import shapley_r2_decomposition

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT = ROOT / "analysis/data/deep_determinants/exercise1_shapley_results.parquet"

SUBSTRATES = [
    "sigma_v_T_pre1750",
    "H_pred_pwadj",
    "ancestral_yield_log",
    "pandemic_intensity_norm",
]
OUTCOMES = [
    "log_pop_growth_1950_2025",
    "urban_change_1950_2025",
    "log_gdppc_2015",
    "dt_timing_year",
]
CONTROLS = [
    "abs_lat",
    "log_area",
    "landlocked",
    "ruggedness_proxy",
    "log_dist_neolithic",
]


def _stars(p: float) -> str:
    if p < 0.01:
        return "$^{***}$"
    if p < 0.05:
        return "$^{**}$"
    if p < 0.10:
        return "$^{*}$"
    return ""


def _emit_full_ols_table(df: pd.DataFrame, out: Path) -> None:
    """Table 3: pooled OLS, 4 columns (one per outcome), all regressors.

    Pathway dummies pathway_1–pathway_4 are included (pathway_0 omitted
    as the reference category to avoid perfect collinearity).
    HC3 heteroskedasticity-robust standard errors.
    """
    import statsmodels.api as sm

    # pathway_0 is the reference (omitted); include pathway_1–pathway_4
    pathway_cols = sorted(c for c in df.columns if c.startswith("pathway_"))[1:]
    regressors = SUBSTRATES + CONTROLS

    fits = {}
    for outcome in OUTCOMES:
        keep = [outcome] + regressors + pathway_cols
        sub = df.dropna(subset=keep)
        X = sm.add_constant(sub[regressors + pathway_cols])
        res = sm.OLS(sub[outcome], X).fit(cov_type="HC3")
        fits[outcome] = res

    # Friendly display labels
    label_map = {
        "sigma_v_T_pre1750": r"Climate vol.\ ($\sigma_T$)",
        "H_pred_pwadj": r"Pred.\ heterozygosity",
        "ancestral_yield_log": r"Ancestral crop yield (log)",
        "pandemic_intensity_norm": r"Pandemic intensity",
        "abs_lat": r"Abs.\ latitude",
        "log_area": r"Log area",
        "landlocked": r"Landlocked",
        "ruggedness_proxy": r"Ruggedness",
        "log_dist_neolithic": r"Log dist.\ Neolithic",
    }

    outcome_labels = {
        "log_pop_growth_1950_2025": r"$\Delta\ln\text{Pop}$",
        "urban_change_1950_2025": r"$\Delta\text{Urban}$",
        "log_gdppc_2015": r"$\ln\text{GDPpc}$",
        "dt_timing_year": r"DT timing",
    }

    with open(out, "w") as f:
        col_spec = "l" + "r" * len(OUTCOMES)
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")

        # Header row
        headers = " & ".join(outcome_labels.get(o, o) for o in OUTCOMES)
        f.write(" & " + headers + " \\\\\n")
        f.write(" & " + " & ".join(f"({i+1})" for i in range(len(OUTCOMES))) + " \\\\\n")
        f.write("\\midrule\n")

        # Substrate block
        f.write("\\multicolumn{" + str(len(OUTCOMES) + 1) + "}{l}{\\textit{Substrates}} \\\\\n")
        for r in SUBSTRATES:
            _write_regressor_row(f, r, OUTCOMES, fits, label_map)

        # Controls block
        f.write("\\multicolumn{" + str(len(OUTCOMES) + 1) + "}{l}{\\textit{Controls}} \\\\\n")
        for r in CONTROLS:
            _write_regressor_row(f, r, OUTCOMES, fits, label_map)

        # Footer
        f.write("\\midrule\n")
        n_obs = [int(fits[o].nobs) for o in OUTCOMES]
        r2_vals = [fits[o].rsquared for o in OUTCOMES]
        f.write(
            "$N$ & "
            + " & ".join(str(n) for n in n_obs)
            + " \\\\\n"
        )
        f.write(
            "$R^2$ & "
            + " & ".join(f"{r:.3f}" for r in r2_vals)
            + " \\\\\n"
        )
        f.write(
            "Pathway dummies & "
            + " & ".join(["Yes"] * len(OUTCOMES))
            + " \\\\\n"
        )
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Wrote {out}")


def _write_regressor_row(f, r, outcomes, fits, label_map):
    """Write a coefficient + SE row pair for one regressor."""
    label = label_map.get(r, r)
    coef_cells = []
    se_cells = []
    for outcome in outcomes:
        res = fits[outcome]
        if r in res.params:
            coef = res.params[r]
            se = res.bse[r]
            p = res.pvalues[r]
            coef_cells.append(f"{coef:.3f}{_stars(p)}")
            se_cells.append(f"({se:.3f})")
        else:
            coef_cells.append("--")
            se_cells.append("")
    f.write(label + " & " + " & ".join(coef_cells) + " \\\\\n")
    f.write(" & " + " & ".join(se_cells) + " \\\\\n")


def main() -> None:
    df = pd.read_parquet(PANEL)
    rows = []
    for outcome in OUTCOMES:
        print(f"Running Shapley decomposition for outcome: {outcome}")
        result = shapley_r2_decomposition(
            df,
            y_col=outcome,
            substrates=SUBSTRATES,
            controls=CONTROLS,
        )
        for s in SUBSTRATES:
            rows.append(
                {
                    "outcome": outcome,
                    "substrate": s,
                    "shapley_r2": result["shapley"][s],
                    "baseline_r2": result["baseline_r2"],
                    "full_model_r2": result["full_model_r2"],
                    "n_obs": result["n_obs"],
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out_df)} rows)")

    _emit_full_ols_table(
        df, ROOT / "analysis/figures/paper5_horserace/tab03_full_ols.tex"
    )

    pivot = out_df.pivot(index="substrate", columns="outcome", values="shapley_r2")
    print("\nShapley R² decomposition:")
    print(pivot.round(4).to_string())

    print("\nLargest substrate per outcome:")
    for outcome in OUTCOMES:
        col = pivot[outcome]
        best = col.idxmax()
        print(f"  {outcome}: {best} ({col[best]:.4f})")


if __name__ == "__main__":
    main()
