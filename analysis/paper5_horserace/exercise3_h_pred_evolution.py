"""Exercise 3: H_pred coefficient-evolution test, with multicollinearity diagnostics.

For the two outcomes where H_pred was a FWER-surviving v2 predictor (log GDPpc,
log pop growth), document how the H_pred coefficient evolves as the modern
population-genetics substrates are progressively added.

We run THREE parallel exercises:

A. OLS evolution with the full 8-locus functional-allele bundle.
   Expected pattern: H_pred coefficient becomes unstable due to multicollinearity
   with the bundle (both are PW-ancestry-weighted from continental anchor values).
   This is the substantive evidence of overlap, NOT a clean attenuation story.

B. OLS evolution with PCA(functional bundle, k=3) — the first three principal
   components of the standardised 8-SNP bundle. Reduces dimension, removes
   within-bundle collinearity, and produces clean attenuation of H_pred toward
   zero as the modern substrates absorb its signal.

C. Commonality decomposition. Partition the joint variance into:
       unique_H   = R²(H, bundle, X) − R²(bundle, X)   "H given bundle"
       unique_B   = R²(H, bundle, X) − R²(H, X)        "bundle given H"
       common    = R²(H, X) + R²(bundle, X) − R²(H, bundle, X) − R²(X)
   Sum equals the total R² gain over the X-only baseline. The common component
   quantifies the shared (multicollinear) variance, providing a principled
   answer to "how much of H_pred's signal is the bundle also capturing?"

Output: analysis/data/deep_determinants/h_pred_evolution.parquet (OLS-A panel)
        analysis/data/deep_determinants/h_pred_evolution_pca.parquet (PCA-B panel)
        analysis/data/deep_determinants/h_pred_commonality.parquet (C panel)
        analysis/figures/paper5_horserace/tab_h_pred_evolution.tex      (A)
        analysis/figures/paper5_horserace/tab_h_pred_evolution_pca.tex  (B)
        analysis/figures/paper5_horserace/tab_h_pred_commonality.tex    (C)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.paper5_horserace.exercise1_shapley import (
    FUNCTIONAL_BUNDLE,
    CONTROLS,
)

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
OUT_OLS = ROOT / "analysis/data/deep_determinants/h_pred_evolution.parquet"
OUT_PCA = ROOT / "analysis/data/deep_determinants/h_pred_evolution_pca.parquet"
OUT_COM = ROOT / "analysis/data/deep_determinants/h_pred_commonality.parquet"
TEX_OLS = ROOT / "analysis/figures/paper5_horserace/tab_h_pred_evolution.tex"
TEX_PCA = ROOT / "analysis/figures/paper5_horserace/tab_h_pred_evolution_pca.tex"
TEX_COM = ROOT / "analysis/figures/paper5_horserace/tab_h_pred_commonality.tex"

TARGET_OUTCOMES = ["log_gdppc_2015", "log_pop_growth_1950_2025"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_h_coef(df: pd.DataFrame, y_col: str, regressors: list[str]) -> dict:
    """Fit y ~ H_pred + regressors; return H_pred's coefficient diagnostics."""
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


def _r2(df: pd.DataFrame, y_col: str, regressors: list[str]) -> float:
    if not regressors:
        sub = df.dropna(subset=[y_col])
        return 0.0
    needed = [y_col] + regressors
    sub = df.dropna(subset=needed).copy()
    X = sm.add_constant(sub[regressors])
    res = sm.OLS(sub[y_col], X).fit()
    return float(res.rsquared)


def _compute_pcs(df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """First k PCs of the standardised functional-allele bundle.

    Returns a DataFrame with iso3 and fa_pc1, fa_pc2, ..., fa_pck columns.
    PCs are computed on the panel rows with all 8 SNPs non-null.
    """
    fa_cols = list(FUNCTIONAL_BUNDLE)
    sub = df[["iso3"] + fa_cols].dropna(subset=fa_cols).copy()

    scaler = StandardScaler()
    fa_std = scaler.fit_transform(sub[fa_cols].values)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(fa_std)

    pc_df = pd.DataFrame(pcs, columns=[f"fa_pc{i+1}" for i in range(n_components)],
                         index=sub.index)
    out = pd.concat([sub[["iso3"]], pc_df], axis=1)

    print(f"  PCA explained variance ratio: "
          f"{pca.explained_variance_ratio_.round(3).tolist()} "
          f"(cumulative {pca.explained_variance_ratio_.cumsum()[-1]:.2f})")
    print(f"  PC loadings (rows=SNP, cols=PC):")
    loadings = pd.DataFrame(pca.components_.T, index=fa_cols,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    print(loadings.round(2).to_string())
    return out


# ---------------------------------------------------------------------------
# Panel A: OLS evolution with full 8-locus bundle
# ---------------------------------------------------------------------------

def run_ols_evolution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== A. OLS coefficient-evolution with full 8-SNP bundle ===")
    specs = {
        "baseline":        [],
        "plus_neolithic":  ["neolithic_frac"],
        "plus_functional": ["neolithic_frac"] + list(FUNCTIONAL_BUNDLE),
    }
    rows = []
    for outcome in TARGET_OUTCOMES:
        print(f"\n[{outcome}]")
        for spec_name, extra in specs.items():
            r = _fit_h_coef(df, outcome, CONTROLS + extra)
            rows.append({"outcome": outcome, "spec": spec_name, **r})
            stars = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
                     else "*" if r["p"] < 0.10 else "")
            print(f"  {spec_name:18s} β_H={r['coef']:+8.3f} (se {r['se']:.2f}) "
                  f"|t|={abs(r['t']):.2f}{stars}  n={r['n']}, R²={r['r2']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel B: OLS evolution with PCA(functional bundle, k=3)
# ---------------------------------------------------------------------------

def run_pca_evolution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== B. OLS coefficient-evolution with PCA-3 bundle ===")
    pcs_df = _compute_pcs(df, n_components=3)
    df_aug = df.merge(pcs_df, on="iso3", how="left")

    specs = {
        "baseline":        [],
        "plus_neolithic":  ["neolithic_frac"],
        "plus_pcs":        ["neolithic_frac", "fa_pc1", "fa_pc2", "fa_pc3"],
    }
    rows = []
    for outcome in TARGET_OUTCOMES:
        print(f"\n[{outcome}]")
        for spec_name, extra in specs.items():
            r = _fit_h_coef(df_aug, outcome, CONTROLS + extra)
            rows.append({"outcome": outcome, "spec": spec_name, **r})
            stars = ("***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
                     else "*" if r["p"] < 0.10 else "")
            print(f"  {spec_name:18s} β_H={r['coef']:+8.3f} (se {r['se']:.2f}) "
                  f"|t|={abs(r['t']):.2f}{stars}  n={r['n']}, R²={r['r2']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel C: Commonality decomposition
# ---------------------------------------------------------------------------

def run_commonality(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== C. Commonality decomposition (H vs functional bundle) ===")
    fa_cols = list(FUNCTIONAL_BUNDLE)
    rows = []
    for outcome in TARGET_OUTCOMES:
        # Restrict to common sample (all variables non-null) for clean comparison
        needed = [outcome, "H_pred_pwadj"] + fa_cols + CONTROLS
        sub = df.dropna(subset=needed).copy()
        n = len(sub)

        r2_x = _r2(sub, outcome, CONTROLS)
        r2_hx = _r2(sub, outcome, CONTROLS + ["H_pred_pwadj"])
        r2_bx = _r2(sub, outcome, CONTROLS + fa_cols)
        r2_hbx = _r2(sub, outcome, CONTROLS + ["H_pred_pwadj"] + fa_cols)

        total = r2_hbx - r2_x
        unique_h = r2_hbx - r2_bx       # H's marginal R² given B
        unique_b = r2_hbx - r2_hx       # B's marginal R² given H
        common = r2_hx + r2_bx - r2_hbx - r2_x
        # Identity: total = unique_h + unique_b + common (within rounding)

        rows.append({
            "outcome": outcome,
            "n": n,
            "r2_controls": r2_x,
            "r2_H_plus_X": r2_hx,
            "r2_B_plus_X": r2_bx,
            "r2_full": r2_hbx,
            "total_gain": total,
            "unique_H": unique_h,
            "unique_B": unique_b,
            "common": common,
            "frac_unique_H": unique_h / total if total > 0 else np.nan,
            "frac_unique_B": unique_b / total if total > 0 else np.nan,
            "frac_common":   common / total if total > 0 else np.nan,
        })
        print(f"\n[{outcome}]  n={n}")
        print(f"  R²(controls)            = {r2_x:.3f}")
        print(f"  R²(H + controls)        = {r2_hx:.3f}")
        print(f"  R²(bundle + controls)   = {r2_bx:.3f}")
        print(f"  R²(H + bundle + ctrl)   = {r2_hbx:.3f}")
        print(f"  Total gain (H+B over X) = {total:.3f}")
        print(f"  Unique H (given B)      = {unique_h:.3f}  ({100*unique_h/total:5.1f}%)")
        print(f"  Unique B (given H)      = {unique_b:.3f}  ({100*unique_b/total:5.1f}%)")
        print(f"  Common H × B            = {common:.3f}  ({100*common/total:5.1f}%)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX emitters
# ---------------------------------------------------------------------------

def _emit_evolution_table(out_df: pd.DataFrame, spec_labels: dict,
                          spec_order: list, outcome_labels: dict, dest: Path) -> None:
    with open(dest, "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Specification & $\\hat\\beta_{H}$ & SE & $|t|$ & $N$ \\\\\n")
        for outcome in TARGET_OUTCOMES:
            f.write("\\midrule\n")
            f.write("\\multicolumn{5}{l}{\\textit{" + outcome_labels[outcome] + "}} \\\\\n")
            for spec in spec_order:
                row = out_df[(out_df["outcome"] == outcome) & (out_df["spec"] == spec)].iloc[0]
                stars = ("$^{***}$" if row["p"] < 0.01
                         else "$^{**}$" if row["p"] < 0.05
                         else "$^{*}$" if row["p"] < 0.10 else "")
                f.write(f"\\quad {spec_labels[spec]} & {row['coef']:+.2f}{stars} & "
                        f"{row['se']:.2f} & {abs(row['t']):.2f} & {int(row['n'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote {dest}")


def _emit_commonality_table(com_df: pd.DataFrame, dest: Path) -> None:
    outcome_labels = {
        "log_gdppc_2015": r"$\ln\text{GDPpc}_{2015}$",
        "log_pop_growth_1950_2025": r"$\Delta\ln\!P_{1950\to2025}$",
    }
    with open(dest, "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Outcome & Unique $H$ & Unique bundle & Common & Total \\\\\n")
        f.write(" & ($\\Delta R^2$) & ($\\Delta R^2$) & ($\\Delta R^2$) & gain \\\\\n")
        f.write("\\midrule\n")
        for _, row in com_df.iterrows():
            o = outcome_labels.get(row["outcome"], row["outcome"])
            f.write(f"{o} & "
                    f"{row['unique_H']:.3f} ({100*row['frac_unique_H']:.0f}\\%) & "
                    f"{row['unique_B']:.3f} ({100*row['frac_unique_B']:.0f}\\%) & "
                    f"{row['common']:.3f} ({100*row['frac_common']:.0f}\\%) & "
                    f"{row['total_gain']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_parquet(PANEL)

    # Panel A: original OLS evolution
    ols_df = run_ols_evolution(df)
    ols_df.to_parquet(OUT_OLS, index=False)

    # Panel B: PCA evolution
    pca_df = run_pca_evolution(df)
    pca_df.to_parquet(OUT_PCA, index=False)

    # Panel C: commonality
    com_df = run_commonality(df)
    com_df.to_parquet(OUT_COM, index=False)

    # LaTeX tables
    outcome_labels = {
        "log_gdppc_2015": r"$\ln\text{GDPpc}_{2015}$",
        "log_pop_growth_1950_2025": r"$\Delta\ln\!P_{1950\to2025}$",
    }
    print("\n=== Writing LaTeX tables ===")
    _emit_evolution_table(
        ols_df,
        spec_labels={"baseline": "Baseline",
                     "plus_neolithic": "+ Neolithic frac",
                     "plus_functional": "+ Functional alleles (8 SNPs)"},
        spec_order=["baseline", "plus_neolithic", "plus_functional"],
        outcome_labels=outcome_labels,
        dest=TEX_OLS,
    )
    _emit_evolution_table(
        pca_df,
        spec_labels={"baseline": "Baseline",
                     "plus_neolithic": "+ Neolithic frac",
                     "plus_pcs": "+ Functional alleles (PC1-3)"},
        spec_order=["baseline", "plus_neolithic", "plus_pcs"],
        outcome_labels=outcome_labels,
        dest=TEX_PCA,
    )
    _emit_commonality_table(com_df, TEX_COM)


if __name__ == "__main__":
    main()
