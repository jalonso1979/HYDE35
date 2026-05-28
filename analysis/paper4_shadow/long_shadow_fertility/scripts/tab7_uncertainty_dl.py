"""Tab 7: Climate uncertainty channel — distributed-lag regression table.

Reads phase10_uncertainty_dl.json and produces a LaTeX table with:
  - One column per model (M1, M2, M3)
  - Cumulative temperature beta (with SE)
  - Uncertainty coefficient (realized SD for M1/M3, ensstd for M2)
  - Interaction coefficient t_sd_x_above (M3 only)
  - N, N_units

Output:
  /Users/jalonso/.../Fertility/long_shadow/tables/tab7_uncertainty_dl.tex
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_IN = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility"
    "/phase10_uncertainty_dl.json"
)
OUT = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/Fertility/long_shadow/tables/tab7_uncertainty_dl.tex"
)


def _fmt(b: float, s: float, digits: int = 3) -> tuple[str, str]:
    """Return (coef_str, se_str) with stars."""
    z = abs(b / s) if s > 0 else 0
    stars = "***" if z > 2.576 else ("**" if z > 1.960 else ("*" if z > 1.645 else ""))
    b_str = f"{b:+.{digits}f}{stars}"
    s_str = f"({s:.{digits}f})"
    return b_str, s_str


def make_tab7():
    d = json.loads(DATA_IN.read_text())

    # Extract values for each model
    models = [
        ("m1_baseline_with_realized_sd", "M1"),
        ("m2_baseline_with_ensstd", "M2"),
        ("m3_realized_sd_x_threshold", "M3"),
    ]
    uncertainty_key = {
        "m1_baseline_with_realized_sd": "t_anom_c_within_season_sd",
        "m2_baseline_with_ensstd": "ensstd_t_growing",
        "m3_realized_sd_x_threshold": "t_anom_c_within_season_sd",
    }

    # Build column data
    cols = {}
    for mkey, mlabel in models:
        res = d[mkey]
        irf = res["irf"]
        ctrl = res["ctrl_coefs"]

        # Cumulative temperature beta
        cum_row = [r for r in irf if r["lag"] == "cumulative"][0]
        cum_b = cum_row["beta"]
        cum_s = cum_row["se"]

        # Uncertainty coefficient
        unc_k = uncertainty_key[mkey]
        unc = ctrl.get(unc_k, {"beta": float("nan"), "se": float("nan")})
        unc_b = unc["beta"]
        unc_s = unc["se"]

        # Interaction (M3 only)
        int_coef = ctrl.get("t_sd_x_above", None)

        cols[mlabel] = {
            "cum_b": cum_b, "cum_s": cum_s,
            "unc_b": unc_b, "unc_s": unc_s,
            "unc_key": unc_k,
            "int_coef": int_coef,
            "n_obs": res["n_obs"],
            "n_units": res["n_units"],
        }

    # LaTeX table
    m1, m2, m3 = cols["M1"], cols["M2"], cols["M3"]

    def row_pair(label: str, b1, s1, b2, s2, b3, s3,
                 skip2: bool = False, skip3: bool = False) -> str:
        if skip2 and skip3:
            c2, c2s = "---", ""
            c3, c3s = "---", ""
        elif skip2:
            c2, c2s = "---", ""
            c3, c3s = _fmt(b3, s3) if b3 == b3 else ("---", "")
        elif skip3:
            c2, c2s = _fmt(b2, s2) if b2 == b2 else ("---", "")
            c3, c3s = "---", ""
        else:
            c2, c2s = _fmt(b2, s2) if b2 == b2 else ("---", "")
            c3, c3s = _fmt(b3, s3) if b3 == b3 else ("---", "")

        c1, c1s = _fmt(b1, s1) if b1 == b1 else ("---", "")
        out = (f"        {label} & {c1} & {c2} & {c3} \\\\\n"
               f"                        & {c1s} & {c2s} & {c3s} \\\\\n")
        return out

    # Uncertainty row labels
    unc_row1 = r"Realized SD$_T$ (within-season)"
    unc_row2 = r"Ensemble spread (ensstd$_T$)"
    unc_row3 = r"Realized SD$_T$ (base)"
    int_row3 = r"Realized SD$_T \times$ Above threshold"

    int3 = m3["int_coef"] or {"beta": float("nan"), "se": float("nan")}

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{Climate Uncertainty Channel: Distributed-Lag Regressions}",
        r"  \label{tab:uncertainty_dl}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    & M1 & M2 & M3 \\",
        r"    & Realized SD & ensstd & SD $\times$ Regime \\",
        r"    \midrule",
        r"    \multicolumn{4}{l}{\textit{Cumulative temperature effect}} \\",
        row_pair(r"Cumulative $\hat\beta(T)$",
                 m1["cum_b"], m1["cum_s"],
                 m2["cum_b"], m2["cum_s"],
                 m3["cum_b"], m3["cum_s"]),
        r"    \addlinespace",
        r"    \multicolumn{4}{l}{\textit{Uncertainty coefficient}} \\",
        # M1 row: realized SD
        f"        {unc_row1} & {_fmt(m1['unc_b'], m1['unc_s'])[0]} & --- & --- \\\\\n"
        f"                        & {_fmt(m1['unc_b'], m1['unc_s'])[1]} &     &     \\\\\n",
        # M2 row: ensstd
        f"        {unc_row2} & --- & {_fmt(m2['unc_b'], m2['unc_s'])[0]} & --- \\\\\n"
        f"                        &     & {_fmt(m2['unc_b'], m2['unc_s'])[1]} &     \\\\\n",
        # M3 base SD
        f"        {unc_row3} & --- & --- & {_fmt(m3['unc_b'], m3['unc_s'])[0]} \\\\\n"
        f"                        &     &     & {_fmt(m3['unc_b'], m3['unc_s'])[1]} \\\\\n",
        r"    \addlinespace",
        r"    \multicolumn{4}{l}{\textit{Regime interaction (M3)}} \\",
        f"        {int_row3} & --- & --- & {_fmt(int3['beta'], int3['se'])[0]} \\\\\n"
        f"                        &     &     & {_fmt(int3['beta'], int3['se'])[1]} \\\\\n",
        r"    \midrule",
        (f"        Observations & {m1['n_obs']:,} & {m2['n_obs']:,} & {m3['n_obs']:,} \\\\\n"),
        (f"        Countries & {m1['n_units']} & {m2['n_units']} & {m3['n_units']} \\\\\n"),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}",
        r"    \footnotesize",
        r"    \item Notes: Pooled OLS with country and year fixed effects.",
        r"    Distributed lag on T$_{\text{growing}}$ with 3 lags (k=0..3).",
        r"    M1/M3 use within-season realized SD of monthly temperature anomaly",
        r"    as uncertainty proxy; M2 uses ModE-RA ensemble spread (ensstd).",
        r"    M3 includes interaction with Above-threshold dummy",
        r"    (log real wage $> 9.97$, Hansen Phase 9 threshold).",
        r"    Cluster-robust SEs in parentheses (clustered by country).",
        r"    *, **, *** denote 10\%, 5\%, 1\% significance.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Wrote {OUT}")

    # Print summary
    print("\n=== Tab 7 key numbers ===")
    print(f"M1: cum T beta = {m1['cum_b']:+.4f} (SE {m1['cum_s']:.4f}), "
          f"realized_SD = {m1['unc_b']:+.4f} (SE {m1['unc_s']:.4f})")
    print(f"M2: cum T beta = {m2['cum_b']:+.4f} (SE {m2['cum_s']:.4f}), "
          f"ensstd_t = {m2['unc_b']:+.4f} (SE {m2['unc_s']:.4f})")
    print(f"M3: cum T beta = {m3['cum_b']:+.4f} (SE {m3['cum_s']:.4f}), "
          f"t_sd_x_above = {int3['beta']:+.4f} (SE {int3['se']:.4f})")


if __name__ == "__main__":
    make_tab7()
