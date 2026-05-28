"""Tab 6v2 -- Hansen shock decomposition: LaTeX tabular for SPEI, temperature,
precipitation, and GDP-threshold + temperature (development-proxy comparison).

Reads phase10p5_hansen_precip.json (joint-sample results for all three shocks)
and phase10_threshold_grid.json (GDP threshold result).

Outputs tables/tab6v2_shock_decomposition.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

PRECIP_JSON = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10p5_hansen_precip.json"
)
GRID_JSON = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json"
)
OUT = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/Fertility/long_shadow/tables/tab6v2_shock_decomposition.tex"
)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def make_tab6v2() -> None:
    pd = json.loads(PRECIP_JSON.read_text())
    gd = json.loads(GRID_JSON.read_text())

    spei = pd["spei_comparator"]
    temp = pd["temperature_comparator"]
    prec = pd["precipitation"]
    gdp  = gd.get("log_gdppc", {})

    rows = [
        # (label, threshold_var, res)
        ("SPEI (joint stress) & log real wage", spei),
        ("Temperature ($t_{\\text{growing}}$) & log real wage", temp),
        ("Precipitation ($p_{\\text{growing}}$) & log real wage", prec),
        ("Temperature ($t_{\\text{growing}}$) & log GDP per capita", gdp),
    ]

    tex_rows = []
    for label, res in rows:
        if not res:
            tex_rows.append(f"{label} & --- & --- & --- & --- & --- & --- \\\\")
            continue
        row = (
            f"{label} & "
            f"{res.get('c_hat', 0):.3f} & "
            f"[{res.get('c_ci_lo', 0):.3f}, {res.get('c_ci_hi', 0):.3f}] & "
            f"{res.get('beta_M', 0):.4f} & "
            f"{res.get('beta_T', 0):.4f} & "
            f"{fmt_p(res.get('sup_wald_pvalue', 1))} & "
            f"{res.get('n', 0)} \\\\"
        )
        tex_rows.append(row)

    body = "\n".join(tex_rows)

    tex = (
        "% Tab 6v2: Hansen shock decomposition — three shocks x two threshold variables\n"
        r"\begin{tabular}{llcccccc}" + "\n"
        r"\toprule" + "\n"
        r"Shock variable & Threshold variable & $\hat{c}$ & 95\% LR CI & "
        r"$\hat{\beta}_{<}$ & $\hat{\beta}_{>}$ & sup-Wald $p$ & $N$ \\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        "% Notes: All Panels A--C estimated on the identical joint sample ($N=1{,}314$,\n"
        "% 7 countries, 1541--2008) where SPEI, temperature, precipitation, and log real\n"
        "% wage are all non-null. Panel D uses the full wage-available sample ($N=1{,}413$)\n"
        "% with GDP per capita as threshold. sup-Wald $p$-values from 500-replicate\n"
        "% wild-cluster bootstrap (Rademacher weights, clustered on country).\n"
        "% $\\hat\\beta_<$: Malthusian regime (below threshold); $\\hat\\beta_>$: modern regime.\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Wrote {OUT}")

    print("\n=== Tab 6v2 rows ===")
    for label, res in rows:
        if not res:
            print(f"  {label}: MISSING")
            continue
        print(f"  {label[:40]:40s}  c={res.get('c_hat',0):.3f}  "
              f"p={res.get('sup_wald_pvalue',1):.3f}  "
              f"bM={res.get('beta_M',0):.4f}  bT={res.get('beta_T',0):.4f}")


if __name__ == "__main__":
    make_tab6v2()
