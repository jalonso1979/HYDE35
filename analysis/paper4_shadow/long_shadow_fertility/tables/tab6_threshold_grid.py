"""Tab 6 -- Hansen threshold grid: LaTeX tabular for three development proxies.

Reads pre-computed phase10_threshold_grid.json and emits a LaTeX tabular
with c_hat, 95% LR CI, beta_M, beta_T, sup-Wald p-value, and N.
"""
from __future__ import annotations

import json
from pathlib import Path

IN = Path("/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json")
OUT = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/Fertility/long_shadow/tables/tab6_threshold_grid.tex"
)

LABELS = {
    "log_real_wage": "Log real wage",
    "log_cdr": "Log crude death rate",
    "log_gdppc": "Log GDP per capita",
}


def make_tab6() -> None:
    d = json.loads(IN.read_text())
    rows = []
    for z, res in d.items():
        label = LABELS.get(z, z)
        row = (
            f"{label} & "
            f"{res['c_hat']:.3f} & "
            f"[{res['c_ci_lo']:.3f}, {res['c_ci_hi']:.3f}] & "
            f"{res['beta_M']:.3f} & "
            f"{res['beta_T']:.3f} & "
            f"{res['sup_wald_pvalue']:.3f} & "
            f"{res['n']} \\\\"
        )
        rows.append(row)

    body = "\n".join(rows)
    tex = (
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r"Threshold variable & $\hat{c}$ & 95\% LR CI & "
        r"$\hat{\beta}_{<}$ & $\hat{\beta}_{>}$ & sup-Wald $p$ & $N$ \\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    make_tab6()
