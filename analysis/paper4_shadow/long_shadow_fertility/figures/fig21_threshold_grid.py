"""Fig 21 -- Side-by-side Hansen threshold LR paths for three development proxies.

Reads pre-computed phase10_threshold_grid.json and plots one panel per proxy,
showing the LR statistic path, the 95% LR cutoff (7.35), the 95% CI shading,
and the estimated threshold c_hat.
"""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
from pathlib import Path

IN = Path("/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json")
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig21_threshold_grid")
COPY_PDF = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/Fertility/long_shadow/figures/fig21_threshold_grid.pdf"
)
LR95 = 7.35  # Hansen 2000 95% LR cutoff


LABELS = {
    "log_real_wage": "log real wage\n(Allen-Maddison harmonized)",
    "log_cdr": "log crude death rate",
    "log_gdppc": "log GDP per capita",
}


def make_fig21() -> None:
    d = json.loads(IN.read_text())
    n = len(d)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (z, res) in zip(axes, d.items()):
        cs, lrs = zip(*res["lr_path"])
        ax.plot(cs, lrs, color="C0", lw=1.2)
        ax.axhline(
            LR95,
            color="grey",
            lw=0.7,
            linestyle="--",
            label=f"95% LR cutoff = {LR95}",
        )
        # Shade CI region (axvspan)
        ax.axvspan(
            res["c_ci_lo"],
            res["c_ci_hi"],
            alpha=0.20,
            color="C0",
            label="95% LR CI",
        )
        # c_hat vertical line
        ax.axvline(
            res["c_hat"],
            color="C3",
            linestyle="--",
            lw=1.0,
            label=rf"$\hat c$ = {res['c_hat']:.3f}",
        )
        ax.set_xlabel(LABELS.get(z, z), fontsize=9)
        title = (
            rf"$\hat c$ = {res['c_hat']:.3f}  "
            rf"($p_{{\rm sup\text{{-}}Wald}}$ = {res['sup_wald_pvalue']:.3f})"
            "\n"
            rf"$\hat\beta_{{<}}$ = {res['beta_M']:.3f},  "
            rf"$\hat\beta_{{>}}$ = {res['beta_T']:.3f}  "
            rf"($N$ = {res['n']})"
        )
        ax.set_title(title, fontsize=8.5)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_ylabel("LR statistic", fontsize=9)

    fig.suptitle(
        "Hansen (1996, 2000) threshold regression: three development proxies",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = OUT.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Wrote {out_path}")

    # Copy PDF to paper repo
    import shutil
    COPY_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(OUT.with_suffix(".pdf")), str(COPY_PDF))
    print(f"Copied PDF to {COPY_PDF}")

    plt.close(fig)


if __name__ == "__main__":
    make_fig21()
