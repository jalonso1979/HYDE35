"""Fig 21v2 -- 4-panel Hansen LR paths for three shock variables + development proxy comparison.

Reads:
  - phase10p5_hansen_precip.json  (precipitation, temperature, and SPEI on joint sample)
  - phase10_threshold_grid.json   (log_gdppc with temperature as comparator)

Panels (2x2 grid):
  [A] SPEI as shock          — headline (p=0.012)
  [B] Temperature as shock   — v0.5 result (p=0.012)
  [C] Precipitation as shock — new test (p=0.732)
  [D] GDP per capita threshold + temperature — development-proxy comparison

Each panel: LR path, 95% cutoff at 7.35, LR CI shading, c_hat vertical line.

Output: fig21v2_shock_decomposition.{pdf,png}
Copy PDF to paper repo figures/ folder.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Input data
PRECIP_JSON = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10p5_hansen_precip.json"
)
GRID_JSON = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_threshold_grid.json"
)
OUT = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility/fig21v2_shock_decomposition"
)
COPY_PDF = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/Fertility/long_shadow/figures/fig21v2_shock_decomposition.pdf"
)

LR95 = 7.35  # Hansen 2000 95% LR cutoff


def _panel(ax, lr_path, c_hat, c_ci_lo, c_ci_hi, beta_M, beta_T, p, n, xlabel, title_prefix):
    """Draw one Hansen LR path panel."""
    cs, lrs = zip(*lr_path)
    ax.plot(cs, lrs, color="C0", lw=1.3)
    ax.axhline(LR95, color="grey", lw=0.8, linestyle="--", label=f"95\\% LR cutoff = {LR95}")
    ax.axvspan(c_ci_lo, c_ci_hi, alpha=0.20, color="C0", label="95\\% LR CI")
    ax.axvline(c_hat, color="C3", linestyle="--", lw=1.1,
               label=rf"$\hat c$ = {c_hat:.3f}")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("LR statistic", fontsize=9)
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    title = (
        f"{title_prefix}\n"
        rf"$\hat c={c_hat:.3f}$, $p={p_str}$" + "\n"
        rf"$\hat\beta_{{<}}={beta_M:.3f}$, $\hat\beta_{{>}}={beta_T:.3f}$  ($N={n}$)"
    )
    ax.set_title(title, fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7)


def make_fig21v2() -> None:
    prec_d = json.loads(PRECIP_JSON.read_text())
    grid_d = json.loads(GRID_JSON.read_text())

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Panel A: SPEI ---
    ax_a = fig.add_subplot(gs[0, 0])
    s = prec_d["spei_comparator"]
    _panel(
        ax_a,
        lr_path=s["lr_path"],
        c_hat=s["c_hat"], c_ci_lo=s["c_ci_lo"], c_ci_hi=s["c_ci_hi"],
        beta_M=s["beta_M"], beta_T=s["beta_T"],
        p=s["sup_wald_pvalue"], n=s["n"],
        xlabel="log real wage (threshold variable)",
        title_prefix="(A) SPEI shock — headline",
    )

    # --- Panel B: Temperature ---
    ax_b = fig.add_subplot(gs[0, 1])
    t = prec_d["temperature_comparator"]
    _panel(
        ax_b,
        lr_path=t["lr_path"],
        c_hat=t["c_hat"], c_ci_lo=t["c_ci_lo"], c_ci_hi=t["c_ci_hi"],
        beta_M=t["beta_M"], beta_T=t["beta_T"],
        p=t["sup_wald_pvalue"], n=t["n"],
        xlabel="log real wage (threshold variable)",
        title_prefix="(B) Temperature shock",
    )

    # --- Panel C: Precipitation ---
    ax_c = fig.add_subplot(gs[1, 0])
    p = prec_d["precipitation"]
    _panel(
        ax_c,
        lr_path=p["lr_path"],
        c_hat=p["c_hat"], c_ci_lo=p["c_ci_lo"], c_ci_hi=p["c_ci_hi"],
        beta_M=p["beta_M"], beta_T=p["beta_T"],
        p=p["sup_wald_pvalue"], n=p["n"],
        xlabel="log real wage (threshold variable)",
        title_prefix="(C) Precipitation shock",
    )
    # Annotate the non-significance clearly
    ax_c.text(
        0.05, 0.95, "No regime change detected",
        transform=ax_c.transAxes, fontsize=8, color="C3",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="C3", alpha=0.8)
    )

    # --- Panel D: GDP per capita threshold + Temperature (development-proxy comparison) ---
    ax_d = fig.add_subplot(gs[1, 1])
    gdp_res = grid_d.get("log_gdppc")
    if gdp_res is not None:
        _panel(
            ax_d,
            lr_path=gdp_res["lr_path"],
            c_hat=gdp_res["c_hat"], c_ci_lo=gdp_res["c_ci_lo"], c_ci_hi=gdp_res["c_ci_hi"],
            beta_M=gdp_res["beta_M"], beta_T=gdp_res["beta_T"],
            p=gdp_res["sup_wald_pvalue"], n=gdp_res["n"],
            xlabel="log GDP per capita (threshold variable)",
            title_prefix="(D) Temperature shock / GDP threshold",
        )
    else:
        ax_d.text(0.5, 0.5, "GDP threshold data not found",
                  ha="center", va="center", transform=ax_d.transAxes)

    fig.suptitle(
        "Hansen (1996, 2000) threshold regression: shock decomposition\n"
        "Panels A-C: log real wage threshold, varying the climate shock. "
        "Panel D: GDP per capita threshold (temperature shock).",
        fontsize=11, y=1.01,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = OUT.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Wrote {out_path}")

    # Copy PDF to paper repo
    COPY_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(OUT.with_suffix(".pdf")), str(COPY_PDF))
    print(f"Copied PDF to {COPY_PDF}")

    plt.close(fig)


if __name__ == "__main__":
    make_fig21v2()
