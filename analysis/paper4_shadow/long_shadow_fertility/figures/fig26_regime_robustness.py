"""Fig 26 — Robustness of the regime-dependent mortality-FEVD collapse.

Companion to fig25 (the headline FEVD). Three panels in the Long Shadow
grayscale style, all reading phase13_regime_robustness.json (read-only):

  (A) IRF of log fertility to a one-s.d. orthogonalized mortality shock, by
      regime, with the orthogonalized mortality-shock SD annotated. Shows that
      BOTH the shock size and the per-shock response shrink in the modern regime.
  (B) Mortality FEVD share at h=15 under three Cholesky orderings, by regime
      (paired bars). The collapse (Malthusian >> Modern) survives every ordering.
  (C) Leave-one-country-out mortality FEVD share at h=15, by regime (dot plot
      over the 12 single drops), with the headline value marked.

Run run_phase13_regime_robustness first to produce the JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import (
    GRAYS,
    apply_pub_style,
)

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
JSON_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/"
    "phase13_regime_robustness.json"
)
PAPER_FIG_DIR = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/"
    "My Drive/Fertility/long_shadow/figures"
)

ORDER_LABELS = {
    "headline": "[SPEI, W, M, F]\n(headline)",
    "fert_before_mort": "[SPEI, W, F, M]\n(fert<mort)",
    "mort_before_wage": "[SPEI, M, W, F]\n(mort<wage)",
}
MAL_GRAY = GRAYS[0]   # black  = Malthusian
MOD_GRAY = GRAYS[2]   # medium = Modern


def _load() -> dict:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"{JSON_PATH} not found — run run_phase13_regime_robustness first."
        )
    return json.loads(JSON_PATH.read_text())


def make_fig26():
    apply_pub_style(font_size=10, serif=True)
    d = _load()
    A = d["A_irf_and_shock_variance"]
    B = d["B_ordering_swaps"]["orderings"]
    C = d["C_leave_one_country_out"]
    horizons = A["horizons"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # ---- Panel A: IRF of fertility to a 1-s.d. mortality shock, by regime ----
    axA = axes[0]
    axA.axhline(0, color="black", linewidth=0.6)
    mal_irf = np.asarray(A["malthusian"]["irf_logcbr_to_mort_shock"])
    mod_irf = np.asarray(A["modern"]["irf_logcbr_to_mort_shock"])
    axA.plot(horizons, mal_irf, color=MAL_GRAY, linestyle="-",
             label=f"Malthusian (shock SD={A['malthusian']['mort_shock_sd_orth_logcdr']:.3f})")
    axA.plot(horizons, mod_irf, color=MOD_GRAY, linestyle="--",
             label=f"Modern (shock SD={A['modern']['mort_shock_sd_orth_logcdr']:.3f})")
    axA.set_xlim(horizons[0], horizons[-1])
    axA.set_xlabel("Horizon (years)")
    axA.set_ylabel(r"$\log$ fertility response")
    axA.set_title("(A) Response to a 1-s.d.\nmortality shock, by regime", fontsize=10)
    axA.legend(frameon=False, fontsize=8, loc="lower right")

    # ---- Panel B: mortality FEVD share at h=15 across orderings (paired bars) ----
    axB = axes[1]
    names = ["headline", "fert_before_mort", "mort_before_wage"]
    x = np.arange(len(names))
    w = 0.38
    mal_vals = [B[n]["mort_share_h15_malthusian"] for n in names]
    mod_vals = [B[n]["mort_share_h15_modern"] for n in names]
    axB.bar(x - w / 2, mal_vals, w, color=MAL_GRAY, edgecolor="black",
            linewidth=0.5, label="Malthusian", hatch="//")
    axB.bar(x + w / 2, mod_vals, w, color=MOD_GRAY, edgecolor="black",
            linewidth=0.5, label="Modern", hatch="xx")
    axB.set_xticks(x)
    axB.set_xticklabels([ORDER_LABELS[n] for n in names], fontsize=8)
    axB.set_ylabel("Mortality share of fertility\nFE variance, $h=15$")
    axB.set_title("(B) Cholesky ordering swaps", fontsize=10)
    axB.legend(frameon=False, fontsize=8, loc="upper right")

    # ---- Panel C: leave-one-country-out dot plot, by regime ----
    axC = axes[2]
    countries = C["countries"]
    drops = C["drops"]
    mal_drop = [drops[c]["mort_share_h15_malthusian"] for c in countries]
    mod_drop = [drops[c]["mort_share_h15_modern"] for c in countries]
    yc = np.arange(len(countries))[::-1]  # GBR-first reads top-down alpha
    axC.scatter(mal_drop, yc, marker="o", s=26, facecolors="none",
                edgecolors=MAL_GRAY, linewidths=1.0, label="Malthusian (drop)")
    axC.scatter(mod_drop, yc, marker="s", s=22, color=MOD_GRAY, label="Modern (drop)")
    axC.axvline(C["headline_malthusian_share"], color=MAL_GRAY, linestyle="-",
                linewidth=0.8, alpha=0.7)
    axC.axvline(C["headline_modern_share"], color=MOD_GRAY, linestyle="--",
                linewidth=0.8, alpha=0.7)
    axC.set_yticks(yc)
    axC.set_yticklabels(countries, fontsize=7.5)
    axC.set_xlabel("Mortality share, $h=15$")
    axC.set_title("(C) Leave-one-country-out\n(lines = headline)", fontsize=10)
    axC.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Robustness of the regime-dependent mortality-FEVD collapse in fertility",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig26_regime_robustness.pdf"
    png = FIG_DIR / "fig26_regime_robustness.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    copied = None
    try:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        copied = PAPER_FIG_DIR / "fig26_regime_robustness.pdf"
        copied.write_bytes(pdf.read_bytes())
    except OSError as exc:
        print(f"WARNING: could not copy PDF to paper repo: {exc}")
        copied = None

    return {"pdf": pdf, "png": png, "paper_pdf": copied}


if __name__ == "__main__":
    out = make_fig26()
    print(f"wrote {out['pdf']}")
    print(f"wrote {out['png']}")
    if out["paper_pdf"]:
        print(f"copied to {out['paper_pdf']}")
