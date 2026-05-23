"""Fig 16 — Cross-phase progression of the headline climate-fertility coefficient."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")

# Headline numbers across phases (from each phase memo)
PHASE_DATA = [
    ("P1 England STR β_M", -0.617, 0.302),
    ("P2 Pooled STR β_M", -0.459, 0.342),
    ("P3 Per-country DL cum (avg)", -0.42, 0.10),
    ("P4 Pooled DL cum (FE+ctrl)", -0.016, 0.048),
    ("P4 Sigl IV β (LATE)", -1.10, 0.07),
    ("P5 Pooled DL cum (7-country)", 0.009, 0.055),
    ("P5 Indirect via harmonized wage", -0.186, 0.074),
]


def make_fig16():
    labels = [r[0] for r in PHASE_DATA]
    betas = [r[1] for r in PHASE_DATA]
    ses = [r[2] for r in PHASE_DATA]
    colors = ["steelblue", "steelblue", "steelblue", "firebrick", "darkgreen", "firebrick", "darkgreen"]
    fig, ax = plt.subplots(figsize=(12, 5))
    pos = list(range(len(PHASE_DATA)))
    ax.bar(pos, betas, yerr=[1.96 * s for s in ses], color=colors, alpha=0.7)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"Climate-fertility headline $\beta$")
    ax.set_title("Cross-phase progression — methodological tightening shrinks the direct effect; wage channel persists")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig16_phase_progression.pdf"
    png = FIG_DIR / "fig16_phase_progression.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png


if __name__ == "__main__":
    pdf, png = make_fig16()
    print(f"wrote {pdf}")
