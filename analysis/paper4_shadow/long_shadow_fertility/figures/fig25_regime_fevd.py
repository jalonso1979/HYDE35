"""Fig 25 — Forecast-error variance decomposition of log fertility, by regime.

The headline of the Long Shadow paper: a two-panel stacked-area FEVD chart from
the state-dependent core system LP-FEVD [SPEI, log_real_wage, log_cdr, log_cbr]
estimated separately on the Malthusian (left) and Modern (right) wage regimes.

x-axis = forecast horizon (0..15 years)
y-axis = share of log-CBR forecast-error variance
stacked areas = {weather (SPEI), wages, mortality, fertility-own}

The elasticity transition rendered dynamically: across the Hansen wage threshold
the weather share of fertility variance grows while the mortality share shrinks.

Grayscale publication styling: distinct Greys levels + hatching per shock so the
areas are distinguishable in pure black-and-white print.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import apply_pub_style

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
JSON_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase11_regime_fevd.json"
)
PAPER_FIG_DIR = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/"
    "My Drive/Fertility/long_shadow/figures"
)

# Stacked order (bottom -> top) and display labels for the core system shocks.
STACK_VARS = ["spei_growing", "log_real_wage", "log_cdr", "log_cbr"]
STACK_LABELS = ["Weather (SPEI)", "Wages", "Mortality", "Fertility (own)"]
HATCHES = ["//", "\\\\", "xx", ".."]


def _load_fevd() -> dict:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"{JSON_PATH} not found — run run_phase11_regime_fevd first."
        )
    return json.loads(JSON_PATH.read_text())


def _stack_matrix(regime_entry: dict) -> np.ndarray:
    """Return (n_shocks, H) FEVD share matrix in STACK_VARS order."""
    path = regime_entry["fevd_log_cbr_path"]
    return np.vstack([np.asarray(path[v], dtype=float) for v in STACK_VARS])


def make_fig25():
    apply_pub_style(font_size=10, serif=True)
    data = _load_fevd()
    core = data["core_by_regime"]
    horizons = core["horizons"]

    # 4 evenly-spaced grayscale levels (avoid pure white / pure black extremes).
    cmap = plt.get_cmap("Greys")
    levels = np.linspace(0.30, 0.85, len(STACK_VARS))
    fill_colors = [cmap(x) for x in levels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [
        (axes[0], core["malthusian"], "Malthusian regime", r"$\log W \leq 9.97$"),
        (axes[1], core["modern"], "Modern regime", r"$\log W > 9.97$"),
    ]

    for ax, entry, title, regime_lbl in panels:
        shares = _stack_matrix(entry)
        polys = ax.stackplot(
            horizons,
            shares,
            labels=STACK_LABELS,
            colors=fill_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        # Add hatching so areas separate in pure grayscale.
        for poly, hatch in zip(polys, HATCHES):
            poly.set_hatch(hatch)
        ax.set_xlim(horizons[0], horizons[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast horizon (years)")
        ax.set_title(f"{title}\n({regime_lbl}, $N={entry['n']}$)", fontsize=10)

    axes[0].set_ylabel("Share of log-fertility forecast-error variance")

    # Single legend below both panels, reversed so top-of-stack reads top-of-legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Forecast-error variance decomposition of log fertility, by development regime",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig25_regime_fevd.pdf"
    png = FIG_DIR / "fig25_regime_fevd.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # Copy the vector PDF to the paper repo figures dir.
    copied = None
    try:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        copied = PAPER_FIG_DIR / "fig25_regime_fevd.pdf"
        copied.write_bytes(pdf.read_bytes())
    except OSError as exc:  # paper repo may be unmounted in some environments
        print(f"WARNING: could not copy PDF to paper repo: {exc}")
        copied = None

    weather_mal = core["malthusian"]["fevd_log_cbr_h15"]["spei_growing"]
    weather_mod = core["modern"]["fevd_log_cbr_h15"]["spei_growing"]
    return {
        "pdf": pdf,
        "png": png,
        "paper_pdf": copied,
        "weather_share_h15": {"malthusian": weather_mal, "modern": weather_mod},
    }


if __name__ == "__main__":
    out = make_fig25()
    print(f"wrote {out['pdf']}")
    print(f"wrote {out['png']}")
    if out["paper_pdf"]:
        print(f"copied to {out['paper_pdf']}")
    ws = out["weather_share_h15"]
    print(
        f"weather (SPEI) share of fertility variance at h=15: "
        f"Malthusian={ws['malthusian']:.3f}  Modern={ws['modern']:.3f}  "
        f"(delta={ws['modern'] - ws['malthusian']:+.3f})"
    )
