"""Fig 16 — Cross-phase progression of the headline climate-fertility coefficient.

Phase 1-4 numbers are hardcoded (historical, won't change). Phase 5 indirect-via-wage
is re-sourced from the current fig11v3 mediation output to reflect Phase 7's wage
harmonization fix. Phase 7 LP-FEVD adds the dynamic wage-mediated FEVD share at h=15.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")

# Phase 1-4 historical headline numbers (memos snapshot at phase merge)
PHASE_DATA_HISTORICAL = [
    ("P1 England STR β_M", -0.617, 0.302),
    ("P2 Pooled STR β_M", -0.459, 0.342),
    ("P3 Per-country DL cum (avg)", -0.42, 0.10),
    ("P4 Pooled DL cum (FE+ctrl)", -0.016, 0.048),
    ("P4 Sigl IV β (LATE)", -1.10, 0.07),
]


def _current_p5_dl_cum():
    """Re-fit Phase 5 pooled DL on current data; return (cum_beta, cum_se)."""
    from analysis.paper4_shadow.long_shadow_fertility.figures.fig7v3_pooled_dl_irf_v3 import (
        make_fig7v3,
    )
    result = make_fig7v3()
    if len(result) == 3:
        _pdf, _png, summary = result
    else:
        raise ValueError(f"Unexpected fig7v3 return: {type(result)}")

    if isinstance(summary, dict):
        return (
            float(summary.get("cum_beta", summary.get("cumulative_beta", 0.0))),
            float(summary.get("cum_se", summary.get("cumulative_se", 0.0))),
        )
    # DataFrame with a "cumulative" row in the "lag" column
    cum_row = summary[summary["lag"] == "cumulative"].iloc[0]
    return float(cum_row["beta"]), float(cum_row["se"])


def _current_p5_indirect():
    """Re-fit Phase 5 mediation; return (indirect_beta, indirect_se)."""
    from analysis.paper4_shadow.long_shadow_fertility.figures.fig11v3_mediation_wild_bootstrap import (
        make_fig11v3,
    )
    result = make_fig11v3()
    if len(result) == 3:
        _pdf, _png, summary = result
    else:
        raise ValueError(f"Unexpected fig11v3 return: {type(result)}")
    if isinstance(summary, dict):
        indirect = float(summary.get("indirect", 0.0))
        # fig11v3 uses "indirect_wild_cluster_se"; fall back to "indirect_se"
        se = float(
            summary.get(
                "indirect_wild_cluster_se",
                summary.get("indirect_se", 0.0),
            )
        )
        return indirect, se
    raise ValueError(f"fig11v3 summary not a dict: {type(summary)}")


def _current_p7_lpfevd_w_share():
    """Phase 7 LP-FEVD W-share of F variance at h=15."""
    from analysis.paper4_shadow.long_shadow_fertility.figures.fig18_pooled_system_lp_fevd import (
        make_fig18,
    )
    _pdf, _png, summary = make_fig18()
    w_share_h15 = float(summary["fevd_F"]["W"][-1])
    return w_share_h15


def _collect_phase_data():
    """Combine historical + live numbers into the bar series."""
    rows = list(PHASE_DATA_HISTORICAL)
    cum_b, cum_s = _current_p5_dl_cum()
    rows.append(("P5 Pooled DL cum (7-country)", cum_b, cum_s))
    ind_b, ind_s = _current_p5_indirect()
    rows.append(("P5 Indirect via harmonized wage", ind_b, ind_s))
    p7_w = _current_p7_lpfevd_w_share()
    rows.append(("P7 LP-FEVD W-share at h=15", p7_w, 0.0))
    return rows


def make_fig16():
    phase_data = _collect_phase_data()
    labels = [r[0] for r in phase_data]
    betas = [r[1] for r in phase_data]
    ses = [r[2] for r in phase_data]
    colors = (
        ["steelblue", "steelblue", "steelblue", "firebrick", "darkgreen"]
        + ["firebrick", "darkgreen", "purple"]
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    pos = list(range(len(phase_data)))
    ax.bar(pos, betas, yerr=[1.96 * s for s in ses], color=colors, alpha=0.7)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"Headline $\beta$ or variance share")
    ax.set_title("Cross-phase progression — methodological tightening + dynamic LP-FEVD reframe")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig16_phase_progression.pdf"
    png = FIG_DIR / "fig16_phase_progression.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png, phase_data


if __name__ == "__main__":
    pdf, png, data = make_fig16()
    print(f"wrote {pdf}")
    for label, b, s in data:
        print(f"  {label}: {b:+.4f} ({s:.4f})")
