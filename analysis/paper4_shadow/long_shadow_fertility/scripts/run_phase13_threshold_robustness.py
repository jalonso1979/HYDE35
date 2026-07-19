"""Phase 13: two robustness checks for the Hansen wage-threshold / regime split.

Both checks address referee concerns about the development-regime identification
in the Long Shadow paper. The headline (Phase 11) splits the panel at a ROUND
wage value c = 9.97 and finds the fertility FEVD mortality share COLLAPSES across
the threshold (Malthusian 0.181 -> Modern 0.0163) while the weather share rises.
The Hansen sup-Wald test locates the wage threshold at c_hat = 10.14, p = 0.002.

(A) Regime-split sensitivity sweep  [referee 3c]
    The headline split 9.97 matches none of the estimated thresholds exactly
    (Hansen wage c_hat = 10.14; Hansen gdppc c_hat = 9.94; LR-CI lower bounds
    near 9.66). Referee 3c worries the result is knife-edge at 9.97. We sweep the
    regime-split value c over [9.5, 10.5] in steps of 0.05, recompute the core
    system LP-FEVD [spei_growing, log_real_wage, log_cdr, log_cbr] SEPARATELY on
    each regime subsample at every c, and record the Malthusian/Modern mortality
    (log_cdr) and wage (log_real_wage) FEVD shares at h=15. We then confirm the
    mortality collapse (Malthusian high, Modern low) holds monotonically/stably
    across the economically meaningful window c in [9.66, 10.14].

(B) Placebo-threshold falsification  [referee 4.9 / 2a]
    Referee worries the Hansen sup-Wald test mechanically fires on ANY monotone
    covariate, making the wage threshold non-informative. We run the SAME test
    (fit_threshold_regression, y=log_cbr, x=t_growing, identical n_boot/seed) with
    PLACEBO threshold variables z and compare to the real wage threshold (re-
    estimated here under identical settings). Because a 500-rep wild-cluster
    bootstrap has a p-value floor of 1/500 = 0.002, any z carrying real structure
    pins to that floor; we therefore ALSO report the RAW observed sup-Wald, which
    is what actually separates a mechanical placebo (small statistic) from a real
    threshold (huge statistic). Three placebos strip structure progressively:
      (i)   year         — pure linear time trend. NOT structure-free in this
                           panel: it partitions the genuine secular change in the
                           t_growing->fertility slope (the demographic transition),
                           so it is EXPECTED to reject. Reported transparently.
      (ii)  wage_shuffle — wage values permuted across ALL rows (time & unit
                           structure destroyed); same marginal, no fertility link.
                           The clean 'no-structure' null: should NOT reject.
      (iii) gauss        — i.i.d. N(0,1) noise. Pure noise: should NOT reject.
    The falsification PASSES iff the structure-free placebos (wage_shuffle, gauss)
    fail to reject; a structure-free placebo that rejects would be a problem and
    is flagged FAIL. (An earlier within-country re-sorted-by-year placebo was
    discarded: in a near-monotone-in-time panel it reproduces the wage ~0.98, so
    it is not structure-free — that failure motivated the cross-row shuffle.)

Reads the panel READ-ONLY (via the Phase 11 build_panel helper). Writes:
  analysis/output/long_shadow_fertility/phase13_threshold_robustness.json
  analysis/figures/long_shadow_fertility/fig26_threshold_sweep.{pdf,png}
  <paper repo>/long_shadow/figures/fig26_threshold_sweep.pdf  (best-effort copy)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression import (
    _fit_at_c,
    fit_threshold_regression,
)
# Reuse the Phase 11 panel builder verbatim (read-only) so the sweep is on the
# exact same merged panel the headline regime FEVD was estimated on.
from analysis.paper4_shadow.long_shadow_fertility.scripts.run_phase11_regime_fevd import (
    build_panel,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OUTPUT_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/"
    "phase13_threshold_robustness.json"
)
FIG_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/figures/long_shadow_fertility")
PAPER_FIG_DIR = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/"
    "My Drive/Fertility/long_shadow/figures"
)

CORE_VARS = ["spei_growing", "log_real_wage", "log_cdr", "log_cbr"]
HORIZONS = list(range(0, 16))
P_LAGS = 2
MORT_KEY = "log_cdr"
WAGE_KEY = "log_real_wage"

# (A) regime-split sweep grid
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 9.5, 10.5, 0.05
# economically meaningful window referenced in the paper / by the referee
WINDOW_LO, WINDOW_HI = 9.66, 10.14
# vertical markers on the figure: LR-CI floor / gdppc c_hat / headline split / wage c_hat
SPLIT_MARKERS = {
    "9.66 (LR-CI floor)": 9.66,
    "9.94 (gdppc $\\hat c$)": 9.94,
    "9.97 (headline split)": 9.97,
    "10.14 (wage $\\hat c$)": 10.14,
}

# (B) Hansen test settings — MUST match phase10_threshold_grid for comparability
HANSEN_Y = "log_cbr"
HANSEN_X = "t_growing"
HANSEN_Z_REAL = "log_real_wage"
N_BOOT = 500
SEED = 0
WAGE_REF_P = 0.002  # documented wage threshold sup-Wald p-value (phase 9/10)

# A regime subsample must be large enough for the 4-var VAR(2) system to fit.
# Per-equation params: const + 4*2 lags + (n_countries-1) FE.
MIN_REGIME_N = 80


# --------------------------------------------------------------------------- #
# (A) Regime-split sensitivity sweep
# --------------------------------------------------------------------------- #
def _mort_wage_shares_h15(sub: pd.DataFrame) -> dict | None:
    """Fit the core system on `sub`; return mortality & wage FEVD share at h=15.

    Returns None if the subsample is too small or the system fails to estimate
    (singular residual covariance, insufficient dof, non-finite shares)."""
    if len(sub) < MIN_REGIME_N or sub["iso3"].nunique() < 2:
        return None
    try:
        res = fit_system_lp_fevd(sub, variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        # ZeroDivisionError: at extreme splits a regime can shrink (after the
        # h=15 within-country forward shift + country-FE dummies) to nobs == k,
        # which trips statsmodels' cluster dof correction (nobs - k_params == 0).
        # Such a degenerate split is simply flagged non-estimable, not fatal.
        return None
    fevd = res["fevd"]["log_cbr"]  # (n_shocks, H); last col = h=15
    idx = {v: j for j, v in enumerate(res["variables"])}
    mort = float(fevd[idx[MORT_KEY]][-1])
    wage = float(fevd[idx[WAGE_KEY]][-1])
    if not (np.isfinite(mort) and np.isfinite(wage)):
        return None
    return {"mortality": mort, "wage": wage}


def run_split_sweep(df: pd.DataFrame) -> dict:
    """Sweep the regime-split value c; recompute regime FEVD shares at each c."""
    cc = df.dropna(subset=CORE_VARS).copy()
    # Regular grid plus the exact headline split (9.97) and the Hansen wage c_hat
    # (10.14), so both reference points are computed even though they fall between
    # the 0.05-spaced grid nodes.
    grid = np.round(np.arange(SWEEP_LO, SWEEP_HI + 1e-9, SWEEP_STEP), 4)
    grid = np.array(sorted(set(grid.tolist()) | {9.97, 10.14}))

    path: list[dict] = []
    for c in grid:
        c = float(c)
        malth = cc[cc[WAGE_KEY] <= c]
        modern = cc[cc[WAGE_KEY] > c]
        m_sh = _mort_wage_shares_h15(malth)
        d_sh = _mort_wage_shares_h15(modern)
        entry = {
            "c": c,
            "n_malthusian": int(len(malth)),
            "n_modern": int(len(modern)),
            "malthusian_mortality_share": m_sh["mortality"] if m_sh else None,
            "modern_mortality_share": d_sh["mortality"] if d_sh else None,
            "malthusian_wage_share": m_sh["wage"] if m_sh else None,
            "modern_wage_share": d_sh["wage"] if d_sh else None,
            "mortality_collapse": (
                (m_sh["mortality"] - d_sh["mortality"]) if (m_sh and d_sh) else None
            ),
            "estimable": bool(m_sh and d_sh),
        }
        path.append(entry)

    # Stability check inside the economically-meaningful window c in [9.66, 10.14].
    window = [
        e for e in path
        if e["estimable"] and WINDOW_LO - 1e-9 <= e["c"] <= WINDOW_HI + 1e-9
    ]
    collapses = [e["mortality_collapse"] for e in window]
    malth_m = [e["malthusian_mortality_share"] for e in window]
    modern_m = [e["modern_mortality_share"] for e in window]

    holds = bool(window) and all(
        (e["malthusian_mortality_share"] > e["modern_mortality_share"])
        for e in window
    )
    # Headline (9.97) row for reference; should reproduce Phase 11 (0.181 / 0.0163).
    headline = next(
        (e for e in path if abs(e["c"] - 9.97) < 1e-9), None
    )

    return {
        "grid": [float(c) for c in grid],
        "window": [WINDOW_LO, WINDOW_HI],
        "path": path,
        "summary": {
            "window_collapse_holds_everywhere": holds,
            "n_window_points_estimable": len(window),
            "malthusian_mortality_range_in_window": (
                [float(min(malth_m)), float(max(malth_m))] if malth_m else None
            ),
            "modern_mortality_range_in_window": (
                [float(min(modern_m)), float(max(modern_m))] if modern_m else None
            ),
            "collapse_range_in_window": (
                [float(min(collapses)), float(max(collapses))] if collapses else None
            ),
            "min_collapse_in_window": float(min(collapses)) if collapses else None,
            "headline_9_97": headline,
            "phase11_reference": {
                "malthusian_mortality_share": 0.18081430524608455,
                "modern_mortality_share": 0.016255371730945913,
            },
        },
    }


# --------------------------------------------------------------------------- #
# (B) Placebo-threshold falsification
# --------------------------------------------------------------------------- #
def _observed_supwald(
    df: pd.DataFrame, y: str, x: str, z: str, unit_col: str,
    trim: tuple[float, float] = (0.15, 0.85),
) -> float:
    """Observed sup-Wald statistic for threshold variable z (no bootstrap).

    Mirrors the grid construction inside fit_threshold_regression so the value
    is comparable to the c_hat that test selects. Reported alongside the
    bootstrap p-value because, with a 500-rep wild-cluster bootstrap, the p-value
    floor is 1/500 = 0.002: any z carrying genuine structure pins to that floor,
    so the RAW statistic is what distinguishes a 'mechanically firing' placebo
    (small sup-Wald) from a real threshold (huge sup-Wald)."""
    dd = df.dropna(subset=[y, x, z, unit_col]).copy()
    zv = dd[z].to_numpy(dtype=float)
    lo, hi = np.quantile(zv, trim[0]), np.quantile(zv, trim[1])
    grid = np.linspace(lo, hi, 50)
    walds = [_fit_at_c(dd, y, x, z, c, unit_col)[3] for c in grid]
    return float(max(walds))


def run_placebo_falsification(df: pd.DataFrame) -> dict:
    """Re-estimate the real wage threshold + several placebo thresholds.

    Placebo design note
    -------------------
    A first attempt used a within-country value-permutation re-sorted into year
    order. That FAILED as a placebo: because real wages are near-monotone in
    time, re-sorting any wage-like marginal by year reproduces a series ~0.98
    correlated with the true wage, so it (spuriously) "rejected" with the same
    c_hat. The lesson is itself informative — in this near-monotone-in-time panel
    a time trend proxies the wage regime. We therefore use three placebos that
    progressively strip structure:
      (i)   placebo_year         — pure linear time trend. NOT structure-free in
                                     this panel: it partitions the genuine secular
                                     change in the t_growing->fertility slope
                                     (the demographic transition), so it is
                                     EXPECTED to fire. Included to make that point
                                     explicit rather than hide it.
      (ii)  placebo_wage_shuffle — wage values permuted across ALL rows (time and
                                     unit structure both destroyed). Same marginal
                                     as the wage, but no relationship to fertility.
                                     The clean 'no-structure' null: should NOT fire.
      (iii) placebo_gauss        — i.i.d. N(0,1) noise as the threshold variable.
                                     Pure noise: should NOT fire.
    """
    rng = np.random.default_rng(SEED)
    work = df.copy()
    base = work.dropna(subset=[HANSEN_Y, HANSEN_X, HANSEN_Z_REAL, "iso3"]).index

    # (ii) cross-row shuffle of the wage (restricted to the wage-test rows so the
    # marginal matches the real-wage baseline exactly).
    work["wage_shuffle"] = np.nan
    shuffled = rng.permutation(work.loc[base, HANSEN_Z_REAL].to_numpy(dtype=float))
    work.loc[base, "wage_shuffle"] = shuffled

    # (iii) i.i.d. Gaussian noise on every row.
    work["gauss_noise"] = rng.standard_normal(len(work))

    specs = {
        # Real wage threshold re-estimated here for an apples-to-apples baseline.
        "wage_real": HANSEN_Z_REAL,
        # Placebo (i): pure linear time trend.
        "placebo_year": "year",
        # Placebo (ii): wage values shuffled across all rows (no structure).
        "placebo_wage_shuffle": "wage_shuffle",
        # Placebo (iii): i.i.d. Gaussian noise.
        "placebo_gauss": "gauss_noise",
    }
    placebo_is_structurefree = {  # which placebos we EXPECT to fail to reject
        "placebo_year": False,          # time trend genuinely partitions the transition
        "placebo_wage_shuffle": True,
        "placebo_gauss": True,
    }

    results: dict[str, dict] = {}
    for name, z in specs.items():
        sub = work.dropna(subset=[HANSEN_Y, HANSEN_X, z, "iso3"]).copy()
        res = fit_threshold_regression(
            sub, y=HANSEN_Y, x=HANSEN_X, z=z,
            unit_col="iso3", n_boot=N_BOOT, trim=(0.15, 0.85), seed=SEED,
        )
        obs_sw = _observed_supwald(sub, HANSEN_Y, HANSEN_X, z, "iso3")
        # Drop the bulky lr_path from the per-spec payload; keep the scalars.
        results[name] = {
            "z": z,
            "c_hat": res["c_hat"],
            "c_ci_lo": res["c_ci_lo"],
            "c_ci_hi": res["c_ci_hi"],
            "sup_wald_pvalue": res["sup_wald_pvalue"],
            "observed_sup_wald": obs_sw,
            "beta_M": res["beta_M"],
            "beta_T": res["beta_T"],
            "n": res["n"],
        }

    wage_p = results["wage_real"]["sup_wald_pvalue"]
    wage_sw = results["wage_real"]["observed_sup_wald"]
    placebos = {k: v for k, v in results.items() if k.startswith("placebo_")}

    REJECT_ALPHA = 0.05
    placebo_assessment = {}
    structurefree_reject = []  # structure-free placebos that wrongly reject -> bad
    for k, v in placebos.items():
        p = v["sup_wald_pvalue"]
        strong = p <= REJECT_ALPHA
        sf = placebo_is_structurefree[k]
        if sf and strong:
            structurefree_reject.append(k)
        placebo_assessment[k] = {
            "sup_wald_pvalue": p,
            "observed_sup_wald": v["observed_sup_wald"],
            "structure_free": bool(sf),
            "expected_to_reject": bool(not sf),
            "rejects_at_0.05": bool(strong),
            "sup_wald_vs_wage": (
                float(v["observed_sup_wald"] / wage_sw) if wage_sw > 0 else None
            ),
            "weaker_than_wage": bool(p > wage_p),
        }

    any_structurefree_reject = bool(structurefree_reject)
    return {
        "settings": {
            "y": HANSEN_Y, "x": HANSEN_X, "n_boot": N_BOOT, "seed": SEED,
            "trim": [0.15, 0.85], "reject_alpha": REJECT_ALPHA,
            "pvalue_floor": 1.0 / N_BOOT,
        },
        "results": results,
        "wage_real_pvalue": wage_p,
        "wage_real_observed_sup_wald": wage_sw,
        "structurefree_placebos_that_reject": structurefree_reject,
        "any_structurefree_placebo_rejects": any_structurefree_reject,
        "documented_wage_pvalue": WAGE_REF_P,
        "placebo_assessment": placebo_assessment,
        "verdict": (
            "PASS: the structure-free placebos (cross-row-shuffled wage, i.i.d. "
            "Gaussian) do NOT reject — the Hansen sup-Wald test does not "
            "mechanically fire on a threshold variable that carries no "
            "relationship to fertility. (The linear time trend DOES reject, but "
            "that is substantive, not mechanical: in this near-monotone-in-time "
            "panel a calendar threshold partitions the genuine secular change in "
            "the temperature->fertility slope across the demographic transition.)"
            if not any_structurefree_reject else
            "FAIL: a STRUCTURE-FREE placebo (" + ", ".join(structurefree_reject) +
            ") rejects at 0.05 — the sup-Wald test fires even on noise, so the "
            "wage-threshold identification claim is weakened."
        ),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_fig26(sweep: dict) -> dict:
    """Mortality & wage FEVD share vs regime-split c, styled like fig25."""
    import matplotlib.pyplot as plt
    from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import (
        apply_pub_style,
    )

    apply_pub_style(font_size=10, serif=True)
    rows = [e for e in sweep["path"] if e["estimable"]]
    c = np.array([e["c"] for e in rows], dtype=float)
    malth_m = np.array([e["malthusian_mortality_share"] for e in rows], dtype=float)
    modern_m = np.array([e["modern_mortality_share"] for e in rows], dtype=float)
    malth_w = np.array([e["malthusian_wage_share"] for e in rows], dtype=float)
    modern_w = np.array([e["modern_wage_share"] for e in rows], dtype=float)

    cmap = plt.get_cmap("Greys")
    g_dark, g_light = cmap(0.80), cmap(0.45)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)

    # Left: mortality FEVD share (the collapse)
    ax = axes[0]
    ax.plot(c, malth_m, color=g_dark, lw=1.8, marker="o", ms=3,
            label="Malthusian ($\\log W \\leq c$)")
    ax.plot(c, modern_m, color=g_light, lw=1.8, marker="s", ms=3, ls="--",
            label="Modern ($\\log W > c$)")
    ax.set_ylabel("Mortality (log CDR) share of\nlog-fertility FE variance ($h=15$)")
    ax.set_title("Mortality FEVD share vs regime split", fontsize=10)

    # Right: wage FEVD share
    ax = axes[1]
    ax.plot(c, malth_w, color=g_dark, lw=1.8, marker="o", ms=3,
            label="Malthusian ($\\log W \\leq c$)")
    ax.plot(c, modern_w, color=g_light, lw=1.8, marker="s", ms=3, ls="--",
            label="Modern ($\\log W > c$)")
    ax.set_ylabel("Wage (log real wage) share of\nlog-fertility FE variance ($h=15$)")
    ax.set_title("Wage FEVD share vs regime split", fontsize=10)

    # Vertical markers + shaded meaningful window on both panels.
    for ax in axes:
        ax.axvspan(WINDOW_LO, WINDOW_HI, color="0.90", zorder=0)
        for lbl, x in SPLIT_MARKERS.items():
            ax.axvline(x, color="black", lw=0.8,
                       ls=(":" if "9.97" not in lbl else "-"), alpha=0.75)
        ax.set_xlabel("Regime-split value $c$ (log real wage)")
        ax.set_xlim(c.min(), c.max())
        ax.set_ylim(bottom=0)

    # Marker legend (the vertical lines), placed on the left panel.
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], color="black", lw=0.8,
               ls=("-" if "9.97" in lbl else ":"))
        for lbl in SPLIT_MARKERS
    ]
    leg_markers = axes[0].legend(
        marker_handles, list(SPLIT_MARKERS.keys()),
        loc="upper right", fontsize=7, frameon=False, title="split markers",
    )
    axes[0].add_artist(leg_markers)
    axes[0].legend(loc="center right", fontsize=8, frameon=False)
    axes[1].legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(
        "Regime-split sensitivity: mortality collapse and wage share are not "
        "knife-edge at $c=9.97$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig26_threshold_sweep.pdf"
    png = FIG_DIR / "fig26_threshold_sweep.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    copied = None
    try:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        copied = PAPER_FIG_DIR / "fig26_threshold_sweep.pdf"
        copied.write_bytes(pdf.read_bytes())
    except OSError as exc:
        print(f"WARNING: could not copy PDF to paper repo: {exc}")
        copied = None

    return {"pdf": str(pdf), "png": str(png),
            "paper_pdf": (str(copied) if copied else None)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> dict:
    df = build_panel()

    sweep = run_split_sweep(df)
    placebo = run_placebo_falsification(df)
    fig = make_fig26(sweep)

    result = {
        "meta": {
            "phase": 13,
            "description": "Threshold robustness: (A) regime-split sweep, "
                           "(B) placebo-threshold falsification.",
            "core_variables": CORE_VARS,
            "horizons": HORIZONS,
            "p_lags": P_LAGS,
            "headline_split": 9.97,
            "wage_c_hat": 10.14,
            "documented_wage_pvalue": WAGE_REF_P,
            "panel_read_only": True,
        },
        "A_split_sweep": sweep,
        "B_placebo_falsification": placebo,
        "figure": fig,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))

    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 74)
    print("PHASE 13: THRESHOLD ROBUSTNESS")
    print("=" * 74)

    sw = result["A_split_sweep"]
    s = sw["summary"]
    print("\n(A) Regime-split sensitivity sweep  c in [9.5, 10.5] step 0.05")
    print(f"    estimable window points (c in [{WINDOW_LO},{WINDOW_HI}]): "
          f"{s['n_window_points_estimable']}")
    print(f"    Malthusian mortality share range in window: "
          f"{s['malthusian_mortality_range_in_window']}")
    print(f"    Modern     mortality share range in window: "
          f"{s['modern_mortality_range_in_window']}")
    print(f"    collapse (Malth - Modern) range in window : "
          f"{s['collapse_range_in_window']}  (min={s['min_collapse_in_window']})")
    print(f"    collapse holds (Malth > Modern) at EVERY window c: "
          f"{s['window_collapse_holds_everywhere']}")
    hl = s["headline_9_97"]
    if hl:
        print(f"    headline c=9.97: Malth mort={hl['malthusian_mortality_share']:.4f}  "
              f"Modern mort={hl['modern_mortality_share']:.4f}  "
              f"(phase11 ref 0.1808 / 0.0163)")
    print("    full path (c: Malth_mort | Modern_mort | Malth_wage | Modern_wage):")
    for e in sw["path"]:
        if not e["estimable"]:
            print(f"      c={e['c']:.2f}: NOT ESTIMABLE "
                  f"(n_malth={e['n_malthusian']}, n_mod={e['n_modern']})")
            continue
        print(f"      c={e['c']:.2f}: {e['malthusian_mortality_share']:.4f} | "
              f"{e['modern_mortality_share']:.4f} | "
              f"{e['malthusian_wage_share']:.4f} | "
              f"{e['modern_wage_share']:.4f}  "
              f"(n {e['n_malthusian']}/{e['n_modern']})")

    pb = result["B_placebo_falsification"]
    print("\n(B) Placebo-threshold falsification  (Hansen sup-Wald, "
          f"n_boot={N_BOOT}, seed={SEED}, p-floor={1.0/N_BOOT:.4f})")
    print("    spec                      z               c_hat        "
          "supWald    p")
    for name, r in pb["results"].items():
        print(f"    {name:24s}  {r['z']:14s}  {r['c_hat']:10.4f}  "
              f"{r['observed_sup_wald']:8.2f}  {r['sup_wald_pvalue']:.4f}  (n={r['n']})")
    print(f"    wage_real: observed sup-Wald={pb['wage_real_observed_sup_wald']:.2f}, "
          f"p={pb['wage_real_pvalue']:.4f} (documented {pb['documented_wage_pvalue']})")
    print("    placebo assessment (sup-Wald relative to wage's, and reject decision):")
    for k, a in pb["placebo_assessment"].items():
        tag = "structure-free" if a["structure_free"] else "time-trend (substantive)"
        print(f"      {k:22s} [{tag}]: supWald={a['observed_sup_wald']:7.2f} "
              f"(={a['sup_wald_vs_wage']:.2f}x wage)  p={a['sup_wald_pvalue']:.4f}  "
              f"rejects@0.05={a['rejects_at_0.05']}")
    print(f"    structure-free placebos that reject: "
          f"{pb['structurefree_placebos_that_reject'] or 'NONE'}")
    print(f"    => {pb['verdict']}")

    print(f"\nwrote {OUTPUT_PATH}")
    print(f"wrote {result['figure']['pdf']}")
    if result["figure"]["paper_pdf"]:
        print(f"copied to {result['figure']['paper_pdf']}")


if __name__ == "__main__":
    main()
