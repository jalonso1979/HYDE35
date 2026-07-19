"""Phase 13 — Bootstrap CI on the DIFFERENCE in regime FEVD mortality shares.

Referee item 2d. The Phase-11 bootstrap reports *per-regime* mortality-share
CIs (Malthusian [0.117, 0.355] vs Modern [0.0066, 0.101]) and argues they
"separate." Non-overlapping marginal CIs are NOT a difference test. The correct
object is a bootstrap CI on

    Δ_mortality = mortality_share_Malthusian − mortality_share_Modern

computed WITHIN each bootstrap draw, on the SAME resampled set of countries.

Design (faithful to run_phase11_regime_fevd_bootstrap.py)
---------------------------------------------------------
* Same panel build (build_panel), same complete-case dropna on CORE_VARS.
* Same unit of resampling: country cluster, drawn with replacement.
* Same N_BOOT=500, same seed=0, same Hansen threshold, same CORE_VARS,
  same horizons (h=0..15), same p=2, same h-target index (-1, i.e. h=15).
* CRITICAL DIFFERENCE vs Phase 11: a SINGLE draw of 12 country clusters is
  used for BOTH regimes within that draw. We build one stacked pseudo-panel
  (duplicated countries -> distinct pseudo-iso3 labels so FE work), then split
  that stacked panel by `regime` and run fit_system_lp_fevd separately on the
  Malthusian rows and the Modern rows. Δ is recorded per draw. This pairs the
  two regimes on identical resampled countries, which is exactly what a
  difference test requires. (All 12 countries appear in both regimes, so every
  draw yields non-empty rows on both sides.)

For completeness we also record Δ_wage = wage_share_Modern − wage_share_Malthusian.

Reports
-------
* Point estimate of Δ_mortality (from the unresampled panel).
* 95% percentile CI of Δ_mortality across draws.
* One-sided bootstrap p-value for H0: Δ_mortality <= 0, computed as the share
  of bootstrap draws with Δ_mortality <= 0 (i.e. fraction of the bootstrap
  distribution that lands on the null side).
* Same summary for Δ_wage.

Output: JSON at
  /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase13_bootstrap_diff.json

Reads the panel READ-ONLY. Does not modify any estimator or existing script.

Usage:
    python -m analysis.paper4_shadow.long_shadow_fertility.scripts.run_phase13_bootstrap_diff
or run directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Volumes/BIGDATA/HYDE35")

# Reuse the EXACT panel build + config from the Phase-11 bootstrap so the
# resampling universe is identical. We import (not copy) to guarantee parity.
from analysis.paper4_shadow.long_shadow_fertility.scripts.run_phase11_regime_fevd_bootstrap import (  # noqa: E501
    CORE_VARS,
    HANSEN_THRESHOLD,
    HORIZONS,
    H_TARGET_IDX,
    N_BOOT,
    P_LAGS,
    SEED,
    SHOCK_LABELS,
    build_panel,
)
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)

OUTPUT_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/"
    "phase13_bootstrap_diff.json"
)

# Label -> variable map (mirrors zip(SHOCK_LABELS, CORE_VARS) in Phase 11)
LABEL_TO_VAR = dict(zip(SHOCK_LABELS, CORE_VARS))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fevd_shares_at_h(res: dict, outcome: str, h_idx: int) -> dict[str, float]:
    """Extract FEVD shares for `outcome` at horizon index `h_idx` (var-keyed)."""
    variables = res["variables"]
    fevd = res["fevd"][outcome]  # ndarray (n_shocks, H)
    return {v: float(fevd[j][h_idx]) for j, v in enumerate(variables)}


def _shares_by_label(res: dict) -> dict[str, float]:
    """h=15 fertility FEVD shares mapped to SHOCK_LABELS order."""
    var_map = _fevd_shares_at_h(res, "log_cbr", H_TARGET_IDX)
    return {label: var_map[var] for label, var in LABEL_TO_VAR.items()}


def _build_stacked_panel(
    cc: pd.DataFrame, draw: list[str]
) -> pd.DataFrame:
    """Stack the full (both-regime) country series for the bootstrap draw.

    Each occurrence of a country gets a distinct pseudo-iso3 label so that
    duplicated countries get separate FE dummies (e.g. "FRA__0", "FRA__1").
    Mirrors _build_bootstrap_panel in Phase 11, but stacks ALL rows for the
    country (both regimes) so we can split afterwards.
    """
    frames = []
    counter: dict[str, int] = {}
    for iso in draw:
        n = counter.get(iso, 0)
        counter[iso] = n + 1
        ctry_rows = cc[cc["iso3"] == iso].copy()
        ctry_rows["iso3"] = f"{iso}__{n}"
        frames.append(ctry_rows)
    return pd.concat(frames, ignore_index=True)


def _shares_for_regime(stacked: pd.DataFrame, regime: int) -> dict[str, float] | None:
    """Run FEVD on the regime-subset of the stacked panel; None on failure."""
    sub = stacked[stacked["regime"] == regime]
    res = fit_system_lp_fevd(sub, variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS)
    shares = _shares_by_label(res)
    if not all(np.isfinite(v) for v in shares.values()):
        return None
    return shares


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> dict:
    print("Building panel …", flush=True)
    df = build_panel()
    cc = df.dropna(subset=CORE_VARS).copy()
    countries = sorted(cc["iso3"].unique())
    n_countries = len(countries)
    n_mal = int((cc["regime"] == 0).sum())
    n_mod = int((cc["regime"] == 1).sum())
    print(f"Complete-case panel: N={len(cc)}, countries={countries}")
    print(f"Regime split: Malthusian N={n_mal}, Modern N={n_mod}")
    print(f"Resampling universe: {n_countries} country clusters (both regimes).")

    # ── Point estimate (unresampled) ───────────────────────────────────────────
    print("\nComputing point estimates (unresampled, by regime) …", flush=True)
    mal_point = _shares_by_label(
        fit_system_lp_fevd(
            cc[cc["regime"] == 0], variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS
        )
    )
    mod_point = _shares_by_label(
        fit_system_lp_fevd(
            cc[cc["regime"] == 1], variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS
        )
    )
    point_diff_mortality = mal_point["mortality"] - mod_point["mortality"]
    point_diff_wage = mod_point["wages"] - mal_point["wages"]
    print(
        f"  Malthusian mortality={mal_point['mortality']:.4f}, "
        f"Modern mortality={mod_point['mortality']:.4f}  "
        f"=> Δ_mortality(point)={point_diff_mortality:.4f}"
    )
    print(
        f"  Malthusian wages={mal_point['wages']:.4f}, "
        f"Modern wages={mod_point['wages']:.4f}  "
        f"=> Δ_wage(point)={point_diff_wage:.4f}"
    )

    # ── Bootstrap loop ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    print(
        f"\nRunning paired block-bootstrap with n_boot={N_BOOT}, seed={SEED} "
        f"(one 12-cluster draw per iteration, split by regime) …",
        flush=True,
    )

    diff_mortality: list[float] = []
    diff_wage: list[float] = []
    mal_mort_draws: list[float] = []
    mod_mort_draws: list[float] = []
    n_failed = 0

    for b in range(N_BOOT):
        if (b + 1) % 25 == 0:
            print(
                f"  draw {b+1}/{N_BOOT}  "
                f"(succeeded={len(diff_mortality)}, failed={n_failed})",
                flush=True,
            )
        draw = list(rng.choice(countries, size=n_countries, replace=True))
        try:
            stacked = _build_stacked_panel(cc, draw)
            mal_shares = _shares_for_regime(stacked, regime=0)
            mod_shares = _shares_for_regime(stacked, regime=1)
            if mal_shares is None or mod_shares is None:
                raise ValueError("non-finite FEVD shares in a regime")
            d_mort = mal_shares["mortality"] - mod_shares["mortality"]
            d_wage = mod_shares["wages"] - mal_shares["wages"]
            if not (np.isfinite(d_mort) and np.isfinite(d_wage)):
                raise ValueError("non-finite difference")
            diff_mortality.append(d_mort)
            diff_wage.append(d_wage)
            mal_mort_draws.append(mal_shares["mortality"])
            mod_mort_draws.append(mod_shares["mortality"])
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            if n_failed <= 5:
                print(f"    draw {b} FAILED: {exc}", flush=True)

    n_succeeded = len(diff_mortality)
    print(
        f"\nBootstrap done: succeeded={n_succeeded}, failed={n_failed}",
        flush=True,
    )

    # ── Summarise ────────────────────────────────────────────────────────────────
    def _summary(arr_list: list[float], point: float) -> dict:
        a = np.asarray(arr_list, dtype=float)
        if a.size == 0:
            return {
                "point": round(point, 6),
                "boot_mean": None,
                "ci_lo": None,
                "ci_med": None,
                "ci_hi": None,
            }
        lo, med, hi = np.percentile(a, [2.5, 50.0, 97.5])
        return {
            "point": round(point, 6),
            "boot_mean": round(float(a.mean()), 6),
            "ci_lo": round(float(lo), 6),
            "ci_med": round(float(med), 6),
            "ci_hi": round(float(hi), 6),
        }

    dm = np.asarray(diff_mortality, dtype=float)
    dw = np.asarray(diff_wage, dtype=float)

    # One-sided bootstrap p-value for H0: Δ <= 0 -> fraction of draws at/below 0.
    p_mort = float((dm <= 0).mean()) if dm.size else None
    p_wage = float((dw <= 0).mean()) if dw.size else None

    out = {
        "meta": {
            "description": (
                "Phase 13. Paired block-bootstrap (cluster=country) CI on the "
                "WITHIN-DRAW DIFFERENCE in h=15 fertility FEVD mortality share "
                "between Hansen wage regimes: "
                "Δ_mortality = mortality_share_Malthusian − mortality_share_Modern. "
                "A single 12-country-cluster draw is split by regime each "
                "iteration so the two regime estimates are computed on identical "
                "resampled countries. Also reports "
                "Δ_wage = wage_share_Modern − wage_share_Malthusian. "
                "One-sided bootstrap p-value = share of draws with Δ <= 0. "
                "Mirrors run_phase11_regime_fevd_bootstrap.py settings."
            ),
            "variables": CORE_VARS,
            "shock_labels": SHOCK_LABELS,
            "horizons": HORIZONS,
            "h_target": 15,
            "p_lags": P_LAGS,
            "hansen_threshold": HANSEN_THRESHOLD,
            "n_boot": N_BOOT,
            "seed": SEED,
            "n_clusters": n_countries,
            "countries": countries,
            "n_obs_total": int(len(cc)),
            "n_obs_malthusian": n_mal,
            "n_obs_modern": n_mod,
            "n_succeeded": n_succeeded,
            "n_failed": n_failed,
            "p_value_definition": "one-sided H0: Δ <= 0 ; p = mean(boot_draw <= 0)",
        },
        "delta_mortality": {
            **_summary(diff_mortality, point_diff_mortality),
            "p_one_sided_le0": (round(p_mort, 6) if p_mort is not None else None),
        },
        "delta_wage": {
            **_summary(diff_wage, point_diff_wage),
            "p_one_sided_le0": (round(p_wage, 6) if p_wage is not None else None),
        },
        "marginal_mortality_shares": {
            "malthusian": _summary(mal_mort_draws, mal_point["mortality"]),
            "modern": _summary(mod_mort_draws, mod_point["mortality"]),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {OUTPUT_PATH}")

    _print_table(out)
    return out


def _print_table(out: dict) -> None:
    print("\n" + "=" * 78)
    print("PHASE 13 — BOOTSTRAP CI ON WITHIN-DRAW REGIME DIFFERENCE (h=15 fertility)")
    print("=" * 78)
    dm = out["delta_mortality"]
    dw = out["delta_wage"]
    print(
        f"Δ_mortality (Malth − Modern): point={dm['point']:.4f}  "
        f"95% CI=[{dm['ci_lo']:.4f}, {dm['ci_hi']:.4f}]  "
        f"median={dm['ci_med']:.4f}  one-sided p(Δ<=0)={dm['p_one_sided_le0']}"
    )
    print(
        f"Δ_wage     (Modern − Malth):  point={dw['point']:.4f}  "
        f"95% CI=[{dw['ci_lo']:.4f}, {dw['ci_hi']:.4f}]  "
        f"median={dw['ci_med']:.4f}  one-sided p(Δ<=0)={dw['p_one_sided_le0']}"
    )
    print(
        f"\nDraws: succeeded={out['meta']['n_succeeded']}/"
        f"{out['meta']['n_boot']}, failed={out['meta']['n_failed']}"
    )
    includes_zero = (dm["ci_lo"] is not None) and (dm["ci_lo"] <= 0.0 <= dm["ci_hi"])
    print(
        "Δ_mortality 95% CI "
        + ("INCLUDES 0 (difference not significant at 5%)."
           if includes_zero
           else "EXCLUDES 0 (difference significant at 5%).")
    )


if __name__ == "__main__":
    main()
