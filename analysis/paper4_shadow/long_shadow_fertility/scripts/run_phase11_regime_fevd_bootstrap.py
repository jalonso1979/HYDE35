"""Phase 11 — Block-bootstrap confidence intervals on regime-FEVD shares.

Unit of resampling: country (cluster/block bootstrap, to respect within-country
serial correlation). For each regime we draw 12 iso3 codes with replacement,
stack the full series for those countries (giving duplicated countries distinct
pseudo-iso3 labels so country FE work), restrict to the regime's rows, and run
fit_system_lp_fevd. We record the h=15 fertility FEVD shares for
[spei_growing, log_real_wage, log_cdr, log_cbr].

Output: JSON at /Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/
        phase11_regime_fevd_bootstrap.json

Usage:
    python -m analysis.paper4_shadow.long_shadow_fertility.scripts.run_phase11_regime_fevd_bootstrap
or run directly.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Volumes/BIGDATA/HYDE35")

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_country_mortality_annual import (
    build_country_mortality_annual,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.data.spatial_war import add_nearby_war
from analysis.paper4_shadow.long_shadow_fertility.estimators.system_lp_fevd import (
    fit_system_lp_fevd,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (mirror run_phase11_regime_fevd.py)
# ─────────────────────────────────────────────────────────────────────────────
HANSEN_THRESHOLD = 9.97
HORIZONS = list(range(0, 16))          # h = 0..15; index -1 = h=15
H_TARGET_IDX = -1                      # h=15
P_LAGS = 2
CORE_VARS = ["spei_growing", "log_real_wage", "log_cdr", "log_cbr"]
SHOCK_LABELS = ["weather", "wages", "mortality", "own"]   # maps to CORE_VARS order
N_BOOT = 500
SEED = 0

OUTPUT_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/"
    "phase11_regime_fevd_bootstrap.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data build (mirrors run_phase11_regime_fevd.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_panel() -> pd.DataFrame:
    panel = assemble_panel_multi()
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(mort, on=["iso3", "year"], how="left").merge(
        wage, on=["iso3", "year"], how="left"
    )
    df = add_nearby_war(df, intensity_col="log_war_fatalities")
    df["regime"] = (df["log_real_wage"] > HANSEN_THRESHOLD).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fevd_shares_at_h(res: dict, outcome: str, h_idx: int) -> dict[str, float]:
    """Extract FEVD shares for `outcome` at horizon index `h_idx`."""
    variables = res["variables"]
    fevd = res["fevd"][outcome]   # ndarray (n_shocks, H)
    return {v: float(fevd[j][h_idx]) for j, v in enumerate(variables)}


def _build_bootstrap_panel(
    sub: pd.DataFrame,
    countries: list[str],
    draw: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Stack country series for the bootstrap draw.

    Each occurrence of a country gets a distinct pseudo-iso3 label so that
    duplicated countries get separate FE dummies (e.g. "FRA__0", "FRA__1").
    """
    frames = []
    counter: dict[str, int] = {}
    for iso in draw:
        n = counter.get(iso, 0)
        counter[iso] = n + 1
        ctry_rows = sub[sub["iso3"] == iso].copy()
        ctry_rows["iso3"] = f"{iso}__{n}"
        frames.append(ctry_rows)
    return pd.concat(frames, ignore_index=True)


def run_bootstrap(
    cc: pd.DataFrame,
    regime: int,
    regime_label: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Block-bootstrap FEVD shares for a single regime.

    Parameters
    ----------
    cc    : complete-case panel (already dropna'd on CORE_VARS)
    regime: 0 = Malthusian, 1 = Modern
    """
    sub = cc[cc["regime"] == regime].copy()
    countries = list(sub["iso3"].unique())
    n_countries = len(countries)
    n_obs = len(sub)

    print(
        f"\n[{regime_label}] N={n_obs}, countries={countries}, n_boot={n_boot}"
    )

    # ── Point estimate ────────────────────────────────────────────────────────
    print(f"  Computing point estimate …", flush=True)
    res_point = fit_system_lp_fevd(
        sub, variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS
    )
    point_map = _fevd_shares_at_h(res_point, "log_cbr", H_TARGET_IDX)
    # Map from variable name → SHOCK_LABELS order
    point_by_label = {
        label: point_map[var]
        for label, var in zip(SHOCK_LABELS, CORE_VARS)
    }
    print(f"  Point h=15 shares: " +
          ", ".join(f"{k}={v:.3f}" for k, v in point_by_label.items()))

    # ── Bootstrap loop ────────────────────────────────────────────────────────
    boot_records: list[dict[str, float]] = []   # one dict per successful draw
    n_failed = 0

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(
                f"  [{regime_label}] draw {b+1}/{n_boot}  "
                f"(succeeded={len(boot_records)}, failed={n_failed})",
                flush=True,
            )
        draw = list(rng.choice(countries, size=n_countries, replace=True))
        boot_panel = _build_bootstrap_panel(sub, countries, draw, rng)

        # Restrict to regime rows (already in sub; after relabeling, all rows are regime)
        try:
            res_b = fit_system_lp_fevd(
                boot_panel, variables=CORE_VARS, horizons=HORIZONS, p=P_LAGS
            )
            shares_b = _fevd_shares_at_h(res_b, "log_cbr", H_TARGET_IDX)
            # Validity check: all finite and sum ~1
            vals = [shares_b[v] for v in CORE_VARS]
            if not all(np.isfinite(v) for v in vals):
                raise ValueError(f"non-finite shares: {shares_b}")
            boot_records.append(
                {label: shares_b[var] for label, var in zip(SHOCK_LABELS, CORE_VARS)}
            )
        except Exception as exc:
            n_failed += 1
            if n_failed <= 5:
                print(f"    draw {b} FAILED: {exc}", flush=True)

    n_succeeded = len(boot_records)
    print(
        f"  [{regime_label}] done: succeeded={n_succeeded}, failed={n_failed}",
        flush=True,
    )

    if n_succeeded < 10:
        print(
            f"  WARNING: fewer than 10 successful draws for {regime_label}; "
            "CIs will be unreliable."
        )

    # ── Summarise ─────────────────────────────────────────────────────────────
    out: dict = {"n_obs": n_obs, "n_countries": n_countries,
                 "n_boot": n_boot, "n_failed": n_failed,
                 "n_succeeded": n_succeeded}

    arr = {label: np.array([r[label] for r in boot_records]) for label in SHOCK_LABELS}
    for label in SHOCK_LABELS:
        a = arr[label]
        if len(a) == 0:
            out[label] = {
                "point": round(point_by_label[label], 5),
                "ci_lo": None, "ci_med": None, "ci_hi": None,
            }
        else:
            p2, p50, p97 = np.percentile(a, [2.5, 50.0, 97.5])
            out[label] = {
                "point": round(point_by_label[label], 5),
                "ci_lo":  round(float(p2),  5),
                "ci_med": round(float(p50), 5),
                "ci_hi":  round(float(p97), 5),
            }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> dict:
    print("Building panel …", flush=True)
    df = build_panel()
    cc = df.dropna(subset=CORE_VARS).copy()
    print(f"Complete-case panel: N={len(cc)}, countries={sorted(cc['iso3'].unique())}")

    n_mal = int((cc["regime"] == 0).sum())
    n_mod = int((cc["regime"] == 1).sum())
    print(f"Regime split: Malthusian N={n_mal}, Modern N={n_mod}")

    rng = np.random.default_rng(SEED)

    print(f"\nRunning block-bootstrap with n_boot={N_BOOT}, seed={SEED} …", flush=True)
    mal_result = run_bootstrap(cc, regime=0, regime_label="malthusian",
                               n_boot=N_BOOT, rng=rng)
    mod_result = run_bootstrap(cc, regime=1, regime_label="modern",
                               n_boot=N_BOOT, rng=rng)

    output = {
        "meta": {
            "description": (
                "Block-bootstrap (cluster by country) 95% CIs for h=15 fertility "
                "FEVD shares by Hansen wage regime. Unit of resampling = country. "
                "Core system: " + str(CORE_VARS) + ". "
                f"Hansen threshold: {HANSEN_THRESHOLD}. "
                f"n_boot={N_BOOT}, seed={SEED}."
            ),
            "variables": CORE_VARS,
            "shock_labels": SHOCK_LABELS,
            "horizons": HORIZONS,
            "h_target": 15,
            "p_lags": P_LAGS,
            "hansen_threshold": HANSEN_THRESHOLD,
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "malthusian": mal_result,
        "modern": mod_result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWritten to {OUTPUT_PATH}")

    _print_ci_table(output)
    return output


def _print_ci_table(output: dict) -> None:
    print("\n" + "=" * 75)
    print("BLOCK-BOOTSTRAP 95% CI TABLE — h=15 fertility FEVD shares by regime")
    print("=" * 75)
    hdr = f"{'Shock':<14} {'Regime':<14} {'Point':>8} {'CI_lo':>8} {'CI_med':>8} {'CI_hi':>8}"
    print(hdr)
    print("-" * 75)
    for label, regime_key, regime_name in [
        (None, "malthusian", "Malthusian"),
        (None, "modern",     "Modern"),
    ]:
        r = output[regime_key]
        for shock in ["weather", "wages", "mortality", "own"]:
            d = r[shock]
            lo  = f"{d['ci_lo']:.3f}"  if d["ci_lo"]  is not None else "—"
            med = f"{d['ci_med']:.3f}" if d["ci_med"] is not None else "—"
            hi  = f"{d['ci_hi']:.3f}"  if d["ci_hi"]  is not None else "—"
            print(
                f"{shock:<14} {regime_name:<14} {d['point']:>8.3f} "
                f"{lo:>8} {med:>8} {hi:>8}"
            )
        n_ok  = r['n_succeeded']
        n_tot = r['n_boot']
        n_obs = r['n_obs']
        print(f"  → N_obs={n_obs}, boot draws succeeded={n_ok}/{n_tot}")
        print()


if __name__ == "__main__":
    main()
