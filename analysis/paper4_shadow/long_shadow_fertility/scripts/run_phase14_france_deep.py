"""Phase 14: re-run the headline pieces on the France-deepened panel (Blayo 1740-).

Reuses the Phase 10 (Hansen threshold grid) and Phase 11 (regime LP-FEVD)
machinery verbatim, but on a panel whose France vital series has been spliced
back to 1740 from Blayo (1975). See ``data/build_france_deep_annual.py``.

Baseline outputs are NOT clobbered: this writes to
  output/.../phase14_france_deep_threshold_grid.json
  output/.../phase14_france_deep_regime_fevd.json
and the caller restores the baseline source parquets afterward.

Run as a module from /Volumes/BIGDATA/HYDE35:

  python -m analysis.paper4_shadow.long_shadow_fertility.scripts.run_phase14_france_deep \
      [--death-correction] [--harmonize] [--tag raw]

The France-deepened parquets must already be on disk (written by
build_france_deep_annual --write) BEFORE running this.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper4_shadow.long_shadow_fertility.data.assemble_panel_multi import (
    assemble_panel_multi,
)
from analysis.paper4_shadow.long_shadow_fertility.data.build_real_wage_panel_v2 import (
    build_real_wage_panel_v2,
)
from analysis.paper4_shadow.long_shadow_fertility.data.spatial_war import add_nearby_war
from analysis.paper4_shadow.long_shadow_fertility.estimators.threshold_regression_grid import (
    fit_threshold_grid,
)
from analysis.paper4_shadow.long_shadow_fertility.scripts import run_phase11_regime_fevd as p11

OUT_DIR = Path("/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility")
MORTALITY_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/country_mortality_annual.parquet"
)
PANEL_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
)
HANSEN_THRESHOLD = p11.HANSEN_THRESHOLD


# ---------------------------------------------------------------------------
# Panel builder: identical to p11.build_panel, but mortality is read from the
# (deepened) parquet on disk rather than regenerated from HMD by
# build_country_mortality_annual(). This is the ONLY change vs Phase 11.
# ---------------------------------------------------------------------------
def build_panel_deep() -> pd.DataFrame:
    panel = assemble_panel_multi()                       # reads deepened FRA fertility parquet
    mort = pd.read_parquet(MORTALITY_PATH)[["iso3", "year", "log_cdr"]]  # deepened FRA cdr
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(mort, on=["iso3", "year"], how="left").merge(
        wage, on=["iso3", "year"], how="left"
    )
    df = add_nearby_war(df, intensity_col="log_war_fatalities")
    df["regime"] = (df["log_real_wage"] > HANSEN_THRESHOLD).astype(int)
    return df


# ---------------------------------------------------------------------------
# Phase 10 threshold grid on the deepened panel
# ---------------------------------------------------------------------------
def run_threshold_grid(df: pd.DataFrame, out_path: Path) -> dict:
    z_wanted = ["log_real_wage", "log_cdr", "log_gdppc", "log_tfr"]
    z_candidates = [z for z in z_wanted if z in df.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_threshold_grid(df, y="log_cbr", x="t_growing",
                                 z_candidates=z_candidates, n_boot=500, seed=0)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    out_path.write_text(json.dumps(_clean(out), indent=2))
    return out


# ---------------------------------------------------------------------------
# Phase 11 regime FEVD on the deepened panel (reuses p11 run_* functions)
# ---------------------------------------------------------------------------
def run_regime_fevd(df: pd.DataFrame, out_path: Path) -> dict:
    result = {
        "meta": {
            "hansen_threshold": HANSEN_THRESHOLD,
            "horizons": p11.HORIZONS,
            "p_lags": p11.P_LAGS,
            "core_variables": p11.CORE_VARS,
            "extended_variables": p11.EXTENDED_VARS,
            "regime_definition": "1{log_real_wage > 9.97} (1=Modern, 0=Malthusian)",
            "note": "France deepened to 1740 via Blayo (1975) splice; see build_france_deep_annual.py",
        },
        "core_by_regime": p11.run_core_by_regime(df, p11.CORE_VARS, "spei_growing"),
        "core_by_regime_temperature": p11.run_core_by_regime(df, p11.CORE_VARS_TEMP, "t_growing"),
        "extended_pooled": p11.run_extended_pooled(df),
        "extended_by_regime": p11.run_extended_by_regime(df),
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="raw", help="label for stdout (raw|corrected|...)")
    args = ap.parse_args()

    df = build_panel_deep()

    # diagnostics: France pre-1806 entry into the core complete-case sample
    cc = df.dropna(subset=p11.CORE_VARS)
    fra_new = cc[(cc["iso3"] == "FRA") & (cc["year"] < 1806)]
    print(f"[{args.tag}] core complete-case N = {len(cc)}  "
          f"(Malthusian {int((cc['regime'] == 0).sum())}, Modern {int((cc['regime'] == 1).sum())})")
    print(f"[{args.tag}] NEW France <1806 country-years in core = {len(fra_new)}  "
          f"({fra_new['year'].min() if len(fra_new) else '-'}-"
          f"{fra_new['year'].max() if len(fra_new) else '-'}); "
          f"Malthusian among them = {int((fra_new['regime'] == 0).sum())}")

    thr_path = OUT_DIR / "phase14_france_deep_threshold_grid.json"
    fevd_path = OUT_DIR / "phase14_france_deep_regime_fevd.json"
    grid = run_threshold_grid(df, thr_path)
    fevd = run_regime_fevd(df, fevd_path)

    rw = grid.get("log_real_wage", {})
    cdr = grid.get("log_cdr", {})
    print(f"[{args.tag}] Hansen wage: c_hat={rw.get('c_hat'):.4f} p={rw.get('sup_wald_pvalue'):.4f} "
          f"beta_M={rw.get('beta_M'):.4f} beta_T={rw.get('beta_T'):.4f} n={rw.get('n')}")
    print(f"[{args.tag}] Hansen cdr : c_hat={cdr.get('c_hat'):.4f} p={cdr.get('sup_wald_pvalue'):.4f} n={cdr.get('n')}")
    core = fevd["core_by_regime"]
    mal, mod = core["malthusian"], core["modern"]
    print(f"[{args.tag}] regime-FEVD core N={core['n_total']} "
          f"(Malthusian {mal['n']}, Modern {mod['n']})")
    print(f"[{args.tag}] mortality (log_cdr) FEVD share of fertility @h15: "
          f"Malthusian={mal['fevd_log_cbr_h15']['log_cdr']:.4f} -> "
          f"Modern={mod['fevd_log_cbr_h15']['log_cdr']:.4f}")
    print(f"  wrote {thr_path}")
    print(f"  wrote {fevd_path}")


if __name__ == "__main__":
    main()
