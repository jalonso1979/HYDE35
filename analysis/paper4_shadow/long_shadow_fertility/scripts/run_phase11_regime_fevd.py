"""Phase 11 headline: state-dependent system LP-FEVD by development regime.

State = Hansen wage regime, S_ct = 1{log_real_wage_ct > 9.97}
  regime 0 = Malthusian (below threshold)
  regime 1 = Modern     (above threshold)

Three decompositions, all persisted to JSON:

  (A) core_by_regime          : [spei_growing, log_real_wage, log_cdr, log_cbr]
                                 estimated SEPARATELY on each regime subsample.
                                 HEADLINE. Does the weather share of fertility
                                 variance rise from Malthusian to Modern?
  (A') core_by_regime_temperature : same, with t_growing replacing spei_growing
                                 (v0.6: the regime change is temperature-driven).
  (B) extended_pooled         : [nearby_war, log_war_fatalities, pandemic_active,
                                 disaster_count, spei_growing, log_real_wage,
                                 log_cdr, log_cbr] on the FULL pooled sample.
                                 All shocks compete for fertility variance.
  (C) extended_by_regime      : the 8-var system split by regime IF dof permits.
                                 Fallback: skip + record status if a regime has
                                 fewer than 10 x n_params obs, or estimation fails.

Cholesky order = order of `variables` (weather most exogenous -> fertility last;
catastrophe block ordered first in the extended system).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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

OUTPUT_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase11_regime_fevd.json"
)
HANSEN_THRESHOLD = 9.97
HORIZONS = list(range(0, 16))
P_LAGS = 2

CORE_VARS = ["spei_growing", "log_real_wage", "log_cdr", "log_cbr"]
CORE_VARS_TEMP = ["t_growing", "log_real_wage", "log_cdr", "log_cbr"]
EXTENDED_VARS = [
    "nearby_war",
    "log_war_fatalities",
    "pandemic_active",
    "disaster_count",
    "spei_growing",
    "log_real_wage",
    "log_cdr",
    "log_cbr",
]


def build_panel() -> pd.DataFrame:
    """Merge fertility/climate/catastrophe panel + wages + mortality + nearby-war."""
    panel = assemble_panel_multi()
    mort = build_country_mortality_annual()[["iso3", "year", "log_cdr"]]
    wage = build_real_wage_panel_v2()[["iso3", "year", "log_real_wage"]]
    df = panel.merge(mort, on=["iso3", "year"], how="left").merge(
        wage, on=["iso3", "year"], how="left"
    )
    df = add_nearby_war(df, intensity_col="log_war_fatalities")
    df["regime"] = (df["log_real_wage"] > HANSEN_THRESHOLD).astype(int)
    return df


def _fevd_shares_at(res: dict, outcome: str, horizon_idx: int) -> dict[str, float]:
    """Fertility (outcome) FEVD shares by shock at a single horizon index."""
    variables = res["variables"]
    fevd = res["fevd"][outcome]  # (n_shocks, H)
    return {v: float(fevd[j][horizon_idx]) for j, v in enumerate(variables)}


def _fevd_path(res: dict, outcome: str) -> dict[str, list[float]]:
    """Full horizon path of fertility (outcome) FEVD shares by shock."""
    variables = res["variables"]
    fevd = res["fevd"][outcome]
    return {v: [float(x) for x in fevd[j]] for j, v in enumerate(variables)}


def _n_params_per_eq(n_vars: int, p: int, n_countries: int) -> int:
    """Per-equation parameter count: const + n_vars*p lags + (n_countries-1) FE dummies."""
    return 1 + n_vars * p + (n_countries - 1)


def run_core_by_regime(df: pd.DataFrame, variables: list[str], climate_col: str) -> dict:
    """Estimate the 4-var core system separately on each regime subsample."""
    cc = df.dropna(subset=variables).copy()
    out: dict = {
        "variables": variables,
        "climate_var": climate_col,
        "horizons": HORIZONS,
        "n_total": int(len(cc)),
    }
    for regime, label in ((0, "malthusian"), (1, "modern")):
        sub = cc[cc["regime"] == regime].copy()
        res = fit_system_lp_fevd(sub, variables=variables, horizons=HORIZONS, p=P_LAGS)
        out[label] = {
            "n": int(len(sub)),
            "fevd_log_cbr_h15": _fevd_shares_at(res, "log_cbr", -1),
            "fevd_log_cbr_path": _fevd_path(res, "log_cbr"),
        }
    return out


def run_extended_pooled(df: pd.DataFrame) -> dict:
    """Estimate the 8-var extended system on the full pooled complete-case sample."""
    cc = df.dropna(subset=EXTENDED_VARS).copy()
    n_countries = cc["iso3"].nunique()
    res = fit_system_lp_fevd(cc, variables=EXTENDED_VARS, horizons=HORIZONS, p=P_LAGS)
    return {
        "variables": EXTENDED_VARS,
        "horizons": HORIZONS,
        "n": int(len(cc)),
        "n_params_per_eq": _n_params_per_eq(len(EXTENDED_VARS), P_LAGS, n_countries),
        "fevd_log_cbr_h15": _fevd_shares_at(res, "log_cbr", -1),
        "fevd_log_cbr_path": _fevd_path(res, "log_cbr"),
    }


def run_extended_by_regime(df: pd.DataFrame) -> dict:
    """Estimate the 8-var system per regime IF dof permits; else record fallback."""
    cc = df.dropna(subset=EXTENDED_VARS).copy()
    out: dict = {
        "variables": EXTENDED_VARS,
        "horizons": HORIZONS,
    }
    for regime, label in ((0, "malthusian"), (1, "modern")):
        sub = cc[cc["regime"] == regime].copy()
        n = int(len(sub))
        n_countries = sub["iso3"].nunique()
        n_params = _n_params_per_eq(len(EXTENDED_VARS), P_LAGS, n_countries)
        dof_floor = 10 * n_params
        if n < dof_floor:
            out[label] = {
                "status": "insufficient_dof",
                "n": n,
                "n_params_per_eq": n_params,
                "dof_floor": dof_floor,
            }
            continue
        try:
            res = fit_system_lp_fevd(sub, variables=EXTENDED_VARS, horizons=HORIZONS, p=P_LAGS)
            shares = _fevd_shares_at(res, "log_cbr", -1)
            if not all(np.isfinite(v) for v in shares.values()):
                raise ValueError("non-finite FEVD shares")
            out[label] = {
                "status": "ok",
                "n": n,
                "n_params_per_eq": n_params,
                "dof_floor": dof_floor,
                "fevd_log_cbr_h15": shares,
                "fevd_log_cbr_path": _fevd_path(res, "log_cbr"),
            }
        except (np.linalg.LinAlgError, ValueError) as exc:
            out[label] = {
                "status": "insufficient_dof",
                "n": n,
                "n_params_per_eq": n_params,
                "dof_floor": dof_floor,
                "error": str(exc),
            }
    return out


def main() -> dict:
    df = build_panel()

    result = {
        "meta": {
            "hansen_threshold": HANSEN_THRESHOLD,
            "horizons": HORIZONS,
            "p_lags": P_LAGS,
            "core_variables": CORE_VARS,
            "extended_variables": EXTENDED_VARS,
            "regime_definition": "1{log_real_wage > 9.97} (1=Modern, 0=Malthusian)",
            "cholesky_note": "order = order of variables; weather exogenous -> fertility last",
            "disaster_count_caveat": (
                "disaster_count is EMDAT-era only (1900-2022); extended system is "
                "restricted to the EMDAT complete-case sample rather than fabricating "
                "pre-1900 zero-disaster years."
            ),
        },
        "core_by_regime": run_core_by_regime(df, CORE_VARS, "spei_growing"),
        "core_by_regime_temperature": run_core_by_regime(df, CORE_VARS_TEMP, "t_growing"),
        "extended_pooled": run_extended_pooled(df),
        "extended_by_regime": run_extended_by_regime(df),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))

    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    core = result["core_by_regime"]
    nm, nmod = core["malthusian"]["n"], core["modern"]["n"]
    wkey = "spei_growing"
    mal = core["malthusian"]["fevd_log_cbr_h15"]
    mod = core["modern"]["fevd_log_cbr_h15"]

    print("\n" + "=" * 70)
    print("PHASE 11 HEADLINE: state-dependent system LP-FEVD of log fertility")
    print("=" * 70)
    print(f"Core system {core['variables']}  (h=15)")
    print(f"  N_malthusian = {nm}   N_modern = {nmod}")
    print(f"  WEATHER (SPEI) share of fertility variance:")
    print(f"    Malthusian = {mal[wkey]:.3f}   Modern = {mod[wkey]:.3f}   "
          f"(delta = {mod[wkey] - mal[wkey]:+.3f})")
    rose = "RISES" if mod[wkey] > mal[wkey] else "DOES NOT RISE"
    print(f"  => weather share {rose} from Malthusian to Modern.")
    print(f"  Full h=15 shares:")
    for lab, d in (("Malthusian", mal), ("Modern", mod)):
        print(f"    {lab:>10}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items()))

    coreT = result["core_by_regime_temperature"]
    print(f"\nCore-temperature variant {coreT['variables']} (h=15) WEATHER (T) share:")
    print(f"    Malthusian = {coreT['malthusian']['fevd_log_cbr_h15']['t_growing']:.3f}   "
          f"Modern = {coreT['modern']['fevd_log_cbr_h15']['t_growing']:.3f}")

    ext = result["extended_pooled"]
    print(f"\nExtended pooled system (N={ext['n']}) h=15 fertility FEVD shares:")
    for k, v in ext["fevd_log_cbr_h15"].items():
        print(f"    {k:>22}: {v:.3f}")

    extr = result["extended_by_regime"]
    print(f"\nExtended-by-regime (C):")
    for lab in ("malthusian", "modern"):
        entry = extr[lab]
        print(f"    {lab:>10}: status={entry.get('status')}  n={entry.get('n')}  "
              f"dof_floor={entry.get('dof_floor')}")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
