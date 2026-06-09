"""Extended colonial-origin partialling: state-history + SSA-ancestry controls.

The §6.5 colonial-partialling test found DARC's negative log-GDPpc coefficient
attenuates by only 10-15% under AJR-extractive-colonial and ~3% under the
broader European-colonised binary. The §3.1 interpretation accordingly framed
DARC as a marker of a *bundle* of correlated channels (tropical-disease
burden, slave-trade exposure, pre-colonial state formation, colonial extraction).

This script tightens the partialling along two more directions:
  - state_hist: Bockstette-Putterman pre-colonial state-history index
    (already in the horserace panel). Tests the pre-colonial-state-capacity
    channel directly.
  - ssa_ancestry: Putterman-Weil-weighted Sub-Saharan African ancestry
    fraction (= W_AFR + C_AFR + E_AFR_BANTU + S_AFR anchor weights).
    Tests "is DARC anything beyond a raw SSA-ancestry marker?"

If DARC survives both controls jointly, the negative loading runs through
neither pre-colonial state capacity nor raw African-ancestry collinearity,
narrowing the residual interpretation to tropical-disease ecology and/or
slave-trade-specific damage.
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.paper5_horserace.subsamples import AJR_COLONIAL
from analysis.paper5_horserace.exercise_colonial_partial import (
    EUR_COLONIZED, ALLELE_COLS, NON_FA_SUBSTRATES, CONTROLS,
)
from analysis.paper5_horserace.build_functional_alleles import PW_TO_ANCHOR

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
PW = DATA / "deep_determinants" / "_raw" / "putterman_weil_2010" / "pw_migration_matrix_v1p1.xlsx"

# Sub-Saharan African anchor populations (excluding N_AFR = mostly Afro-Asiatic
# Arabised, and E_AFR_AAA = Horn of Africa, which has substantial Eurasian
# ancestry component). The four below are the canonical SSA anchors.
SSA_ANCHORS = {"W_AFR", "C_AFR", "E_AFR_BANTU", "S_AFR"}


def build_ssa_ancestry() -> pd.DataFrame:
    """Per-country PW-weighted SSA ancestry fraction.

    Each country's modern ancestry is a vector over 1500 source populations;
    each source maps to one of 19 anchor populations via PW_TO_ANCHOR. The
    SSA fraction is the sum of PW weights for sources mapped to {W_AFR,
    C_AFR, E_AFR_BANTU, S_AFR}.
    """
    # File is named .xlsx but is actually old .xls (Composite Document Format)
    try:
        pw = pd.read_excel(PW, sheet_name="Original migration matrix",
                            engine="xlrd")
    except Exception:
        pw = pd.read_excel(PW, sheet_name=0, engine="xlrd")
    # First column is wbcode of modern country; remaining columns are source
    # countries with weights summing to ~1.0 per row.
    cols = pw.columns.tolist()
    # The leading non-numeric columns
    keep_meta = [c for c in cols if c in ("wbcode", "wbname", "update")]
    src_cols = [c for c in cols if c not in keep_meta]

    # Map each source column to its anchor (lowercase wbcode)
    src_anchor = {c: PW_TO_ANCHOR.get(c.lower(), None) for c in src_cols}
    ssa_src = [c for c, a in src_anchor.items() if a in SSA_ANCHORS]
    print(f"  SSA source columns ({len(ssa_src)}): {sorted(ssa_src)}")

    rows = []
    for _, r in pw.iterrows():
        iso3 = str(r["wbcode"]).upper().strip()
        ssa_frac = float(sum(r[c] for c in ssa_src if pd.notna(r[c])))
        rows.append({"iso3": iso3, "ssa_ancestry_pw": ssa_frac})
    out = pd.DataFrame(rows)
    print(f"  built SSA-ancestry fraction for {len(out)} countries")
    print(f"  range: [{out['ssa_ancestry_pw'].min():.3f}, "
          f"{out['ssa_ancestry_pw'].max():.3f}]")
    print(f"  N>0.5: {(out['ssa_ancestry_pw'] > 0.5).sum()}; "
          f"N>0.9: {(out['ssa_ancestry_pw'] > 0.9).sum()}")
    return out


def _fit(df: pd.DataFrame, outcome: str, regressors: list[str],
         label: str) -> dict:
    pathway_cols = sorted(c for c in df.columns if c.startswith("pathway_"))[1:]
    keep = [outcome] + regressors + CONTROLS + pathway_cols
    sub = df.dropna(subset=keep).copy()
    X = sm.add_constant(sub[regressors + CONTROLS + pathway_cols].astype(float))
    res = sm.OLS(sub[outcome], X).fit(cov_type="HC3")
    return {"label": label, "n": int(res.nobs), "r2": float(res.rsquared),
            "model": res}


def run() -> pd.DataFrame:
    panel = pd.read_parquet(DATA / "deep_determinants_horserace.parquet")
    panel["ajr_colonial"] = panel["iso3"].isin(AJR_COLONIAL).astype(int)
    panel["eur_colonized"] = panel["iso3"].isin(EUR_COLONIZED).astype(int)
    print("Building SSA ancestry from PW weights...")
    ssa = build_ssa_ancestry()
    panel = panel.merge(ssa, on="iso3", how="left")
    print(f"Panel after SSA merge: N={len(panel)} countries; "
          f"SSA-ancestry non-null: {panel['ssa_ancestry_pw'].notna().sum()}")

    outcome = "log_gdppc_2015"
    base_regs = NON_FA_SUBSTRATES + ALLELE_COLS

    specs = {
        "baseline":         (base_regs, "no extra control"),
        "+state_hist":      (base_regs + ["state_hist"], "+ Bockstette-Putterman state history"),
        "+ssa_ancestry":    (base_regs + ["ssa_ancestry_pw"], "+ PW-weighted SSA ancestry"),
        "+both":            (base_regs + ["state_hist", "ssa_ancestry_pw"], "+ state_hist + SSA ancestry"),
        "+all":             (base_regs + ["state_hist", "ssa_ancestry_pw",
                                            "eur_colonized"],
                              "+ state_hist + SSA + EUR-colonised"),
    }
    # Fix the estimation sample across all specs: restrict to rows non-missing
    # on the union of every regressor used in any spec, so that moving from
    # "baseline" to "+state_hist" reflects conditioning rather than the 18
    # countries that drop out when the (incomplete) state-history index enters.
    pathway_cols = sorted(c for c in panel.columns if c.startswith("pathway_"))[1:]
    union_regs = base_regs + ["state_hist", "ssa_ancestry_pw", "eur_colonized"]
    common_keep = [outcome] + union_regs + CONTROLS + pathway_cols
    n_before = len(panel)
    panel = panel.dropna(subset=common_keep).copy()
    print(f"\nFixed common estimation sample: N={len(panel)} "
          f"(dropped {n_before - len(panel)} rows missing any spec regressor)")

    fits = {name: _fit(panel, outcome, regs, lbl) for name, (regs, lbl) in specs.items()}

    print(f"\n=== Per-allele coefficients on {outcome} ===")
    print(f"{'allele':<12s}  " + "  ".join(f"{n:>20s}" for n in specs))
    for allele in ALLELE_COLS:
        cells = []
        for name in specs:
            m = fits[name]["model"]
            b = m.params.get(allele, np.nan)
            p = m.pvalues.get(allele, np.nan)
            cells.append(f"{b:>+9.3f} (p={p:.3f})")
        print(f"{allele:<12s}  " + "  ".join(cells))

    print(f"\n=== Coefficients on the added controls ===")
    for name, fit in fits.items():
        m = fit["model"]
        for c in ("state_hist", "ssa_ancestry_pw", "eur_colonized"):
            if c in m.params:
                print(f"  {name:<16s} {c:<20s} β={m.params[c]:+.4f} "
                      f"p={m.pvalues[c]:.4f}")

    # Attenuation of DARC across specs
    print(f"\n=== DARC attenuation summary (vs baseline) ===")
    base_b = fits["baseline"]["model"].params["fa_darc"]
    for name in specs:
        b = fits[name]["model"].params["fa_darc"]
        p = fits[name]["model"].pvalues["fa_darc"]
        pct = 100 * (1 - abs(b) / abs(base_b)) if abs(base_b) > 1e-12 else np.nan
        print(f"  {name:<16s}  β(DARC)={b:+.3f}  p={p:.4f}  attenuation={pct:+.1f}%")

    # Persist
    rows = []
    for name in specs:
        m = fits[name]["model"]
        for allele in ALLELE_COLS:
            rows.append({"spec": name, "allele": allele,
                          "beta": float(m.params.get(allele, np.nan)),
                          "se": float(m.bse.get(allele, np.nan)),
                          "p": float(m.pvalues.get(allele, np.nan)),
                          "n": int(m.nobs), "r2": float(m.rsquared)})
    df = pd.DataFrame(rows)
    out = DATA / "deep_determinants" / "exercise_colonial_partial_extended_results.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}")
    return df


if __name__ == "__main__":
    run()
