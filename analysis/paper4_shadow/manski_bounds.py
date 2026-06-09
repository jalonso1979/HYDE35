"""HYDE Manski bounds: re-run key regressions under the three HYDE 3.5
uncertainty scenarios (baseline, lower, upper) and report bounds on the key
coefficients.

The three scenarios reflect HYDE's published uncertainty range in population
and land-use reconstructions. The 'lower' and 'upper' scenarios deliver
plausible alternative reconstructions; their span gives partial-identification
bounds on regression coefficients.

Sub-national HYDE files (sub_hiscrop, sub_hispast, sub_hisirri, subpop) are
available in /Volumes/BIGDATA/HYDE35/gbc2025_7apr_{base,lower,upper}/.

Regressions reproduced under each scenario:
  - Sub-national Stage 1 at 1750: crop_share ~ productive_months + country FE
  - Sub-national log density at 1750
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

SCENARIO_DIRS = {
    "baseline": ROOT / "gbc2025_7apr_base",
    "lower":    ROOT / "gbc2025_7apr_lower",
    "upper":    ROOT / "gbc2025_7apr_upper",
}


def _load_hyde_subnational(scenario_dir: Path,
                            has_pop: bool = True) -> pd.DataFrame:
    """Sub-national wide table for one HYDE scenario."""
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))

    def _read(stem: str) -> pd.DataFrame:
        p = scenario_dir / f"{stem}_4apr2025.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df = df.dropna(subset=["isolink"]).copy()
        df["sub_id"] = df["isolink"].astype(int)
        df["iso_num"] = (df["sub_id"] // 1000).astype(int)
        df["iso3"] = df["iso_num"].map(num_to_iso3)
        df = df.dropna(subset=["iso3"]).copy()
        ycols = [c for c in df.columns if c.startswith("y")]
        # 1750 cropland / grazing / pop
        if "y1750" in df.columns:
            return df[["sub_id", "iso3", "y1750"]].rename(
                columns={"y1750": stem})
        return None

    pieces = {}
    pieces["crop"] = _read("sub_hiscrop")
    pieces["graz"] = _read("sub_hispast")
    if has_pop:
        pieces["pop"] = _read("subpop")
    out = pieces["crop"]
    for k in ["graz", "pop"]:
        if pieces.get(k) is not None:
            out = out.merge(pieces[k], on=["sub_id", "iso3"], how="outer")
    return out


def _fe_reg(d: pd.DataFrame, outcome: str, predictor: str) -> dict:
    d = d.dropna(subset=[outcome, predictor, "iso3"]).copy()
    counts = d["iso3"].value_counts()
    d = d[d["iso3"].isin(counts[counts >= 2].index)]
    if len(d) < 30: return {}
    g = d.groupby("iso3")
    d[outcome] = d[outcome] - g[outcome].transform("mean")
    d[predictor] = d[predictor] - g[predictor].transform("mean")
    X = sm.add_constant(d[[predictor]])
    y = d[outcome].astype(float)
    r = sm.OLS(y, X).fit(cov_type="cluster",
                          cov_kwds={"groups": d["iso3"].values})
    return {
        "beta": r.params[predictor], "se": r.bse[predictor],
        "p": r.pvalues[predictor], "n": int(r.nobs),
        "n_groups": d["iso3"].nunique(),
    }


def main() -> None:
    feats = pd.read_parquet(DATA / "subnational_features.parquet")
    print("Loaded sub-national features (productive_months, sigma_v, sigma_s)")
    print()

    rows = []
    for scen, sdir in SCENARIO_DIRS.items():
        has_pop = (sdir / "subpop_4apr2025.csv").exists()
        print(f"Scenario {scen} (has_pop={has_pop}):")
        h = _load_hyde_subnational(sdir, has_pop=has_pop)
        if h is None:
            print("  no data"); continue
        h["sub_hiscrop"] = h.get("sub_hiscrop", 0).fillna(0)
        h["sub_hispast"] = h.get("sub_hispast", 0).fillna(0)
        h["ag_total"] = h["sub_hiscrop"] + h["sub_hispast"]
        h["crop_share"] = np.where(h["ag_total"] > 0,
                                    h["sub_hiscrop"] / h["ag_total"], np.nan)
        df = feats.merge(h, on=["sub_id", "iso3"], how="inner")
        df = df[df["ag_total"] > 0]
        print(f"  Sample: {len(df):,} sub-units")

        # Stage 1: crop_share ~ productive_months + country FE
        r1 = _fe_reg(df, "crop_share", "productive_months")
        if r1:
            print(f"    crop_share ~ productive_months: β={r1['beta']:+.4f} "
                  f"(SE {r1['se']:.4f}) p={r1['p']:.4f} N={r1['n']}")
            rows.append({"scenario": scen, "outcome": "crop_share",
                          "predictor": "productive_months", **r1})

        # log density
        if has_pop and "subpop" in df.columns:
            df["log_density"] = np.log((df["subpop"] / df["ag_total"].clip(lower=1)).clip(lower=1e-6))
            r2 = _fe_reg(df, "log_density", "productive_months")
            if r2:
                print(f"    log_density ~ productive_months: β={r2['beta']:+.4f} "
                      f"(SE {r2['se']:.4f}) p={r2['p']:.4f} N={r2['n']}")
                rows.append({"scenario": scen, "outcome": "log_density",
                              "predictor": "productive_months", **r2})

            r3 = _fe_reg(df, "log_density", "sigma_s")
            if r3:
                print(f"    log_density ~ sigma_s: β={r3['beta']:+.4f} "
                      f"(SE {r3['se']:.4f}) p={r3['p']:.4f} N={r3['n']}")
                rows.append({"scenario": scen, "outcome": "log_density",
                              "predictor": "sigma_s", **r3})
        print()

    res = pd.DataFrame(rows)
    res.to_parquet(DATA / "manski_bounds.parquet", index=False)

    print("\n=== Manski bounds (min-max across scenarios) ===")
    for (outcome, predictor), sub in res.groupby(["outcome", "predictor"]):
        lo, hi = sub["beta"].min(), sub["beta"].max()
        print(f"  {outcome} ~ {predictor}: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
