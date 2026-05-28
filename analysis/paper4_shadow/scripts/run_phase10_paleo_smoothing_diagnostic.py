"""
Phase 10 Diagnostic 1: Paleo-smoothing artifact test for within-season SD trend.

Tests whether the upward trend in t_anom_c_within_season_sd (Fig 22 top panel,
0.53->0.94 deg C across 5 centuries) is a paleo-smoothing artifact or a real
climate-change signal, by comparing era-specific OLS slopes with country FE.

Eras:
  pre_1700   : proxy-only (tree-ring, ice-core, documentary)
  1700_1850  : sparse instrumental (weather diaries, early met networks)
  1850_2008  : dense instrumental (mature European met networks)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PANEL = "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_paleo_smoothing_diagnostic.json")


def trend_in_era(df_era, label):
    if df_era.empty:
        return {"label": label, "slope_per_century": None, "se": None, "n": 0, "countries_present": []}
    df_era = df_era.copy()
    # slope expressed per century
    df_era["t_pct"] = df_era["year"] / 100.0
    dums = pd.get_dummies(df_era["iso3"], drop_first=True, dtype=float)
    X = np.column_stack([
        np.ones(len(df_era)),
        df_era["t_pct"].to_numpy(dtype=float),
        dums.to_numpy(dtype=float)
    ])
    Y = df_era["t_anom_c_within_season_sd"].to_numpy(dtype=float)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ b
    dof = len(Y) - X.shape[1]
    s2 = float((resid @ resid) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(s2 * XtX_inv[1, 1]))
    countries = sorted(df_era["iso3"].unique().tolist())
    return {
        "label": label,
        "slope_per_century": float(b[1]),
        "se": se,
        "n": int(len(df_era)),
        "countries_present": countries,
    }


def main():
    df = pd.read_parquet(PANEL)
    df = df.dropna(subset=["t_anom_c_within_season_sd", "year", "iso3"]).copy()
    df = df[df["year"] <= 2008].copy()  # ModE-RA ends 2008

    print(f"Total obs after dropna + <=2008 filter: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries: {sorted(df['iso3'].unique().tolist())}")

    eras = {
        "pre_1700":  df[df["year"] < 1700],
        "1700_1850": df[(df["year"] >= 1700) & (df["year"] < 1850)],
        "1850_2008": df[(df["year"] >= 1850) & (df["year"] <= 2008)],
    }

    for k, v in eras.items():
        print(f"  {k}: {len(v)} obs, years {v['year'].min() if len(v) else 'NA'}-{v['year'].max() if len(v) else 'NA'}")

    out = {k: trend_in_era(v, k) for k, v in eras.items()}

    # Wald test: post-1850 slope == pre-1700 slope?
    b_pre  = out["pre_1700"]["slope_per_century"]
    b_post = out["1850_2008"]["slope_per_century"]
    se_pre  = out["pre_1700"]["se"]
    se_post = out["1850_2008"]["se"]

    if b_pre is not None and b_post is not None and se_pre and se_post:
        diff = b_post - b_pre
        se_diff = float(np.sqrt(se_pre**2 + se_post**2))
        z = diff / se_diff
        chi2 = float(z**2)
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        out["wald_post1850_vs_pre1700"] = {
            "diff_slope_per_century": float(diff),
            "z": float(z),
            "chi2": chi2,
            "p": p,
        }
    else:
        out["wald_post1850_vs_pre1700"] = {"chi2": None, "p": None}

    # Interpretation
    b_pre_v  = out["pre_1700"]["slope_per_century"]
    b_post_v = out["1850_2008"]["slope_per_century"]
    p_val    = out["wald_post1850_vs_pre1700"].get("p")

    if b_post_v is not None and b_pre_v is not None:
        if abs(b_post_v) > 2 * abs(b_pre_v):
            interp = (
                f"Post-1850 trend ({b_post_v:.4f} deg C/century) is more than twice the pre-1700 trend "
                f"({b_pre_v:.4f} deg C/century) — modern climate change is contributing materially and "
                f"paleo smoothing alone cannot explain the within-season SD rise "
                f"(Wald p={p_val:.4f})."
            )
        elif b_pre_v is not None and abs(b_pre_v) > abs(b_post_v):
            interp = (
                f"Pre-1700 trend ({b_pre_v:.4f} deg C/century) exceeds post-1850 "
                f"({b_post_v:.4f} deg C/century) — consistent with paleo-smoothing recovery toward "
                f"true variance as instrumental observations densify (Wald p={p_val:.4f})."
            )
        else:
            interp = (
                f"Pre-1700 ({b_pre_v:.4f} deg C/century) and post-1850 ({b_post_v:.4f} deg C/century) "
                f"slopes are comparable — neither paleo smoothing nor modern climate change alone "
                f"dominates the upward trend (Wald p={p_val:.4f})."
            )
    else:
        interp = "Insufficient data to determine interpretation."

    out["interpretation"] = interp

    # Clean up label keys to match schema (remove nested "label" field)
    result = {}
    for k in ["pre_1700", "1700_1850", "1850_2008"]:
        d = out[k].copy()
        d.pop("label", None)
        result[k] = d
    result["wald_post1850_vs_pre1700"] = out["wald_post1850_vs_pre1700"]
    result["interpretation"] = out["interpretation"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print("\n=== OUTPUT ===")
    print(json.dumps(result, indent=2))
    print(f"\nWritten to: {OUT}")


if __name__ == "__main__":
    main()
