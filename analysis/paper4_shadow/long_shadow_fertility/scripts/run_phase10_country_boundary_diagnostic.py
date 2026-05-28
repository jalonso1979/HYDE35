"""
phase10/diag3: Country-boundary diagnostic for within-season SD jumps.

Tests whether large year-over-year jumps in t_anom_c_within_season_sd
cluster near known historical territorial changes using a Monte Carlo
permutation test.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PANEL = "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/panel_multi_country_year.parquet"
OUT = Path("/Volumes/BIGDATA/HYDE35/analysis/output/long_shadow_fertility/phase10_country_boundary_diagnostic.json")

KNOWN_CHANGES = {
    "ITA": [1861, 1870, 1919],
    "BEL": [1830],
    "NLD": [1830, 1648],
    "GBR": [1707, 1801, 1922],
    "SWE": [1809, 1815, 1905],
    "ESP": [1700, 1714, 1898],
    "FRA": [1789, 1815, 1860, 1871, 1919],
}

WINDOW = 5  # ± years around a known change


def find_jumps(sub, col="t_anom_c_within_season_sd", thresh_sd=2.0):
    sub = sub.dropna(subset=[col, "year"]).sort_values("year")
    diffs = sub[col].diff().abs()
    sd = diffs.std(ddof=1)
    if sd is None or np.isnan(sd) or sd == 0:
        return []
    mask = diffs > thresh_sd * sd
    return sub.loc[mask, "year"].astype(int).tolist()


def near_known_change(year, known_years, window=WINDOW):
    return any(abs(year - k) <= window for k in known_years)


df = pd.read_parquet(PANEL)
print(f"Loaded panel: {df.shape[0]} rows, columns: {list(df.columns)}")
print(f"iso3 values: {sorted(df['iso3'].unique())}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# Check column exists
if "t_anom_c_within_season_sd" not in df.columns:
    raise ValueError(
        f"Column 't_anom_c_within_season_sd' not found. Available: {list(df.columns)}"
    )

by_country = {}
all_jumps = []  # tuples (iso3, year)

for iso3, sub in df.groupby("iso3"):
    jumps = find_jumps(sub)
    known = KNOWN_CHANGES.get(iso3, [])
    near = [y for y in jumps if near_known_change(y, known)]
    not_near = [y for y in jumps if y not in near]
    by_country[iso3] = {
        "large_jump_years": jumps,
        "jumps_near_known_changes": near,
        "jumps_not_explained": not_near,
        "known_changes_used": known,
    }
    all_jumps.extend((iso3, y) for y in jumps)
    print(
        f"  {iso3}: {len(jumps)} jumps, {len(near)} near territorial changes, "
        f"not explained: {not_near}"
    )

# Pooled permutation test
n_total = len(all_jumps)
n_near = sum(
    1
    for iso3, y in all_jumps
    if near_known_change(y, KNOWN_CHANGES.get(iso3, []))
)
actual = n_near / n_total if n_total else 0.0
print(f"\nTotal jumps: {n_total}, near known changes: {n_near}, fraction: {actual:.3f}")

rng = np.random.default_rng(0)
nulls = []
years_by_country = {
    iso3: sub.dropna(subset=["t_anom_c_within_season_sd"])["year"].astype(int).tolist()
    for iso3, sub in df.groupby("iso3")
}

for _ in range(1000):
    fake_jumps = []
    for iso3, ys in years_by_country.items():
        n = len(by_country.get(iso3, {}).get("large_jump_years", []))
        if n == 0 or not ys:
            continue
        sampled = rng.choice(ys, size=min(n, len(ys)), replace=False)
        fake_jumps.extend((iso3, int(y)) for y in sampled)
    n_t = len(fake_jumps)
    n_n = sum(
        1
        for iso3, y in fake_jumps
        if near_known_change(y, KNOWN_CHANGES.get(iso3, []))
    )
    nulls.append(n_n / n_t if n_t else 0.0)

null_arr = np.array(nulls)
p = float(np.mean(null_arr >= actual))

print(f"Null mean: {null_arr.mean():.3f}, p-value: {p:.3f}")

out = {
    "by_country": by_country,
    "permutation_test": {
        "actual_fraction_near_changes": float(actual),
        "null_mean": float(null_arr.mean()),
        "null_p_value": p,
        "n_permutations": 1000,
        "n_jumps_total": int(n_total),
        "n_jumps_near_changes": int(n_near),
    },
}

# Interpretation
if p < 0.05:
    out["interpretation"] = (
        f"Large jumps in within-season SD cluster significantly more near known "
        f"territorial changes than chance ({actual:.0%} vs null mean {null_arr.mean():.0%}, "
        f"p={p:.3f}). Country-boundary changes are a partial explanation."
    )
elif p < 0.20:
    out["interpretation"] = (
        f"Large jumps show suggestive but not significant clustering near territorial "
        f"changes ({actual:.0%} vs {null_arr.mean():.0%}, p={p:.3f}). Effect direction "
        f"consistent with the country-boundary hypothesis but not conclusive."
    )
else:
    out["interpretation"] = (
        f"Large jumps do not cluster near territorial changes ({actual:.0%} vs "
        f"{null_arr.mean():.0%}, p={p:.3f}). Country-boundary changes are not the "
        f"main driver of the jumps; volcanic episodes and small-sample noise are "
        f"more likely candidates."
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2))
print(f"\nOutput written to: {OUT}")
print(json.dumps(out, indent=2))
