# Growing-Season Climate Measures for long_shadow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the annual-mean temperature volatility regressor $\sigma_v^T$ across all `long_shadow.tex` regressions with climate-defined growing-season-restricted analogues $\{\sigma_v^{T,\mathrm{GS}}, \sigma_v^{P,\mathrm{GS}}\}$ (plus a non-GS placebo $\sigma_v^{T,\overline{\mathrm{GS}}}$ as side-by-side headline column), and replace interval-mean / within-interval climate vectors in the Malthusian and joint VAR regressions with their GS-mean analogues. Country (cropland-weighted headline) and sub-national (pop-weighted headline) panels are built in the same pass.

**Architecture:** One new builder `analysis/paper4_shadow/build_gs_climate.py` produces four parquet outputs (country cross-section, country annual panel, sub-national cross-section, sub-national annual panel) with `_area`/`_pop`/`_cropw` suffixed columns for each weighting variant. All existing regression scripts that consume $\sigma_v^T$ or interval-mean climate are edited to read the GS columns from the new files (the old `country_seasonality_preindustrial.parquet` measures remain available as robustness rows). The paper text is updated section by section. A smoke-test script (`analysis/paper4_shadow/test_gs_climate_smoke.py`) asserts edge-case properties of the new builder.

**Tech Stack:** Python 3.13, pandas, numpy, statsmodels (OLS, HC1/HC3 SEs), pyarrow (parquet I/O), matplotlib. LaTeX for paper. Input data already in `analysis/data/`.

**Spec reference:** `docs/superpowers/specs/2026-05-19-growing-season-volatility-design.md`

---

## Phase A — Build the GS climate data files

### Task 1: Country-level GS mask + cross-section + annual panel (3 weightings)

**Files:**
- Create: `analysis/paper4_shadow/build_gs_climate.py`
- Test: `analysis/paper4_shadow/test_gs_climate_smoke.py`
- Inputs (read-only): `analysis/data/modera_country_monthly.parquet`, `modera_country_monthly_popw.parquet`, `modera_country_monthly_cropw.parquet`, `cru_country_climatology_1901_1950.parquet`
- Outputs: `analysis/data/country_seasonality_gs_preindustrial.parquet`, `analysis/data/country_climate_gs_1421_2025.parquet`

- [ ] **Step 1: Write the smoke test asserting expected edge cases**

Create `analysis/paper4_shadow/test_gs_climate_smoke.py`:

```python
"""Smoke tests for build_gs_climate.py country-level outputs.

These are the pre-registered edge-case properties the builder must satisfy.
Run after the builder produces output parquets. Exit 0 on success.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

DATA = Path("/Volumes/BIGDATA/HYDE35/analysis/data")
WEIGHTINGS = ["area", "pop", "cropw"]


def assert_cross_section() -> None:
    df = pd.read_parquet(DATA / "country_seasonality_gs_preindustrial.parquet")
    expected_cols = ["iso3", "n_gs_months_cropw", "gs_months_mask_cropw"]
    for w in WEIGHTINGS:
        expected_cols += [f"sigma_v_T_gs_pre1750_{w}",
                          f"sigma_v_P_gs_pre1750_{w}",
                          f"T_gs_mean_pre1750_{w}",
                          f"P_gs_mean_pre1750_{w}",
                          f"sigma_v_T_nongs_pre1750_{w}",
                          f"n_gs_months_{w}"]
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"missing cross-section cols: {missing}"
    assert df["iso3"].is_unique, "iso3 must be unique in cross-section"
    assert len(df) >= 180, f"too few countries: {len(df)}"
    # Egypt is heavily desert; should have low or zero GS months under
    # cropland weighting (Nile-dominated cells are warm but the country
    # mean is bounded by the threshold tests).
    egy = df[df["iso3"] == "EGY"].iloc[0]
    assert egy["n_gs_months_cropw"] <= 6, (
        f"Egypt cropw n_gs_months unexpectedly high: {egy['n_gs_months_cropw']}")
    # Indonesia is wet equatorial; expect 12 GS months under any weighting.
    idn = df[df["iso3"] == "IDN"].iloc[0]
    assert idn["n_gs_months_cropw"] == 12, (
        f"Indonesia cropw n_gs_months not 12: {idn['n_gs_months_cropw']}")
    # France is temperate; expect 5-9 GS months.
    fra = df[df["iso3"] == "FRA"].iloc[0]
    assert 5 <= fra["n_gs_months_cropw"] <= 9, (
        f"France cropw n_gs_months out of range: {fra['n_gs_months_cropw']}")
    # All non-empty-GS countries must have positive sigma values.
    valid = df.dropna(subset=["sigma_v_T_gs_pre1750_cropw"])
    assert (valid["sigma_v_T_gs_pre1750_cropw"] > 0).all(), (
        "sigma_v_T_gs must be positive when defined")
    print(f"  cross-section: {len(df)} countries, "
          f"{df['n_gs_months_cropw'].notna().sum()} with cropw n_gs defined")


def assert_annual_panel() -> None:
    df = pd.read_parquet(DATA / "country_climate_gs_1421_2025.parquet")
    expected_cols = ["iso3", "year"]
    for w in WEIGHTINGS:
        expected_cols += [f"t_gs_mean_{w}", f"p_gs_mean_{w}",
                          f"t_gs_anom_{w}", f"p_gs_anom_{w}"]
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"missing panel cols: {missing}"
    assert df.groupby(["iso3", "year"]).size().max() == 1, "duplicate iso3-year"
    yrs = df["year"].unique()
    assert yrs.min() == 1421 and yrs.max() >= 2008, (
        f"unexpected year range: {yrs.min()}-{yrs.max()}")
    # Anomalies should average to ~0 over the 1421-1750 reference window per country.
    pre = df[df["year"].between(1421, 1750)]
    anom_mean = pre.groupby("iso3")["t_gs_anom_cropw"].mean().abs()
    assert (anom_mean.dropna() < 1e-6).all(), (
        f"t_gs_anom_cropw not centered on 1421-1750: max |mean|={anom_mean.max()}")
    print(f"  annual panel: {len(df)} rows, {df['iso3'].nunique()} countries, "
          f"years {yrs.min()}-{yrs.max()}")


def main() -> None:
    print("Running build_gs_climate smoke tests...")
    assert_cross_section()
    assert_annual_panel()
    print("All smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 2: Run the smoke test to confirm it fails (outputs don't exist yet)**

```bash
python /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/test_gs_climate_smoke.py
```
Expected: `SMOKE TEST FAILED` (FileNotFoundError on `country_seasonality_gs_preindustrial.parquet`).

- [ ] **Step 3: Write the country-level builder (`build_gs_climate.py`)**

Create `analysis/paper4_shadow/build_gs_climate.py`:

```python
"""Climate-defined growing-season volatility / anomaly measures for long_shadow.

Builds four parquet files:
  - country_seasonality_gs_preindustrial.parquet  (1 row / iso3, 3 weightings)
  - country_climate_gs_1421_2025.parquet          (iso3 × year, 3 weightings)
  - subnational_seasonality_gs_preindustrial.parquet (1 row / sub_id, 2 weightings)
  - subnational_climate_gs_1421_2025.parquet      (sub_id × year, 2 weightings)

GS mask is fixed (1421-1750 climatology): m ∈ GS_i iff 5≤T_clim≤30 AND P_clim≥30mm.

Spec: docs/superpowers/specs/2026-05-19-growing-season-volatility-design.md
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"

PRE_WINDOW = (1421, 1750)
T_MIN, T_MAX = 5.0, 30.0
P_MIN = 30.0  # mm / month

COUNTRY_WEIGHTINGS = {
    "area":  "modera_country_monthly.parquet",
    "pop":   "modera_country_monthly_popw.parquet",
    "cropw": "modera_country_monthly_cropw.parquet",
}
SUBNAT_WEIGHTINGS = {
    "area": "modera_subnational_monthly.parquet",
    "pop":  "modera_subnational_monthly_popw.parquet",
}


def _absolute_levels(mod: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Add CRU 1901-1950 climatology to anomalies to recover absolute T, P."""
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    if entity_col != "iso3":
        # Sub-national rows already carry iso3 via merge in builder.
        pass
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    return df


def _gs_mask(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Return one row per (entity, month) with in_gs flag from 1421-1750 climatology."""
    pre = df[df["year"].between(*PRE_WINDOW)]
    clim = pre.groupby([entity_col, "month"], as_index=False).agg(
        t_clim=("t_abs", "mean"), p_clim=("p_abs", "mean"))
    clim["in_gs"] = ((clim["t_clim"] >= T_MIN) & (clim["t_clim"] <= T_MAX) &
                     (clim["p_clim"] >= P_MIN))
    return clim[[entity_col, "month", "in_gs"]]


def _cross_section(df: pd.DataFrame, mask: pd.DataFrame, entity_col: str,
                   weighting: str) -> pd.DataFrame:
    """Per-entity GS-restricted volatility + climatology, 1421-1750."""
    pre = df[df["year"].between(*PRE_WINDOW)].merge(mask, on=[entity_col, "month"])

    gs = pre[pre["in_gs"]]
    gs_yr = gs.groupby([entity_col, "year"], as_index=False).agg(
        t_gs=("t_abs", "mean"), p_gs=("p_abs", "mean"))
    cs_gs = gs_yr.groupby(entity_col, as_index=False).agg(
        **{f"T_gs_mean_pre1750_{weighting}":     ("t_gs", "mean"),
           f"P_gs_mean_pre1750_{weighting}":     ("p_gs", "mean"),
           f"sigma_v_T_gs_pre1750_{weighting}":  ("t_gs", "std"),
           f"sigma_v_P_gs_pre1750_{weighting}":  ("p_gs", "std")})

    nongs = pre[~pre["in_gs"]]
    nongs_yr = nongs.groupby([entity_col, "year"], as_index=False).agg(
        t_nongs=("t_abs", "mean"))
    cs_nongs = nongs_yr.groupby(entity_col, as_index=False).agg(
        **{f"sigma_v_T_nongs_pre1750_{weighting}": ("t_nongs", "std")})

    counts = (mask.groupby(entity_col, as_index=False)
                  .agg(**{f"n_gs_months_{weighting}": ("in_gs", "sum")}))
    months_list = (mask[mask["in_gs"]]
                    .groupby(entity_col)["month"]
                    .apply(lambda s: ",".join(str(m) for m in sorted(s)))
                    .rename(f"gs_months_mask_{weighting}")
                    .reset_index())

    out = counts.merge(months_list, on=entity_col, how="left").merge(
        cs_gs, on=entity_col, how="left").merge(
        cs_nongs, on=entity_col, how="left")
    out[f"gs_months_mask_{weighting}"] = out[f"gs_months_mask_{weighting}"].fillna("")
    return out


def _annual_panel(df: pd.DataFrame, mask: pd.DataFrame, entity_col: str,
                  weighting: str) -> pd.DataFrame:
    """Per (entity, year): GS-mean T, P, and anomalies vs 1421-1750 GS climatology."""
    merged = df.merge(mask, on=[entity_col, "month"])
    gs = merged[merged["in_gs"]]
    yr = gs.groupby([entity_col, "year"], as_index=False).agg(
        **{f"t_gs_mean_{weighting}": ("t_abs", "mean"),
           f"p_gs_mean_{weighting}": ("p_abs", "mean")})
    pre = yr[yr["year"].between(*PRE_WINDOW)]
    clim = pre.groupby(entity_col, as_index=False).agg(
        **{f"t_gs_clim_{weighting}": (f"t_gs_mean_{weighting}", "mean"),
           f"p_gs_clim_{weighting}": (f"p_gs_mean_{weighting}", "mean")})
    yr = yr.merge(clim, on=entity_col, how="left")
    yr[f"t_gs_anom_{weighting}"] = (yr[f"t_gs_mean_{weighting}"]
                                      - yr[f"t_gs_clim_{weighting}"])
    yr[f"p_gs_anom_{weighting}"] = (yr[f"p_gs_mean_{weighting}"]
                                      - yr[f"p_gs_clim_{weighting}"])
    return yr.drop(columns=[f"t_gs_clim_{weighting}", f"p_gs_clim_{weighting}"])


def _build_country() -> None:
    cs_pieces: list[pd.DataFrame] = []
    panel_pieces: list[pd.DataFrame] = []
    for w, fname in COUNTRY_WEIGHTINGS.items():
        print(f"[country/{w}] reading {fname}", flush=True)
        mod = pd.read_parquet(DATA / fname)
        abs_df = _absolute_levels(mod, entity_col="iso3")
        mask = _gs_mask(abs_df, entity_col="iso3")
        cs_pieces.append(_cross_section(abs_df, mask, "iso3", w))
        panel_pieces.append(_annual_panel(abs_df, mask, "iso3", w))

    cs = cs_pieces[0]
    for piece in cs_pieces[1:]:
        cs = cs.merge(piece, on="iso3", how="outer")
    cs["n_gs_months_cropw"] = cs["n_gs_months_cropw"].fillna(0).astype(int)
    cs["gs_months_mask_cropw"] = cs["gs_months_mask_cropw"].fillna("")
    cs_out = DATA / "country_seasonality_gs_preindustrial.parquet"
    cs.to_parquet(cs_out, index=False)
    print(f"[country] wrote {cs_out} ({len(cs)} countries)")

    panel = panel_pieces[0]
    for piece in panel_pieces[1:]:
        panel = panel.merge(piece, on=["iso3", "year"], how="outer")
    panel_out = DATA / "country_climate_gs_1421_2025.parquet"
    panel.to_parquet(panel_out, index=False)
    print(f"[country] wrote {panel_out} ({len(panel):,} rows, "
          f"{panel['iso3'].nunique()} countries)")

    print("\n[country/diagnostic] empty-GS (cropw):")
    empty = cs[cs["n_gs_months_cropw"] == 0]["iso3"].tolist()
    print(f"  N={len(empty)}: {empty}")
    print("[country/diagnostic] short-GS (cropw, 1-3 months):")
    short = cs[cs["n_gs_months_cropw"].between(1, 3)]["iso3"].tolist()
    print(f"  N={len(short)}: {short}")


def main() -> None:
    _build_country()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the builder for country only**

```bash
python /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/build_gs_climate.py
```
Expected: prints reading messages for 3 weightings, writes two parquet files, prints empty-GS / short-GS diagnostic lists.

- [ ] **Step 5: Run the smoke test**

```bash
python /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/test_gs_climate_smoke.py
```
Expected: `All smoke tests passed.` If France or Indonesia fail the n_gs_months range, inspect the cropw mask — likely a sign that one of the threshold values needs adjustment, but with 5/30/30 thresholds these should hold.

- [ ] **Step 6: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add analysis/paper4_shadow/build_gs_climate.py \
        analysis/paper4_shadow/test_gs_climate_smoke.py \
        analysis/data/country_seasonality_gs_preindustrial.parquet \
        analysis/data/country_climate_gs_1421_2025.parquet
git commit -m "$(cat <<'EOF'
long_shadow: build country-level GS climate (mask, σ_v^T_GS, σ_v^P_GS, panel anomalies)

Climate-defined growing-season mask from 1421-1750 climatology (T∈[5,30]
AND P≥30mm). Three weightings (area / pop / cropw) in suffixed columns.
Cross-section + annual panel. Smoke test asserts edge cases (Egypt low GS,
Indonesia 12 months, France temperate range, anomalies centered on 1421-1750).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Sub-national GS mask + cross-section + annual panel (2 weightings)

**Files:**
- Modify: `analysis/paper4_shadow/build_gs_climate.py` (add sub-national builder)
- Modify: `analysis/paper4_shadow/test_gs_climate_smoke.py` (add sub-national assertions)
- Inputs: `analysis/data/modera_subnational_monthly.parquet`, `modera_subnational_monthly_popw.parquet`
- Outputs: `analysis/data/subnational_seasonality_gs_preindustrial.parquet`, `analysis/data/subnational_climate_gs_1421_2025.parquet`

- [ ] **Step 1: Extend the smoke test with sub-national assertions**

Append to `test_gs_climate_smoke.py`:

```python
def assert_subnational() -> None:
    cs = pd.read_parquet(DATA / "subnational_seasonality_gs_preindustrial.parquet")
    pn = pd.read_parquet(DATA / "subnational_climate_gs_1421_2025.parquet")
    for w in ("area", "pop"):
        for col in (f"sigma_v_T_gs_pre1750_{w}", f"sigma_v_P_gs_pre1750_{w}",
                    f"T_gs_mean_pre1750_{w}", f"P_gs_mean_pre1750_{w}",
                    f"sigma_v_T_nongs_pre1750_{w}", f"n_gs_months_{w}"):
            assert col in cs.columns, f"sub-national cross-section missing {col}"
        for col in (f"t_gs_mean_{w}", f"p_gs_mean_{w}",
                    f"t_gs_anom_{w}", f"p_gs_anom_{w}"):
            assert col in pn.columns, f"sub-national panel missing {col}"
    assert cs["sub_id"].is_unique, "sub_id must be unique in cross-section"
    assert cs["sub_id"].nunique() >= 3000, f"too few sub-units: {cs['sub_id'].nunique()}"
    print(f"  sub-national cross-section: {len(cs)} units")
    print(f"  sub-national panel: {len(pn):,} rows, {pn['sub_id'].nunique()} units")
```

And in `main()` add `assert_subnational()` after `assert_annual_panel()`.

- [ ] **Step 2: Add sub-national builder to `build_gs_climate.py`**

Add to the same file (before `main()`):

```python
def _absolute_levels_subnat(mod: pd.DataFrame) -> pd.DataFrame:
    """Sub-national needs absolute levels via the parent country's CRU climatology."""
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")
    df = mod.merge(clim, on=["iso3", "month"], how="inner")
    df["t_abs"] = df["t_anom_c"] + df["tmp_c_clim"]
    df["p_abs"] = (df["p_anom_mm"] + df["pre_mm_clim"]).clip(lower=0)
    return df


def _build_subnational() -> None:
    cs_pieces: list[pd.DataFrame] = []
    panel_pieces: list[pd.DataFrame] = []
    for w, fname in SUBNAT_WEIGHTINGS.items():
        print(f"[subnat/{w}] reading {fname}", flush=True)
        mod = pd.read_parquet(DATA / fname)
        abs_df = _absolute_levels_subnat(mod)
        mask = _gs_mask(abs_df, entity_col="sub_id")
        cs_pieces.append(_cross_section(abs_df, mask, "sub_id", w))
        panel_pieces.append(_annual_panel(abs_df, mask, "sub_id", w))

    cs = cs_pieces[0]
    for piece in cs_pieces[1:]:
        cs = cs.merge(piece, on="sub_id", how="outer")
    # Carry iso3 forward for downstream merges
    iso_map = (pd.read_parquet(DATA / SUBNAT_WEIGHTINGS["pop"],
                                columns=["sub_id", "iso3"])
                 .drop_duplicates("sub_id"))
    cs = cs.merge(iso_map, on="sub_id", how="left")
    cs_out = DATA / "subnational_seasonality_gs_preindustrial.parquet"
    cs.to_parquet(cs_out, index=False)
    print(f"[subnat] wrote {cs_out} ({len(cs)} units)")

    panel = panel_pieces[0]
    for piece in panel_pieces[1:]:
        panel = panel.merge(piece, on=["sub_id", "year"], how="outer")
    panel = panel.merge(iso_map, on="sub_id", how="left")
    panel_out = DATA / "subnational_climate_gs_1421_2025.parquet"
    panel.to_parquet(panel_out, index=False)
    print(f"[subnat] wrote {panel_out} ({len(panel):,} rows, "
          f"{panel['sub_id'].nunique()} units)")
```

Update `main()`:

```python
def main() -> None:
    _build_country()
    _build_subnational()
```

- [ ] **Step 3: Run the builder**

```bash
python /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/build_gs_climate.py
```
Expected: country output reprinted; then sub-national reading messages for 2 weightings; writes two more parquet files. Sub-national runtime ~3-5 minutes due to 22M-row source panels.

- [ ] **Step 4: Run the smoke test**

```bash
python /Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/test_gs_climate_smoke.py
```
Expected: `All smoke tests passed.`

- [ ] **Step 5: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add analysis/paper4_shadow/build_gs_climate.py \
        analysis/paper4_shadow/test_gs_climate_smoke.py \
        analysis/data/subnational_seasonality_gs_preindustrial.parquet \
        analysis/data/subnational_climate_gs_1421_2025.parquet
git commit -m "$(cat <<'EOF'
long_shadow: extend GS climate builder to sub-national (area + pop weightings)

Mirrors country-level outputs at sub-unit resolution. iso3 carried forward
on both files for downstream regression merges. Smoke test extended with
sub-national column-presence and row-count assertions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase B — Cross-section headline regression

### Task 3: Add GS measures + non-GS placebo to `long_shadow.py` headline

**Files:**
- Modify: `analysis/paper4_shadow/long_shadow.py`
- Input: `analysis/data/country_seasonality_gs_preindustrial.parquet`, existing modern panel and pathway files
- Output: results written to `analysis/data/long_shadow_results_gs.parquet` (new) plus existing tables

- [ ] **Step 1: Write the prediction assertion**

Add a function to the end of `long_shadow.py` (above `if __name__`):

```python
def assert_predictions(results: dict) -> None:
    """Pre-registered predictions from spec §7.

    Logged but not raised — a failed prediction is informative, not a bug.
    """
    print("\n=== Pre-registered prediction checks ===")
    annual_beta = results["annual"]["beta"]
    gs_beta = results["gs"]["beta"]
    p_beta = results["gs_p"]["beta"]
    p_pval = results["gs_p"]["p"]
    placebo_beta = results["nongs"]["beta"]
    placebo_pval = results["nongs"]["p"]

    print(f"P1: |β(σ_v^T_GS)|={abs(gs_beta):.3f} vs |β(σ_v^T_annual)|"
          f"={abs(annual_beta):.3f}  "
          f"{'PASS' if abs(gs_beta) > abs(annual_beta) else 'FAIL (channel may be climate-deep, not agronomic)'}")
    print(f"P2: β(σ_v^P_GS)={p_beta:+.3f}, p={p_pval:.4f}  "
          f"{'PASS' if (p_beta < 0 and p_pval < 0.05) else 'FAIL (P-volatility null)'}")
    print(f"P3: β(σ_v^T_nonGS placebo)={placebo_beta:+.3f}, p={placebo_pval:.4f}  "
          f"{'PASS' if placebo_pval > 0.10 else 'FAIL (non-GS placebo significant — channel may not be agronomic)'}")
```

- [ ] **Step 2: Add the GS headline regression**

Add to `long_shadow.py` (before `assert_predictions`):

```python
def run_headline_gs(modern_panel: pd.DataFrame,
                    pathways: pd.DataFrame,
                    abs_lat: pd.DataFrame,
                    annual_seasonality: pd.DataFrame) -> dict:
    """Run the §5 headline cross-section under GS measures + non-GS placebo.

    Modern outcome: log pop growth 1950-1960 → 2015-2025. Returns dict of
    regression-result summaries used by assert_predictions and the LaTeX
    table emitter.
    """
    import statsmodels.api as sm

    gs = pd.read_parquet(DATA_DIR / "country_seasonality_gs_preindustrial.parquet")
    # Build outcome
    p0 = (modern_panel[modern_panel["year"].between(1950, 1960)]
            .groupby("iso3", as_index=False).agg(p0=("pop", "mean")))
    p1 = (modern_panel[modern_panel["year"].between(2015, 2025)]
            .groupby("iso3", as_index=False).agg(p1=("pop", "mean")))
    out = p0.merge(p1, on="iso3").query("p0 > 0 and p1 > 0").copy()
    out["log_pop_growth"] = np.log(out["p1"] / out["p0"])

    df = (out.merge(gs[["iso3", "sigma_v_T_gs_pre1750_cropw",
                         "sigma_v_P_gs_pre1750_cropw",
                         "sigma_v_T_nongs_pre1750_cropw",
                         "n_gs_months_cropw"]], on="iso3", how="left")
              .merge(annual_seasonality[["iso3", "sigma_v_preind"]],
                     on="iso3", how="left")
              .merge(pathways[["iso3", "cluster"]], on="iso3", how="left")
              .merge(abs_lat[["iso3", "abs_lat"]], on="iso3", how="left"))
    # Headline drops empty-GS countries
    headline = df[df["n_gs_months_cropw"] > 0].dropna(
        subset=["sigma_v_T_gs_pre1750_cropw", "sigma_v_P_gs_pre1750_cropw",
                "sigma_v_T_nongs_pre1750_cropw", "abs_lat", "cluster"]).copy()
    pathway_dum = pd.get_dummies(headline["cluster"], prefix="path", drop_first=True)
    X_base = pd.concat([headline[["sigma_v_T_gs_pre1750_cropw",
                                    "sigma_v_P_gs_pre1750_cropw",
                                    "sigma_v_T_nongs_pre1750_cropw",
                                    "abs_lat"]], pathway_dum], axis=1).astype(float)
    X = sm.add_constant(X_base)
    res_gs = sm.OLS(headline["log_pop_growth"], X).fit(cov_type="HC1")
    # Annual comparison on the same headline sample
    X_ann = pd.concat([headline[["sigma_v_preind", "abs_lat"]],
                        pathway_dum], axis=1).astype(float)
    res_ann = sm.OLS(headline["log_pop_growth"],
                     sm.add_constant(X_ann)).fit(cov_type="HC1")

    summary = {
        "annual": {"beta": float(res_ann.params["sigma_v_preind"]),
                    "p":    float(res_ann.pvalues["sigma_v_preind"]),
                    "r2":   float(res_ann.rsquared), "n": int(res_ann.nobs)},
        "gs":     {"beta": float(res_gs.params["sigma_v_T_gs_pre1750_cropw"]),
                    "p":    float(res_gs.pvalues["sigma_v_T_gs_pre1750_cropw"]),
                    "r2":   float(res_gs.rsquared), "n": int(res_gs.nobs)},
        "gs_p":   {"beta": float(res_gs.params["sigma_v_P_gs_pre1750_cropw"]),
                    "p":    float(res_gs.pvalues["sigma_v_P_gs_pre1750_cropw"])},
        "nongs":  {"beta": float(res_gs.params["sigma_v_T_nongs_pre1750_cropw"]),
                    "p":    float(res_gs.pvalues["sigma_v_T_nongs_pre1750_cropw"])},
    }
    out_pq = DATA_DIR / "long_shadow_results_gs.parquet"
    pd.DataFrame([summary]).to_parquet(out_pq, index=False)
    print(f"[gs-headline] wrote {out_pq}")
    return summary
```

- [ ] **Step 3: Wire `run_headline_gs` into the script's `main()` or orchestrator**

The script is consumed by `analysis/paper4_shadow/run_all.py`. Look at how `long_shadow.py` is currently called and add a `run_headline_gs(...)` call alongside the existing cross-section call. If `long_shadow.py` is run directly (has its own `if __name__ == "__main__"`), add to that block:

```python
if __name__ == "__main__":
    from analysis.shared import load_extended_panel, load_pathways  # adapt to actual imports
    modern_panel = load_extended_panel()
    pathways     = load_pathways()
    abs_lat      = modern_panel[["iso3", "centroid_lat"]].drop_duplicates()
    abs_lat["abs_lat"] = abs_lat["centroid_lat"].abs()
    annual_season = pd.read_parquet(DATA_DIR / "country_seasonality_preindustrial.parquet")
    summary = run_headline_gs(modern_panel, pathways, abs_lat, annual_season)
    assert_predictions(summary)
```

(Adapt the import path and helper-function names to whatever `long_shadow.py` currently uses for data loading; the spec is silent on this because it depends on the in-script conventions.)

- [ ] **Step 4: Run the headline cross-section**

```bash
cd /Volumes/BIGDATA/HYDE35
python -m analysis.paper4_shadow.long_shadow
```
Expected: prints regression summary, writes `long_shadow_results_gs.parquet`, prints the three pre-registered prediction lines (PASS/FAIL each).

- [ ] **Step 5: Commit**

```bash
git add analysis/paper4_shadow/long_shadow.py \
        analysis/data/long_shadow_results_gs.parquet
git commit -m "$(cat <<'EOF'
long_shadow: add GS-headline cross-section + non-GS placebo column

Adds run_headline_gs() that loads country_seasonality_gs_preindustrial,
runs the §5 headline with {σ_v^T_GS, σ_v^P_GS, σ_v^T_nonGS} + |lat| +
pathway FE on the empty-GS-dropped sample, and compares against annual
σ_v^T on the same sample. Pre-registered predictions checked at end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Sensitivity tables (empty-GS / short-GS / weighting)

**Files:**
- Modify: `analysis/paper4_shadow/long_shadow.py`
- Output: `analysis/data/long_shadow_sensitivity_gs.parquet`

- [ ] **Step 1: Add sensitivity runner**

In `long_shadow.py`, add:

```python
def run_sensitivities_gs(modern_panel: pd.DataFrame,
                          pathways: pd.DataFrame,
                          abs_lat: pd.DataFrame,
                          annual_seasonality: pd.DataFrame) -> pd.DataFrame:
    """Run the four sensitivity columns called out in spec §5.4."""
    import statsmodels.api as sm

    gs = pd.read_parquet(DATA_DIR / "country_seasonality_gs_preindustrial.parquet")
    p0 = (modern_panel[modern_panel["year"].between(1950, 1960)]
            .groupby("iso3", as_index=False).agg(p0=("pop", "mean")))
    p1 = (modern_panel[modern_panel["year"].between(2015, 2025)]
            .groupby("iso3", as_index=False).agg(p1=("pop", "mean")))
    out = p0.merge(p1, on="iso3").query("p0 > 0 and p1 > 0").copy()
    out["log_pop_growth"] = np.log(out["p1"] / out["p0"])
    out = (out.merge(gs, on="iso3", how="left")
              .merge(annual_seasonality[["iso3", "sigma_v_preind"]],
                     on="iso3", how="left")
              .merge(pathways[["iso3", "cluster"]], on="iso3", how="left")
              .merge(abs_lat[["iso3", "abs_lat"]], on="iso3", how="left"))

    def _fit(d: pd.DataFrame, key: str, label: str) -> dict:
        d = d.dropna(subset=[key, "abs_lat", "cluster"]).copy()
        pathway_dum = pd.get_dummies(d["cluster"], prefix="path", drop_first=True)
        X = sm.add_constant(pd.concat([d[[key, "abs_lat"]], pathway_dum],
                                       axis=1).astype(float))
        r = sm.OLS(d["log_pop_growth"], X).fit(cov_type="HC1")
        return {"spec": label, "key": key, "beta": float(r.params[key]),
                "se": float(r.bse[key]), "p": float(r.pvalues[key]),
                "r2": float(r.rsquared), "n": int(r.nobs)}

    rows = []
    headline = out[out["n_gs_months_cropw"] > 0].copy()
    rows.append(_fit(headline, "sigma_v_T_gs_pre1750_cropw", "headline-drop-empty"))
    rows.append(_fit(out,        "sigma_v_T_gs_pre1750_cropw",
                      "include-empty-w/-annual-fallback"))
    rows.append(_fit(out[out["n_gs_months_cropw"] >= 4],
                      "sigma_v_T_gs_pre1750_cropw", "short-GS-dropped"))
    for w in ("area", "pop"):
        rows.append(_fit(out[out[f"n_gs_months_{w}"] > 0],
                         f"sigma_v_T_gs_pre1750_{w}", f"weighting-{w}"))

    df = pd.DataFrame(rows)
    out_pq = DATA_DIR / "long_shadow_sensitivity_gs.parquet"
    df.to_parquet(out_pq, index=False)
    print(f"[gs-sensitivity]\n{df.to_string(index=False)}\n→ {out_pq}")
    return df
```

Wire into `if __name__ == "__main__":` after the headline call:

```python
    sens = run_sensitivities_gs(modern_panel, pathways, abs_lat, annual_season)
```

- [ ] **Step 2: Run**

```bash
python -m analysis.paper4_shadow.long_shadow
```
Expected: prints sensitivity table with 5 rows; writes parquet.

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/long_shadow.py \
        analysis/data/long_shadow_sensitivity_gs.parquet
git commit -m "$(cat <<'EOF'
long_shadow: GS sensitivity battery (empty / short-GS / weighting)

Five-row sensitivity table: headline (empty-GS dropped), annual-fallback
include, short-GS dropped, area-weighted, pop-weighted. Writes
long_shadow_sensitivity_gs.parquet for the appendix table emitter.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Regenerate cross-section figure with GS measure in panel (a)

**Files:**
- Modify: figure script that produces `analysis/figures/cross_section_final.png`
- Locate via `grep -l "cross_section_final" analysis/paper4_shadow/*.py analysis/*.py`

- [ ] **Step 1: Locate the figure script and inspect**

```bash
cd /Volumes/BIGDATA/HYDE35
grep -l "cross_section_final" analysis/paper4_shadow/*.py analysis/*.py 2>/dev/null
```
Expected: one file (likely `figXY_long_shadow_choropleth.py` or `make_figures.py`).

- [ ] **Step 2: Modify the figure script**

In the identified script, find where it loads `sigma_v_preind` (or equivalent) from `country_seasonality_preindustrial.parquet` and replace the panel-(a) data source with `sigma_v_T_gs_pre1750_cropw` from `country_seasonality_gs_preindustrial.parquet`. Update the panel-(a) title to "Pre-industrial growing-season $\sigma_v^T$, 1421–1750".

If the figure also displays a regression-line annotation with the headline coefficient, update it to read from `long_shadow_results_gs.parquet`.

- [ ] **Step 3: Regenerate**

```bash
python <identified_script>
```
Expected: PNG and PDF outputs written; visual sanity-check by opening `analysis/figures/cross_section_final.png`.

- [ ] **Step 4: Commit**

```bash
git add <figure_script> analysis/figures/cross_section_final.png \
         analysis/figures/cross_section_final.pdf
git commit -m "$(cat <<'EOF'
long_shadow: regenerate cross-section figure with GS σ_v^T in panel (a)

Panel (a) now shows pre-industrial growing-season temperature volatility
(σ_v^T_GS, cropland-weighted, 1421-1750) instead of annual σ_v^T. Panel
(b) modern log pop growth unchanged. Coefficient annotation reads from
long_shadow_results_gs.parquet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase C — Panel regressions

### Task 6: Update Malthusian regressions (preindustrial_malthus, malthusian_extended, conflict-controls)

**Files:**
- Modify: `analysis/paper4_shadow/preindustrial_malthus.py`
- Modify: `analysis/paper4_shadow/preindustrial_malthus_extended.py`
- Modify: `analysis/paper4_shadow/malthusian_extended.py`
- Modify: `analysis/paper4_shadow/malthus_with_conflict_controls.py`
- Input: `analysis/data/country_climate_gs_1421_2025.parquet`

For each file:

- [ ] **Step 1: Locate the climate-merge block**

```bash
grep -n "t_anom\|p_anom\|temp_anom\|prcp_anom\|interval.*temp\|within.*interval\|sigma" \
     analysis/paper4_shadow/preindustrial_malthus.py | head -30
```

- [ ] **Step 2: Substitute interval-mean climate with GS-mean climate**

Replace the existing climate-vector construction with a merge against `country_climate_gs_1421_2025.parquet` on iso3-year, then build interval-aggregates from the GS columns. Specifically:

- Replace `t_anom_c` / `interval_temp_anom` source with `t_gs_anom_cropw` from the GS panel
- Replace `p_anom_mm` / `interval_p_anom` source with `p_gs_anom_cropw`
- Replace within-interval $T$-volatility computed from monthly $T$ with std of `t_gs_mean_cropw` within the interval

Concretely, the merge block becomes:

```python
gs = pd.read_parquet(DATA_DIR / "country_climate_gs_1421_2025.parquet")
gs = gs[["iso3", "year", "t_gs_mean_cropw", "p_gs_mean_cropw",
         "t_gs_anom_cropw", "p_gs_anom_cropw"]]
# Aggregate to HYDE intervals (existing helper, named e.g. `assign_interval`)
gs["interval"] = assign_interval(gs["year"])
interval_climate = gs.groupby(["iso3", "interval"], as_index=False).agg(
    t_gs_anom=("t_gs_anom_cropw", "mean"),
    p_gs_anom=("p_gs_anom_cropw", "mean"),
    t_gs_within_sd=("t_gs_mean_cropw", "std"),
    p_gs_within_sd=("p_gs_mean_cropw", "std"),
)
df = df.merge(interval_climate, on=["iso3", "interval"], how="left")
```

Then update the regression formula to use `t_gs_anom + p_gs_anom + t_gs_within_sd + p_gs_within_sd` in place of the prior climate regressors. Keep the column-renaming downstream so existing table emitters work; or add new emitter calls.

- [ ] **Step 3: Run each script and confirm the new climate columns enter the regression**

```bash
python -m analysis.paper4_shadow.preindustrial_malthus
python -m analysis.paper4_shadow.preindustrial_malthus_extended
python -m analysis.paper4_shadow.malthusian_extended
python -m analysis.paper4_shadow.malthus_with_conflict_controls
```
Expected: each prints regression tables containing the new coefficients; no NameError or KeyError.

- [ ] **Step 4: Commit**

```bash
git add analysis/paper4_shadow/preindustrial_malthus.py \
        analysis/paper4_shadow/preindustrial_malthus_extended.py \
        analysis/paper4_shadow/malthusian_extended.py \
        analysis/paper4_shadow/malthus_with_conflict_controls.py
git commit -m "$(cat <<'EOF'
long_shadow: Malthusian regressions switch climate vector to GS-mean

Interval-mean T anomaly → GS-mean T anomaly; interval-mean P anomaly →
GS-mean P anomaly; within-interval T-volatility → within-interval
GS-mean-T volatility; new within-interval GS-mean-P volatility added.
All four Malthus scripts updated; data source is country_climate_gs_1421_2025
(cropw weighting headline).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update joint VAR battery

**Files:**
- Modify: `analysis/paper4_shadow/joint_landuse_var.py`
- Modify: `analysis/paper4_shadow/joint_var_climate_pathways.py`
- Modify: `analysis/paper4_shadow/joint_var_post1700.py`
- Modify: `analysis/paper4_shadow/joint_var_bootstrap.py`
- Modify: `analysis/paper4_shadow/pathway_irfs.py`

For each file:

- [ ] **Step 1: Apply the climate-vector substitution**

The joint VAR's contemporaneous climate vector is identical in structure to the Malthus regression's. For each of the five files, locate the climate-merge block:

```bash
grep -n "t_anom\|p_anom\|interval.*temp\|within.*interval\|sigma" \
     analysis/paper4_shadow/joint_landuse_var.py | head -20
```

and substitute as in Task 6 Step 2:

```python
gs = pd.read_parquet(DATA_DIR / "country_climate_gs_1421_2025.parquet")
gs = gs[["iso3", "year", "t_gs_mean_cropw", "p_gs_mean_cropw",
         "t_gs_anom_cropw", "p_gs_anom_cropw"]]
gs["interval"] = assign_interval(gs["year"])  # use the existing helper
interval_climate = gs.groupby(["iso3", "interval"], as_index=False).agg(
    t_gs_anom=("t_gs_anom_cropw", "mean"),
    p_gs_anom=("p_gs_anom_cropw", "mean"),
    t_gs_within_sd=("t_gs_mean_cropw", "std"),
    p_gs_within_sd=("p_gs_mean_cropw", "std"),
)
df = df.merge(interval_climate, on=["iso3", "interval"], how="left")
```

Then update each regression's right-hand-side spec: replace `t_anom` (or whatever the file calls the interval-mean $T$) with `t_gs_anom`; `p_anom` with `p_gs_anom`; the within-interval $T$-volatility with `t_gs_within_sd`; and add `p_gs_within_sd` as a new regressor. The VSSI exposure $V_{it}$ and lagged-level terms are unchanged.

- [ ] **Step 2: Run each script**

```bash
python -m analysis.paper4_shadow.joint_landuse_var
python -m analysis.paper4_shadow.joint_var_climate_pathways
python -m analysis.paper4_shadow.joint_var_post1700
python -m analysis.paper4_shadow.joint_var_bootstrap
python -m analysis.paper4_shadow.pathway_irfs
```
Expected: each completes; the VSSI coefficients should be similar to current values (within bootstrap noise), since the volcanic-shock identification doesn't depend on the climate-control substitution.

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/joint_landuse_var.py \
        analysis/paper4_shadow/joint_var_climate_pathways.py \
        analysis/paper4_shadow/joint_var_post1700.py \
        analysis/paper4_shadow/joint_var_bootstrap.py \
        analysis/paper4_shadow/pathway_irfs.py
git commit -m "$(cat <<'EOF'
long_shadow: joint VAR climate controls → GS-mean analogues

All five joint-VAR variants (pooled, climate-pathway-stratified, post-1700,
bootstrap, IRF) substitute interval-mean climate with GS-mean climate in
the contemporaneous control vector. VSSI identification unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Sub-national replications

**Files:**
- Modify: `analysis/paper4_shadow/subnational_long_shadow.py`
- Modify: `analysis/paper4_shadow/subnational_malthus.py`
- Input: `analysis/data/subnational_climate_gs_1421_2025.parquet`, `subnational_seasonality_gs_preindustrial.parquet`

- [ ] **Step 1: Substitute the sub-national cross-section measure**

In `subnational_long_shadow.py`, locate the volatility merge:

```bash
grep -n "sigma_v\|temp_vol\|seasonality_proxy\|country_seasonality" \
     analysis/paper4_shadow/subnational_long_shadow.py | head -20
```

Replace the existing volatility merge with:

```python
gs_cs = pd.read_parquet(DATA_DIR / "subnational_seasonality_gs_preindustrial.parquet")
gs_cs = gs_cs[["sub_id",
               "sigma_v_T_gs_pre1750_pop", "sigma_v_P_gs_pre1750_pop",
               "sigma_v_T_nongs_pre1750_pop", "n_gs_months_pop"]]
df = df.merge(gs_cs, on="sub_id", how="left")
df = df[df["n_gs_months_pop"] > 0]  # empty-GS units dropped from headline
```

Update the regression spec to use `sigma_v_T_gs_pre1750_pop + sigma_v_P_gs_pre1750_pop + sigma_v_T_nongs_pre1750_pop` (plus existing controls and country fixed effects).

- [ ] **Step 2: Substitute the sub-national panel climate vector**

In `subnational_malthus.py`, locate the climate-merge block (same grep pattern as above). Replace with:

```python
gs = pd.read_parquet(DATA_DIR / "subnational_climate_gs_1421_2025.parquet")
gs = gs[["sub_id", "year", "t_gs_mean_pop", "p_gs_mean_pop",
         "t_gs_anom_pop", "p_gs_anom_pop"]]
gs["interval"] = assign_interval(gs["year"])
interval_climate = gs.groupby(["sub_id", "interval"], as_index=False).agg(
    t_gs_anom=("t_gs_anom_pop", "mean"),
    p_gs_anom=("p_gs_anom_pop", "mean"),
    t_gs_within_sd=("t_gs_mean_pop", "std"),
    p_gs_within_sd=("p_gs_mean_pop", "std"),
)
df = df.merge(interval_climate, on=["sub_id", "interval"], how="left")
```

Update the regression to use `t_gs_anom + p_gs_anom + t_gs_within_sd + p_gs_within_sd` in place of the existing interval-mean climate regressors.

- [ ] **Step 2: Run**

```bash
python -m analysis.paper4_shadow.subnational_long_shadow
python -m analysis.paper4_shadow.subnational_malthus
```

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/subnational_long_shadow.py \
        analysis/paper4_shadow/subnational_malthus.py \
        analysis/data/subnational_long_shadow_results.parquet \
        analysis/data/subnational_long_shadow_panel.parquet \
        analysis/data/subnational_malthus_panel.parquet
git commit -m "$(cat <<'EOF'
long_shadow: sub-national replications use GS climate analogues

subnational_long_shadow consumes sigma_v_T_gs_pre1750_pop (+ σ_v^P_GS
and non-GS placebo); subnational_malthus interval climate switches to
GS-mean. Pop-weighting is headline at sub-unit resolution per spec §3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Latitude, placebo-rolling, robustness battery

**Files:**
- Modify: `analysis/paper4_shadow/latitude_controls.py`
- Modify: `analysis/paper4_shadow/placebo_period_and_rolling.py`
- Modify: `analysis/paper4_shadow/robustness.py`
- Modify: `analysis/paper4_shadow/robustness_v2.py`
- Modify: `analysis/paper4_shadow/deep_determinants.py`
- Modify: `analysis/paper4_shadow/deep_determinants_extended.py`

- [ ] **Step 1: Latitude robustness**

In `latitude_controls.py`, find the existing $\sigma_v^T$ block and add three new rows: $\sigma_v^{T,\mathrm{GS}}$, $\sigma_v^{P,\mathrm{GS}}$, $\sigma_v^{T,\overline{\mathrm{GS}}}$. Each row should show coefficients with/without `abs_lat`, matching the existing table format.

- [ ] **Step 2: Placebo-rolling**

In `placebo_period_and_rolling.py`, extend the modern-window placebo to compute $\sigma_v^{T,\mathrm{GS}}$ over 1950–2008. The mask must be recomputed on modern-window climatology (the historical mask is contaminated by sampling overlap). Add this block where the existing modern-window σ_v is computed:

```python
from analysis.paper4_shadow.build_gs_climate import (
    _absolute_levels, _gs_mask, _cross_section, COUNTRY_WEIGHTINGS,
)

# Recompute mask on 1950-2008 climatology under cropw weighting
mod = pd.read_parquet(DATA_DIR / COUNTRY_WEIGHTINGS["cropw"])
abs_df = _absolute_levels(mod, entity_col="iso3")
# Override PRE_WINDOW locally to (1950, 2008) for the placebo
abs_df_modern = abs_df[abs_df["year"].between(1950, 2008)].copy()
modern_clim = abs_df_modern.groupby(["iso3", "month"], as_index=False).agg(
    t_clim=("t_abs", "mean"), p_clim=("p_abs", "mean"))
modern_clim["in_gs"] = ((modern_clim["t_clim"] >= 5)
                        & (modern_clim["t_clim"] <= 30)
                        & (modern_clim["p_clim"] >= 30))
mask_modern = modern_clim[["iso3", "month", "in_gs"]]

merged = abs_df_modern.merge(mask_modern, on=["iso3", "month"])
gs_yr = merged[merged["in_gs"]].groupby(["iso3", "year"], as_index=False).agg(
    t_gs=("t_abs", "mean"))
modern_gs_sigma_v = gs_yr.groupby("iso3", as_index=False).agg(
    sigma_v_T_gs_modern_cropw=("t_gs", "std"))
# Merge into the modern-window placebo's regression frame and run the
# headline-style regression with this column in place of σ_v^T_gs_pre1750
```

For the rolling-window estimator, add GS-mean climate columns alongside the existing climate columns by merging `country_climate_gs_1421_2025.parquet` on iso3-year before the rolling-window loop.

- [ ] **Step 3: Robustness battery**

In `robustness.py` and `robustness_v2.py`, append new rows for the GS measures while keeping all existing annual-σ_v rows (so the comparison is visible).

- [ ] **Step 4: Deep determinants**

`deep_determinants.py` and `deep_determinants_extended.py` use σ_v as one substrate. Add a column variant that swaps it for σ_v_T_GS, alongside the existing annual column.

- [ ] **Step 5: Run each**

```bash
python -m analysis.paper4_shadow.latitude_controls
python -m analysis.paper4_shadow.placebo_period_and_rolling
python -m analysis.paper4_shadow.robustness
python -m analysis.paper4_shadow.robustness_v2
python -m analysis.paper4_shadow.deep_determinants
python -m analysis.paper4_shadow.deep_determinants_extended
```

- [ ] **Step 6: Commit**

```bash
git add analysis/paper4_shadow/latitude_controls.py \
        analysis/paper4_shadow/placebo_period_and_rolling.py \
        analysis/paper4_shadow/robustness.py \
        analysis/paper4_shadow/robustness_v2.py \
        analysis/paper4_shadow/deep_determinants.py \
        analysis/paper4_shadow/deep_determinants_extended.py
git commit -m "$(cat <<'EOF'
long_shadow: robustness battery + latitude + placebo-rolling + deep-determinants
            add GS measures alongside existing annual ones

Each table acquires σ_v^T_GS, σ_v^P_GS, and σ_v^T_nonGS rows. Modern-window
placebo recomputes GS mask on 1950-2008 climatology. Annual-σ_v rows kept
for back-comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Update `run_all.py` orchestrator

**Files:**
- Modify: `analysis/paper4_shadow/run_all.py`

- [ ] **Step 1: Insert `build_gs_climate` step**

Find where `run_all.py` loads data or runs builder steps; insert before the regression cascade:

```python
print(_banner("Build GS climate (country + sub-national)"))
import subprocess
subprocess.run(["python", "-m", "analysis.paper4_shadow.build_gs_climate"],
               check=True)
subprocess.run(["python", "/Volumes/BIGDATA/HYDE35/analysis/paper4_shadow/test_gs_climate_smoke.py"],
               check=True)
```

(Adapt the integration style to whatever pattern `run_all.py` uses for other builder steps.)

- [ ] **Step 2: Run the full orchestrator**

```bash
python -m analysis.paper4_shadow.run_all
```
Expected: builds GS climate, runs all regressions, exits 0. Total runtime ~10-30 minutes depending on bootstrap settings.

- [ ] **Step 3: Commit**

```bash
git add analysis/paper4_shadow/run_all.py
git commit -m "$(cat <<'EOF'
long_shadow: run_all orchestrator includes GS builder + smoke test

GS climate panels built before the regression cascade. Smoke test gates
the cascade — a failed assertion aborts the run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase D — Paper text

### Task 11: §2 panel description + dropped-country footnote

**Files:**
- Modify: `paper/long_shadow.tex` (§2 "A new monthly paleo-economic panel")

- [ ] **Step 1: Locate the productive-months / $\sigma_s^T$ description**

```bash
grep -n "productive-months\|sigma_s\|Prod_{i,t}" paper/long_shadow.tex | head -10
```

- [ ] **Step 2: Add a paragraph after the existing productive-months description**

Insert (around line 65–66, after the description of $\Pi$):

```latex
The productive-month criterion also defines a fixed \emph{growing-season mask}
$\mathrm{GS}_i \equiv \{m : 5{\le}\bar T^{\mathrm{clim}}_{i,m}{\le}30 \,\text{AND}\,
\bar P^{\mathrm{clim}}_{i,m}{\ge}30\,\mathrm{mm}\}$, where the monthly
climatology is the 1421--1750 mean from the cropland-weighted ModE-RA
panel. Throughout the paper, $\sigma_v^{T,\mathrm{GS}}$ and $\sigma_v^{P,\mathrm{GS}}$
denote the standard deviation of the GS-mean of $T$ (respectively $P$)
across years in 1421--1750; $\sigma_v^{T,\overline{\mathrm{GS}}}$ is the
analogous non-GS-months placebo. We exclude $N=\langle\text{count}\rangle$
empty-GS countries from the headline cross-section (listed in
footnote~\ref{fn:empty-gs}); the appendix reports them re-included with
the annual $\sigma_v^T$ substituted.
```

- [ ] **Step 3: Add the dropped-country footnote**

After locating the right footnote anchor in §2, insert:

```latex
\footnote{\label{fn:empty-gs}Empty-GS countries under cropland weighting
($n^{\mathrm{GS}}_i = 0$, 1421--1750 climatology): \texttt{\langle list
from build_gs_climate.py diagnostic output\rangle}. These countries
have no rain-fed productive month and thus no agronomically meaningful
GS measure; the headline cross-section excludes them.}
```

Fill in `\langle...\rangle` with the actual list printed by `build_gs_climate.py` and the actual N from the empty-GS diagnostic.

- [ ] **Step 4: Compile and check cross-references**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex long_shadow.tex && bibtex long_shadow && pdflatex long_shadow.tex && pdflatex long_shadow.tex
```
Expected: no undefined references, footnote renders correctly. Open `long_shadow.pdf` to spot-check.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf
git commit -m "$(cat <<'EOF'
long_shadow §2: define GS mask, σ_v^T_GS / σ_v^P_GS notation, dropped-country footnote

Adds a paragraph after the productive-months description introducing
the fixed GS mask and the GS-restricted volatility / placebo notation
used in later sections. Footnote lists the empty-GS countries dropped
from the headline cross-section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: §3 Malthus + §4.1 joint VAR table updates

**Files:**
- Modify: `paper/long_shadow.tex` (§3 + §4.1)

- [ ] **Step 1: Update §3 equation 'eq:malthus-main' and the empirical analogue**

Find equation `\ref{eq:malthus-main}` and the prose preceding Table `tab:malthus`. Substitute:

- $T_{it}$ → $\bar T^{\mathrm{GS}}_{it}$
- $\sigma_v^T(T_{it})$ → $\sigma_v^{T,\mathrm{GS}}(T_{it})$
- Add $\bar P^{\mathrm{GS}}_{it}$ as $\gamma_P$
- Add $\sigma_v^{P,\mathrm{GS}}$ as $\delta_P$

The equation becomes:

```latex
\Delta \ln P_{it} = \alpha_i + \beta(\tau^\ast) d_{i,t-1}
 + \gamma_T(\tau^\ast) \bar T^{\mathrm{GS}}_{it}
 + \gamma_P(\tau^\ast) \bar P^{\mathrm{GS}}_{it}
 + \delta_T(\tau^\ast) \sigma_v^{T,\mathrm{GS}}_{i,t}
 + \delta_P(\tau^\ast) \sigma_v^{P,\mathrm{GS}}_{i,t}
 + \eta(\tau^\ast) h_{it} + \varepsilon_{it}
\label{eq:malthus-main}
```

Update the prose between this equation and Table `tab:malthus` to reference the new variable names (replace "inter-annual temperature volatility" with "growing-season inter-annual temperature volatility", etc.).

Regenerate Table `tab:malthus` from the updated regression outputs. Locate the table emitter (`grep -n "tab:malthus" paper/long_shadow.tex` and then `grep -l "tab:malthus" analysis/paper4_shadow/`).

- [ ] **Step 2: Update §4.1 joint VAR Tables 'tab:jointvar-pooled' and 'tab:jointvar-stratified'**

Regenerate both tables from the updated joint-VAR outputs (Task 7). Update the caption text in both tables to reflect the GS-mean climate vector.

In the surrounding prose (around `long_shadow.tex:212-265`), change references to "interval-mean temperature" / "interval-mean precipitation" / "within-interval temperature volatility" to their GS-mean analogues.

- [ ] **Step 3: Compile**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex long_shadow.tex && bibtex long_shadow && pdflatex long_shadow.tex && pdflatex long_shadow.tex
```

- [ ] **Step 4: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf \
        analysis/figures/paper4/*.tex 2>/dev/null  # if table emitters write here
git commit -m "$(cat <<'EOF'
long_shadow §3 + §4.1: equation + table updates to GS-mean climate

eq:malthus-main: T → T_GS, σ_v^T → σ_v^T_GS; add P_GS and σ_v^P_GS as
γ_P, δ_P. tab:malthus regenerated from updated Malthus regressions.
tab:jointvar-pooled + tab:jointvar-stratified regenerated; captions
updated to GS-mean. Surrounding prose updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: §5 long-shadow headline rewrite (predictions box + non-GS placebo column)

**Files:**
- Modify: `paper/long_shadow.tex` (§5 "The long shadow on modern outcomes")

- [ ] **Step 1: Insert the pre-registered predictions box at the top of §5**

After the section heading (around `long_shadow.tex:371`), insert:

```latex
\begin{quote}\small
\noindent\textit{Three pre-registered predictions tested below.}
\textbf{P1:} $|\beta(\sigma_v^{T,\mathrm{GS}})| > |\beta(\sigma_v^{T,\mathrm{annual}})|$
 — if the channel is agronomic, the growing-season-restricted measure should
 dominate the annual one on the same sample.
\textbf{P2:} $\beta(\sigma_v^{P,\mathrm{GS}}) < 0$, significant — monsoon-failure
 / drought-year clustering should depress long-run population growth in
 rain-fed systems.
\textbf{P3:} $\beta(\sigma_v^{T,\overline{\mathrm{GS}}}) \approx 0$ — the
 non-GS-months placebo should not predict modern population growth if the
 channel is agronomic.
\end{quote}
```

- [ ] **Step 2: Rewrite the headline paragraph**

The current headline paragraph (around `long_shadow.tex:381`) starts "The pre-industrial climate-pathway dimension...". Update it to lead with the GS coefficient and report all three pre-registered predictions inline. Use the numerical results from `long_shadow_results_gs.parquet`.

Example template (fill in numbers from `long_shadow_results_gs.parquet`):

```latex
The pre-industrial climate-pathway dimension that organised the absorption
of volcanic shocks leaves measurable traces in modern demographic
trajectories. Figure~\ref{fig:long-shadow-map} delivers the bottom-line
answer: countries with more volatile pre-industrial growing-season
temperatures 1421--1750 have systematically lower modern population
growth ($\hat\beta(\sigma_v^{T,\mathrm{GS}}) = \langle\rangle$,
$p < 10^{-\langle\rangle}$, $R^2 = \langle\rangle$, $N=\langle\rangle$
countries with pathway fixed effects, empty-GS countries excluded).
Growing-season precipitation volatility enters with $\hat\beta
(\sigma_v^{P,\mathrm{GS}}) = \langle\rangle$ ($p = \langle\rangle$) ---
\langle confirming~/~not confirming\rangle prediction P2. The non-GS-months
placebo $\sigma_v^{T,\overline{\mathrm{GS}}}$ enters at
$\hat\beta=\langle\rangle$ ($p=\langle\rangle$), \langle consistent~/~at
odds\rangle with prediction P3. Replacing the GS measure with annual
$\sigma_v^T$ on the same sample yields $\hat\beta=\langle\rangle$
($p<10^{-\langle\rangle}$, $R^2=\langle\rangle$), \langle confirming~/~not
confirming\rangle prediction P1.
```

- [ ] **Step 3: Update Table `tab:shadow` to include the non-GS placebo column**

Regenerate Table `tab:shadow` from updated regression outputs. The table should now show, side-by-side: (i) annual $\sigma_v^T$ headline (on empty-GS-dropped sample), (ii) GS $\sigma_v^T$ headline, (iii) GS $\sigma_v^T$ + GS $\sigma_v^P$ + non-GS placebo joint specification.

- [ ] **Step 4: Compile and visually check §5**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex long_shadow.tex && bibtex long_shadow && pdflatex long_shadow.tex && pdflatex long_shadow.tex
```

Open `long_shadow.pdf` to §5 and verify: predictions box renders, headline paragraph reads coherently, Table `tab:shadow` has the non-GS placebo column.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf
git commit -m "$(cat <<'EOF'
long_shadow §5: predictions box + GS-headline + non-GS placebo column

Adds three pre-registered predictions at the top of §5. Headline paragraph
rewritten to lead with σ_v^T_GS and report all three predictions inline.
tab:shadow expanded to a 3-column comparison: annual, GS, GS+P+nonGS-placebo.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: §1 abstract update + appendix sensitivities

**Files:**
- Modify: `paper/long_shadow.tex` (§1 abstract + `app:longshadow-extensions`)

- [ ] **Step 1: Update abstract**

Find the abstract (around `long_shadow.tex:29`). Update the long-shadow sentence:

Old: "Pre-industrial inter-annual climate volatility 1421--1750 is the single strongest cross-country predictor..."

New: "Pre-industrial growing-season inter-annual climate volatility 1421--1750 ($\sigma_v^{T,\mathrm{GS}}$ and $\sigma_v^{P,\mathrm{GS}}$ jointly) is the single strongest cross-country predictor of modern population growth ($R^2 = \langle\rangle$ on the empty-GS-dropped sample of $N=\langle\rangle$ countries with pathway FE). A non-GS-months placebo $\sigma_v^{T,\overline{\mathrm{GS}}}$ enters at zero, locating the channel inside the growing season."

Fill in numbers from `long_shadow_results_gs.parquet`.

- [ ] **Step 2: Add sensitivity tables to `app:longshadow-extensions`**

Locate the existing appendix (`grep -n "app:longshadow-extensions" paper/long_shadow.tex`). Add four new tables there, generated from `long_shadow_sensitivity_gs.parquet`:

1. Empty-GS sensitivity (headline drop vs. annual-fallback include)
2. Short-GS sensitivity ($n^{\mathrm{GS}} \le 3$ dropped)
3. Weighting sensitivity (area vs. pop vs. cropw)
4. Modern-window placebo (GS-$\sigma_v^T$ computed over 1950–2008)

Each table is a simple beta/SE/p/R²/N table. Write a small LaTeX emitter or hand-write from the parquet output.

- [ ] **Step 3: Compile**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex long_shadow.tex && bibtex long_shadow && pdflatex long_shadow.tex && pdflatex long_shadow.tex
```

- [ ] **Step 4: Commit**

```bash
cd /Volumes/BIGDATA/HYDE35
git add paper/long_shadow.tex paper/long_shadow.pdf
git commit -m "$(cat <<'EOF'
long_shadow: abstract + appendix sensitivities

Abstract leads with GS-restricted volatility result. Appendix
app:longshadow-extensions gains four sensitivity tables: empty-GS,
short-GS, weighting (area/pop/cropw), modern-window GS placebo.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase E — Verify

### Task 15: Full re-run + paper compile + pre-registered prediction report

**Files:** none (verification only)

- [ ] **Step 1: Clean re-run of the analysis cascade**

```bash
cd /Volumes/BIGDATA/HYDE35
python -m analysis.paper4_shadow.run_all 2>&1 | tee /tmp/gs_run_all.log
```
Expected: completes with exit 0. Inspect the tail of `/tmp/gs_run_all.log` for the three pre-registered prediction lines (PASS/FAIL each).

- [ ] **Step 2: Re-compile the paper**

```bash
cd /Volumes/BIGDATA/HYDE35/paper
pdflatex long_shadow.tex && bibtex long_shadow && pdflatex long_shadow.tex && pdflatex long_shadow.tex
```
Expected: zero undefined references, zero "?" cross-references. Inspect the `long_shadow.log` for any LaTeX errors.

- [ ] **Step 3: Visual sanity-check the paper**

Open `paper/long_shadow.pdf` and verify:
- Abstract reads coherently with the GS numbers
- §2 paragraph on GS mask + footnote renders
- §3 Malthus equation has the new variables
- §4.1 joint-VAR tables show GS-mean climate captions
- §5 has the predictions box, the GS headline, and the non-GS placebo column in Table `tab:shadow`
- Figure `fig:long-shadow-map` panel (a) shows GS $\sigma_v^T$
- Appendix `app:longshadow-extensions` has the four new sensitivity tables

- [ ] **Step 4: Write a short verification note in the commit message**

```bash
cd /Volumes/BIGDATA/HYDE35
git status
# expected: clean if Tasks 1-14 all committed; no new changes
```

If the run produced an updated figure or table file not yet committed, stage and commit it:

```bash
git add <files>
git commit -m "$(cat <<'EOF'
long_shadow: GS implementation full re-run

Pre-registered prediction outcomes (from /tmp/gs_run_all.log):
  P1 (|β(σ_v^T_GS)| > |β(σ_v^T_annual)|): <PASS|FAIL with numbers>
  P2 (β(σ_v^P_GS) < 0 at p<0.05):         <PASS|FAIL with numbers>
  P3 (non-GS placebo ≈ 0):                <PASS|FAIL with numbers>

Headline cross-section: β=<…>, R²=<…>, N=<…> countries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Report to user**

Print a concise summary of:
- Whether each of P1, P2, P3 passed
- The final headline coefficient and $R^2$
- The list of empty-GS countries dropped
- Any unexpected reversal or attenuation that suggests a narrative rewrite is needed

---

## Notes

- **TDD discipline.** The smoke test (`test_gs_climate_smoke.py`) is the only formal test. Regression "tests" are the pre-registered prediction checks at the end of `long_shadow.py`'s headline run — failing predictions are informative outcomes, not bugs, so do not raise on failure.
- **Commit cadence.** Each task ends in a commit. If a task spans multiple files but logically one change, one commit is fine; if a task naturally splits (e.g., Task 9 has six distinct files and may produce multiple bug-fixes during the regression updates), split into multiple commits.
- **Out-of-scope reminders.** Do not touch `paper/horserace/`, `paper/long_shadow_v52pp_backup.tex`, or any cover letter `.tex`. Do not build a sub-national cropland-weighted ModE-RA panel (deferred per spec §3). Do not change volcanic-event-study scripts (`volcanic_*.py`, `sigl_*.py`, `tambora_*.py`) — they retain their current climate vocabulary per spec "Out of scope".
- **If a prediction fails.** A P1/P2/P3 failure is publishable. Report it as-is in §5 and adjust the surrounding narrative per the spec §7 "Interpretation if false" column. Do not adjust thresholds or weighting to coax a pass.
